#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import math, sys, webbrowser, re, time, os, json, sqlite3, zlib, platform, subprocess, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Lock, Event, Thread

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets import Header, Footer, Input, Label, Static, DataTable, Button
from textual import work

# ---------------- Endpoints / UA ----------------
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "places-tui/2.2 (personal use) https://openstreetmap.org"
ZIP_RE = re.compile(r"^\s*(\d{5})(?:-\d{4})?\s*$")

# ---------------- Cache (SQLite) ----------------
CACHE_DIR = Path(os.path.expanduser("~/.places_tui"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "cache.db"
GEOCODE_TTL_S  = 14 * 24 * 3600
OVERPASS_TTL_S =  1 * 24 * 3600

# SWR windows (serve stale immediately, refresh in background)
OVERPASS_SOFT_S = 15 * 60     # 15 minutes
OVERPASS_HARD_S = 24 * 3600   # 24 hours
GEOCODE_SOFT_S  =  7 * 24 * 3600
GEOCODE_HARD_S  = 30 * 24 * 3600

_db_lock = Lock()
_db_conn: Optional[sqlite3.Connection] = None
def _db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(CACHE_PATH, check_same_thread=False)
        _db_conn.execute("""CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL, t INTEGER NOT NULL)""")
        _db_conn.execute("PRAGMA journal_mode=WAL")
        _db_conn.execute("PRAGMA synchronous=NORMAL")
        _db_conn.commit()
    return _db_conn
def cache_get(key: str, max_age_s: int) -> Optional[bytes]:
    now = int(time.time())
    with _db_lock:
        row = _db().execute("SELECT v,t FROM kv WHERE k=? LIMIT 1", (key,)).fetchone()
    if not row: return None
    v,t = row
    return None if now-int(t) > max_age_s else v
def cache_set(key: str, raw: bytes) -> None:
    now = int(time.time())
    with _db_lock:
        _db().execute("INSERT OR REPLACE INTO kv (k,v,t) VALUES (?,?,?)", (key, raw, now))
        _db().commit()
def cache_get_json(key: str, max_age_s: int) -> Optional[Any]:
    blob = cache_get(key, max_age_s)
    if not blob: return None
    try: return json.loads(zlib.decompress(blob).decode("utf-8"))
    except Exception: return None
def cache_set_json(key: str, obj: Any) -> None:
    cache_set(key, zlib.compress(json.dumps(obj, separators=(",",":")).encode("utf-8"), 6))

# Peek + SWR helpers
def cache_peek_json(key: str) -> tuple[Optional[Any], Optional[int]]:
    with _db_lock:
        row = _db().execute("SELECT v,t FROM kv WHERE k=? LIMIT 1", (key,)).fetchone()
    if not row: return None, None
    v, t = row
    try:
        obj = json.loads(zlib.decompress(v).decode("utf-8"))
    except Exception:
        return None, None
    age = int(time.time()) - int(t)
    return obj, max(0, age)

def swr_json(key: str, soft_ttl_s: int, hard_ttl_s: int, fetch_fn, on_update=None):
    """Return cached data immediately if available; refresh in background when stale."""
    cached, age = cache_peek_json(key)

    def do_refresh_async():
        try:
            fresh = fetch_fn()
            cache_set_json(key, fresh)
            if on_update and fresh:
                on_update(fresh)
        except Exception:
            pass

    # no cache
    if cached is None:
        fresh = fetch_fn()
        cache_set_json(key, fresh)
        return fresh

    # young enough
    if age is None or age <= soft_ttl_s:
        return cached

    # stale but not too old
    if age <= hard_ttl_s:
        Thread(target=do_refresh_async, daemon=True).start()
        return cached

    # too old -> try sync refresh
    try:
        fresh = fetch_fn()
        cache_set_json(key, fresh)
        return fresh
    except Exception:
        return cached

# ---------------- HTTP session ----------------
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retry = Retry(total=2, connect=2, read=2, backoff_factor=0.2,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(["GET","POST"]))
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s
SESSION = _make_session()

# ---------------- Utils ----------------
def haversine_miles(a,b,c,d):
    R=3958.7613; p1,p2=math.radians(a),math.radians(c)
    dphi=math.radians(c-a); dl=math.radians(d-b)
    x=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R*(2*math.asin(math.sqrt(x)))
def fmt_miles(x): return f"{x:.2f} mi" if x<10 else f"{x:.1f} mi"

def build_full_address(tags: Dict[str, str], raw: Optional[Dict[str, Any]] = None) -> str:
    """Prefer addr:* composition; else addr:full; else raw display_name; never fall back to 'name'."""
    parts: List[str] = []
    hn = tags.get("addr:housenumber")
    st = tags.get("addr:street")
    if st and hn:
        parts.append(f"{hn} {st}")
    elif st:
        parts.append(st)
    elif hn:
        parts.append(hn)

    city = tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village")
    state = tags.get("addr:state")
    pc = tags.get("addr:postcode")
    tail = ", ".join([p for p in (city, state) if p])
    if tail: parts.append(tail)
    if pc: parts.append(pc)

    if parts:
        return ", ".join(parts)

    af = tags.get("addr:full")
    if af:
        return af

    if raw:
        if raw.get("address"):
            return str(raw["address"])
        if raw.get("display_name"):
            return str(raw["display_name"])

    return "(no address)"

def pick_name(tags):
    for k in ("name","brand","operator"):
        if tags.get(k): return tags[k]
    for k in ("amenity","shop","leisure"):
        if tags.get(k): return f"({k}: {tags[k]})"
    return "(unnamed)"
def escape_for_overpass_regex(s: str) -> str:
    return s.replace("\\","\\\\").replace('"','\\"').strip()
def simplify_variants(q: str) -> List[str]:
    q1=re.sub(r"\s+"," ",q.replace("’","'")).strip()
    q2=q1.replace("'","")
    return list(dict.fromkeys([q1,q2]))
def bbox_from_center(lat,lon,rad_mi):
    km=rad_mi*1.60934; dlat=km/111.32; dlon=km/(111.32*max(0.01,math.cos(math.radians(lat))))
    return (lon-dlon, lat-dlat, lon+dlon, lat+dlat)

# ---- Clipboard helper (no extra deps; tries pyperclip if present) ----
def _copy_to_clipboard(text: str) -> bool:
    try:
        import pyperclip  # optional
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    sysname = platform.system()
    try:
        if sysname == "Darwin" and shutil.which("pbcopy"):
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(text.encode("utf-8"))
            return p.returncode == 0
        elif sysname == "Windows":
            p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
            p.communicate(text.encode("utf-8"))
            return p.returncode == 0
        else:
            # Linux/BSD: try wl-copy then xclip
            if shutil.which("wl-copy"):
                p = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
                p.communicate(text.encode("utf-8"))
                return p.returncode == 0
            if shutil.which("xclip"):
                p = subprocess.Popen(["xclip","-selection","clipboard"], stdin=subprocess.PIPE)
                p.communicate(text.encode("utf-8"))
                return p.returncode == 0
    except Exception:
        return False
    return False

@dataclass
class Place:
    name: str; address: str; lat: float; lon: float; distance_mi: float; raw: Dict[str,Any]

# ---------------- UI ----------------
class StatusBar(Static):
    def set(self, msg: str)->None: self.update(msg)

class PlacesTUI(App):
    CSS = """
    Screen { layout: vertical; }
    DataTable { height: 1fr; }
    #status { padding: 0 2; color: $text 50%; }
    #inputs { padding: 1 2; }
    """
    BINDINGS = [
        Binding("enter", "search", "Search"),
        Binding("ctrl+s", "search", "Search Now"),
        Binding("/", "focus_query", "Focus Query"),
        Binding("j", "vi_down", show=False), Binding("k", "vi_up", show=False),
        Binding("g", "vi_top", show=False),  Binding("G", "vi_bottom", show=False),
        Binding("o", "open_map", "Open Map"),
        Binding("y", "yank_address", "Copy Address"),
        Binding("Y", "yank_rich", "Copy Name+Addr+Coords"),
        Binding("q", "quit", "Quit"),
        Binding("esc", "cancel_search", "Cancel"),
    ]
    query_text = reactive(""); near_text = reactive(""); radius_mi = reactive(3.0)
    origin: Optional[Tuple[float,float]] = None
    places: List[Place] = []
    _cancel_evt: Event

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Find Places (OpenStreetMap)\nEnter fields and press Enter or click Search")
        with Vertical(id="inputs"):
            yield Label("Query (e.g., coffee shop, mexican restaurant, starbucks):")
            self.query_input = Input(id="query"); yield self.query_input
            yield Label("Near (e.g., Pasadena, CA or 91101):")
            self.near_input  = Input(id="near");  yield self.near_input
            yield Label("Radius mi (0.1–20):")
            self.radius_input= Input(value="3", id="radius"); yield self.radius_input
            with Horizontal():
                self.search_btn = Button("Search",  id="search_btn", variant="primary")
                self.cancel_btn = Button("Cancel",  id="cancel_btn", variant="warning")
                self.warm_btn   = Button("Warm Cache", id="warm_btn", variant="default")
                yield self.search_btn; yield self.cancel_btn; yield self.warm_btn
        self.table = DataTable(zebra_stripes=True)
        self.table.add_columns("#","Name","Address","Distance","Lat","Lon")
        yield self.table
        self.status = StatusBar(id="status"); yield self.status
        yield Footer()

    def on_mount(self): self.query_input.focus()

    # vi movement
    def _row_count(self)->int: return len(self.places)
    def _current_row(self)->int:
        row=getattr(self.table,"cursor_row",None)
        if row is None:
            try: row=self.table.cursor_coordinate[0]
            except Exception: row=0
        if self._row_count()==0: return 0
        return max(0,min(row,self._row_count()-1))
    def _set_row(self,row:int)->None:
        if self._row_count()==0: return
        row=max(0,min(row,self._row_count()-1))
        try:
            if hasattr(self.table,"cursor_coordinate"): self.table.cursor_coordinate=(row,0)
            elif hasattr(self.table,"move_cursor"): self.table.move_cursor(row=row)
        except Exception: pass
    def action_vi_down(self)->None: self._set_row(self._current_row()+1)
    def action_vi_up(self)->None:   self._set_row(self._current_row()-1)
    def action_vi_top(self)->None:  self._set_row(0)
    def action_vi_bottom(self)->None: self._set_row(self._row_count()-1)
    def action_focus_query(self)->None: self.query_input.focus()

    # yank/copy
    def action_yank_address(self) -> None:
        idx = self._current_row()
        if not self.places: return
        addr = self.places[idx].address or ""
        ok = _copy_to_clipboard(addr)
        self.status.set("Address copied to clipboard." if ok else "Copy failed (no clipboard tool found).")

    def action_yank_rich(self) -> None:
        idx = self._current_row()
        if not self.places: return
        p = self.places[idx]
        line = f"{p.name} • {p.address} • {p.lat:.6f},{p.lon:.6f}"
        ok = _copy_to_clipboard(line)
        self.status.set("Name+Address+Coords copied." if ok else "Copy failed (no clipboard tool found).")

    # buttons
    def on_button_pressed(self, event: Button.Pressed):
        if event.button is self.search_btn: self._sync_inputs_and_search()
        elif event.button is self.cancel_btn: self.action_cancel_search()
        elif event.button is self.warm_btn:   self._warm_cache()

    def on_input_submitted(self, event: Input.Submitted): self._sync_inputs_and_search()
    def action_search(self): self._sync_inputs_and_search()

    def action_cancel_search(self):
        try:
            self._cancel_evt.set()
            self.status.set("Canceling current search…")
        except Exception:
            pass

    def _sync_inputs_and_search(self):
        self.query_text = self.query_input.value
        self.near_text  = self.near_input.value
        try:
            v = float((self.radius_input.value or "3").strip()); v=max(0.1,min(20,v))
        except ValueError: v=3
        self.radius_mi = v
        self.table.clear(); self.places = []
        self.status.set("Searching (streaming & SWR)…  Press Esc or ‘Cancel’ to stop.")
        self._cancel_evt = Event()
        self._do_search_stream(self.query_text.strip(), self.near_text.strip(), self.radius_mi, self._cancel_evt)

    # ---------------- STREAMED SEARCH (with SWR) ----------------
    @work(exclusive=True, thread=True)
    def _do_search_stream(self, q: str, near: str, radius_mi: float, cancel_evt: Event):
        if not q or not near:
            self.call_from_thread(self.status.set, "Please enter both Query and Near location."); return
        try:
            lat, lon, _ = geocode_cached_disk_first(near)
        except Exception as e:
            self.call_from_thread(self.status.set, f"Geocoding failed: {e}"); return

        r_small = max(0.25, min(radius_mi, radius_mi*0.5))
        radii_m = [int(r_small*1609.34), int(radius_mi*1609.34)]

        seen = set()
        def push(elems: List[Dict[str,Any]]):
            if cancel_evt.is_set(): return
            new_places: List[Place] = []
            for o in elems:
                oid = (o.get("type",""), o.get("id"))
                if "id" in o and oid in seen: continue
                if "id" in o: seen.add(oid)
                tags = o.get("tags") or {}
                name = pick_name(tags) if tags else o.get("display_name","(unnamed)")
                if "lat" in o and "lon" in o: plat,plon=float(o["lat"]),float(o["lon"])
                elif "center" in o: plat,plon=float(o["center"]["lat"]),float(o["center"]["lon"])
                elif "latlon" in o: plat,plon=o["latlon"]
                else: continue
                addr = build_full_address(tags, o)
                dist = haversine_miles(lat, lon, plat, plon)
                new_places.append(Place(name, addr, plat, plon, dist, o))
            if not new_places: return
            new_places.sort(key=lambda p: p.distance_mi)
            def apply_rows():
                start_len = len(self.places)
                self.places.extend(new_places)
                self.places.sort(key=lambda p: p.distance_mi)
                self.table.clear()
                for i,p in enumerate(self.places,1):
                    self.table.add_row(str(i), p.name, p.address, fmt_miles(p.distance_mi), f"{p.lat:.5f}", f"{p.lon:.5f}")
                if start_len==0 and self.places:
                    try:
                        if hasattr(self.table,"cursor_coordinate"): self.table.cursor_coordinate=(0,0)
                    except Exception: pass
            self.call_from_thread(apply_rows)

        # PHASES (each with short timeouts & early exits) -------------------
        phases: List[Tuple[str, List[str], int]] = []

        # Cuisine-first when generic (pho, pizza, sushi, etc.)
        if is_generic_food_query(q):
            for R in radii_m:
                phases.append(("Cuisine nodes", build_cuisine_blocks("node", q), R))
            for R in radii_m:
                phases.append(("Cuisine ways/relations", build_cuisine_blocks("way", q) + build_cuisine_blocks("relation", q), R))

        def exact_blocks(k):  # k: node/way/relation
            blocks=[]
            for qv in simplify_variants(q):
                rx=f'"^{escape_for_overpass_regex(qv)}$",i'
                blocks += [f'{k}(around:{{R}},{{lat}},{{lon}})["name"~{rx}];',
                           f'{k}(around:{{R}},{{lat}},{{lon}})["brand"~{rx}];',
                           f'{k}(around:{{R}},{{lat}},{{lon}})["operator"~{rx}];']
            return blocks
        def regex_blocks(k):
            blocks=[]
            for qv in simplify_variants(q):
                rx=f'"{escape_for_overpass_regex(qv)}",i'
                blocks += [f'{k}(around:{{R}},{{lat}},{{lon}})["name"~{rx}];',
                           f'{k}(around:{{R}},{{lat}},{{lon}})["brand"~{rx}];']
            return blocks

        for R in radii_m:
            phases.append(("Exact nodes",       exact_blocks("node"), R))
        for R in radii_m:
            phases.append(("Exact ways/relations", exact_blocks("way")+exact_blocks("relation"), R))
        for R in radii_m:
            phases.append(("Regex nodes",       regex_blocks("node"), R))
        for R in radii_m:
            phases.append(("Regex ways/relations", regex_blocks("way")+regex_blocks("relation"), R))

        # Run phases; stream each block; use SWR so cached rows appear instantly and background refresh updates UI.
        for title, blocks, R in phases:
            if cancel_evt.is_set():
                break
            self.call_from_thread(self.status.set, f"{title} (R≈{int(R/1609.34)}mi)…")
            for b in blocks:
                if cancel_evt.is_set():
                    break
                ql = f"[out:json][timeout:12];({b.format(R=R, lat=lat, lon=lon)});out center tags 80;"
                key = "ovp:" + str(hash(ql))
                def _on_update(fresh):
                    if cancel_evt.is_set(): return
                    push(fresh)
                elems = swr_json(
                    key=key,
                    soft_ttl_s=OVERPASS_SOFT_S,
                    hard_ttl_s=OVERPASS_HARD_S,
                    fetch_fn=lambda ql=ql: call_overpass_parallel(ql, per_req_timeout=12),
                    on_update=_on_update
                )
                if elems:
                    push(elems)

        # Fallback Nominatim (bounded)
        if not cancel_evt.is_set() and len(self.places) == 0:
            self.call_from_thread(self.status.set, "Nominatim fallback…")
            try:
                nomi_q = f"{q} restaurant" if is_generic_food_query(q) and "restaurant" not in q.lower() else q
                push(nominatim_bounded_cached(nomi_q, lat, lon, radius_mi))
            except Exception:
                pass

        if not cancel_evt.is_set():
            self.call_from_thread(self.status.set, f"Done. Found {len(self.places)} place(s). Press 'o' to open.  Press 'y' to copy address.")

    # ---------------- Cache warmer ----------------
    def _warm_cache(self):
        qnear = self.near_input.value.strip()
        if not qnear:
            self.status.set("Enter a Near (city or ZIP) first, then click Warm Cache."); return
        try:
            lat, lon, _ = geocode_cached_disk_first(qnear)
        except Exception as e:
            self.status.set(f"Geocoding failed: {e}"); return
        self.status.set("Warming cache around Near (3 mi)…")
        topics = ["coffee shop","restaurant","fast food","supermarket","pharmacy","gas station","bakery","cafe","bar","pho","vietnamese","sushi","pizza"]
        for topic in topics:
            try:
                R = int(3 * 1609.34)
                variants = simplify_variants(topic); blocks=[]
                for qv in variants:
                    rx=f'"{escape_for_overpass_regex(qv)}",i'
                    blocks += [f'node(around:{R},{lat},{lon})["name"~{rx}];',
                               f'node(around:{R},{lat},{lon})["brand"~{rx}];']
                ql = f"[out:json][timeout:12];(" + "".join(blocks) + ");out center tags 80;"
                key = "ovp:" + str(hash(ql))
                _ = swr_json(
                    key=key,
                    soft_ttl_s=OVERPASS_SOFT_S,
                    hard_ttl_s=OVERPASS_HARD_S,
                    fetch_fn=lambda ql=ql: call_overpass_parallel(ql, per_req_timeout=12),
                    on_update=None
                )
            except Exception:
                continue
        self.status.set("Cache warmed for common categories near your location.")

    # ---------------- Table / Map ----------------
    def _render_table(self):
        self.table.clear()
        for i,p in enumerate(self.places,1):
            self.table.add_row(str(i), p.name, p.address, fmt_miles(p.distance_mi), f"{p.lat:.5f}", f"{p.lon:.5f}")
        try:
            if self.places and hasattr(self.table,"cursor_coordinate"): self.table.cursor_coordinate=(0,0)
        except Exception: pass

    def action_open_map(self):
        row=getattr(self.table,"cursor_row",None)
        if row is None:
            try: row=self.table.cursor_coordinate[0]
            except Exception: row=0
        if row<0 or row>=len(self.places): return
        p=self.places[row]
        url=f"https://www.openstreetmap.org/?mlat={p.lat:.6f}&mlon={p.lon:.6f}#map=18/{p.lat:.6f}/{p.lon:.6f}"
        webbrowser.open(url); self.status.set(f"Opened {p.name} in browser.")

# ---------------- Network + Disk-cached helpers ----------------
@lru_cache(maxsize=256)
def geocode_cached_mem(near: str) -> Tuple[float,float,str]:
    return _geocode_network(near)

def geocode_cached_disk_first(near: str) -> Tuple[float,float,str]:
    key=f"geo:{near.strip().lower()}"
    def fetcher():
        lat,lon,name=_geocode_network(near)
        return {"lat":lat,"lon":lon,"name":name}
    obj = swr_json(key, GEOCODE_SOFT_S, GEOCODE_HARD_S, fetcher, on_update=None)
    return float(obj["lat"]), float(obj["lon"]), str(obj["name"])

def _geocode_network(near: str) -> Tuple[float,float,str]:
    s=near.strip(); m=ZIP_RE.match(s)
    params={"format":"json","q":(m.group(1) if m else s),"limit":1,"addressdetails":1}
    if m: params["countrycodes"]="us"
    r=SESSION.get(NOMINATIM_URL, params=params, timeout=10)
    r.raise_for_status(); data=r.json()
    if not data: raise ValueError("No match for location.")
    d=data[0]; return float(d["lat"]), float(d["lon"]), d.get("display_name", s)

def call_overpass_parallel(ql: str, per_req_timeout: int = 15) -> List[Dict[str,Any]]:
    def post(url: str):
        resp=SESSION.post(url, data={"data": ql}, timeout=per_req_timeout)
        resp.raise_for_status(); return resp.json().get("elements", [])
    with ThreadPoolExecutor(max_workers=min(3,len(OVERPASS_ENDPOINTS))) as ex:
        futs={ex.submit(post,u):u for u in OVERPASS_ENDPOINTS}
        for fut in as_completed(futs):
            try: return fut.result()
            except Exception: continue
    return []

def call_overpass_parallel_cached(ql: str, cancel_evt: Optional[Event], per_req_timeout: int = 15) -> List[Dict[str,Any]]:
    """Kept for compatibility; now delegates to SWR."""
    key="ovp:"+str(hash(ql))
    def fetcher():
        if cancel_evt and cancel_evt.is_set(): return []
        return call_overpass_parallel(ql, per_req_timeout=per_req_timeout)
    return swr_json(key, OVERPASS_SOFT_S, OVERPASS_HARD_S, fetcher, on_update=None)

def nominatim_bounded_cached(query: str, lat: float, lon: float, radius_mi: float) -> List[Dict[str,Any]]:
    sig=f"nomi:{query}|{lat:.6f}|{lon:.6f}|{radius_mi:.3f}"
    hit=cache_get_json(sig, OVERPASS_TTL_S)
    if isinstance(hit,list): return hit
    left,bottom,right,top=bbox_from_center(lat,lon,radius_mi)
    params={"format":"json","q":query,"viewbox":f"{left},{top},{right},{bottom}","bounded":1,"limit":40,"addressdetails":1,"countrycodes":"us"}
    r=SESSION.get(NOMINATIM_URL, params=params, timeout=12); r.raise_for_status()
    data=r.json(); out=[]
    for d in data:
        try: plat,plon=float(d["lat"]),float(d["lon"])
        except Exception: continue
        out.append({"display_name":d.get("display_name","(unnamed)"),"lat":plat,"lon":plon,"address":d.get("display_name",""),"tags":{},"latlon":(plat,plon)})
    cache_set_json(sig,out); return out

# Generic-ish food queries that should use cuisine/category fast-path
GENERIC_TERMS = {
    "pho","vietnamese","ramen","sushi","pizza","tacos","taco","burger","bbq",
    "bar","pub","cafe","coffee","bakery","donuts","boba","dim sum","noodles",
    "kebab","korean","thai","indian","mexican","ethiopian","lebanese","falafel",
    "shawarma","hotpot","dumpling","poke","sandwich","bagel","steak","seafood",
    "breakfast","brunch","diner","halal","kosher"
}
def is_generic_food_query(q: str) -> bool:
    s = q.strip().lower()
    return s in GENERIC_TERMS or (len(s.split()) == 1 and s in GENERIC_TERMS)

def cuisine_regex_for(q: str) -> str:
    s = q.strip().lower()
    if s == "pho":
        return r'"(?i)(^|;|\s)(pho|vietnamese)(;|\s|$)"'
    esc = escape_for_overpass_regex(s)
    return fr'"(?i)(^|;|\s){esc}(;|\s|$)"'

def build_cuisine_blocks(obj_type: str, q: str) -> List[str]:
    rx_cui = cuisine_regex_for(q)
    rx_nm  = f'"{escape_for_overpass_regex(q)}",i'
    return [
        f'{obj_type}(around:{{R}},{{lat}},{{lon}})["amenity"="restaurant"]["cuisine"~{rx_cui}];',
        f'{obj_type}(around:{{R}},{{lat}},{{lon}})["amenity"="fast_food"]["cuisine"~{rx_cui}];',
        f'{obj_type}(around:{{R}},{{lat}},{{lon}})["name"~{rx_nm}];',
        f'{obj_type}(around:{{R}},{{lat}},{{lon}})["brand"~{rx_nm}];',
    ]

# ---------------- Exact/Regex builders (used in phases) ----------------
def _blocks_exact(k: str, q: str) -> List[str]:
    blocks=[];
    for qv in simplify_variants(q):
        rx=f'"^{escape_for_overpass_regex(qv)}$",i'
        blocks += [f'{k}(around:{{R}},{{lat}},{{lon}})["name"~{rx}];',
                   f'{k}(around:{{R}},{{lat}},{{lon}})["brand"~{rx}];',
                   f'{k}(around:{{R}},{{lat}},{{lon}})["operator"~{rx}];']
    return blocks
def _blocks_regex(k: str, q: str) -> List[str]:
    blocks=[]
    for qv in simplify_variants(q):
        rx=f'"{escape_for_overpass_regex(qv)}",i'
        blocks += [f'{k}(around:{{R}},{{lat}},{{lon}})["name"~{rx}];',
                   f'{k}(around:{{R}},{{lat}},{{lon}})["brand"~{rx}];']
    return blocks

# ---------------- Entrypoint ----------------
def main():
    import argparse, os, traceback

    parser = argparse.ArgumentParser(prog="places_tui", add_help=True)
    parser.add_argument("--dev", action="store_true", help="Enable Textual devtools")
    parser.add_argument("--clear-cache", action="store_true", help="Delete the on-disk cache and exit")
    parser.add_argument("--no-color", action="store_true", help="Force a dumb TERM (debug)")
    args = parser.parse_args()

    if args.clear_cache:
        try:
            if CACHE_PATH.exists():
                CACHE_PATH.unlink()
            print(f"Cleared cache at {CACHE_PATH}")
        except Exception as e:
            print(f"Could not clear cache: {e}")
        return

    os.environ.setdefault("TERM", "xterm-256color")
    if args.no_color:
        os.environ["TERM"] = "dumb"
    if args.dev:
        os.environ["TEXTUAL_DEVTOOLS"] = "1"

    try:
        app = PlacesTUI()
        app.run()
    except Exception:
        traceback.print_exc()
        input("\nCrash captured. Press Enter to exit.")

if __name__ == "__main__":
    main()

