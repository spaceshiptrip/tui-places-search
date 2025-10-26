#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import sys
import textwrap
import webbrowser
import re
import time
import os
import json
import sqlite3
import zlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets import Header, Footer, Input, Label, Static, DataTable, Button
from textual import work

# ---- Endpoints / UA ----
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "places-tui/1.8 (personal use) https://openstreetmap.org"

ZIP_RE = re.compile(r"^\s*(\d{5})(?:-\d{4})?\s*$")  # capture 5-digit ZIP

# ---- Cache settings ----
CACHE_DIR = Path(os.path.expanduser("~/.places_tui"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "cache.db"

GEOCODE_TTL_S = 14 * 24 * 3600     # 14 days
OVERPASS_TTL_S = 1 * 24 * 3600     # 1 day

_db_lock = Lock()
_db_conn: Optional[sqlite3.Connection] = None

def _db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(CACHE_PATH, check_same_thread=False)
        _db_conn.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v BLOB NOT NULL,
                t INTEGER NOT NULL
            )
        """)
        _db_conn.execute("PRAGMA journal_mode=WAL")
        _db_conn.execute("PRAGMA synchronous=NORMAL")
        _db_conn.commit()
    return _db_conn

def cache_get(key: str, max_age_s: int) -> Optional[bytes]:
    now = int(time.time())
    with _db_lock:
        cur = _db().execute("SELECT v, t FROM kv WHERE k=? LIMIT 1", (key,))
        row = cur.fetchone()
    if not row:
        return None
    v, t = row
    if now - int(t) > max_age_s:
        return None
    return v

def cache_set(key: str, raw_bytes: bytes) -> None:
    now = int(time.time())
    with _db_lock:
        _db().execute("INSERT OR REPLACE INTO kv (k, v, t) VALUES (?, ?, ?)", (key, raw_bytes, now))
        _db().commit()

def cache_set_json(key: str, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    cache_set(key, zlib.compress(data, level=6))

def cache_get_json(key: str, max_age_s: int) -> Optional[Any]:
    blob = cache_get(key, max_age_s)
    if blob is None:
        return None
    try:
        data = zlib.decompress(blob)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None

# ---- HTTP session with keep-alive & retry ----
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retry = Retry(
        total=2, connect=2, read=2, backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET","POST"])
    )
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = _make_session()

# ---- Utils ----
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

def fmt_miles(distance_mi):
    return f"{distance_mi:.2f} mi" if distance_mi < 10 else f"{distance_mi:.1f} mi"

def make_address(tags):
    parts = []
    if hn := tags.get("addr:housenumber"):
        parts.append(hn)
    if st := tags.get("addr:street"):
        parts[-1] = f"{parts[-1]} {st}" if parts else st
    city = tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village")
    state = tags.get("addr:state")
    postcode = tags.get("addr:postcode")
    tail = ", ".join([p for p in [city, state] if p])
    if tail: parts.append(tail)
    if postcode: parts.append(postcode)
    return ", ".join(parts) if parts else tags.get("name", "(no address)")

def pick_name(tags):
    for k in ("name","brand","operator"):
        if tags.get(k): return tags[k]
    for k in ("amenity","shop","leisure"):
        if tags.get(k): return f"({k}: {tags[k]})"
    return "(unnamed)"

def escape_for_overpass_regex(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').strip()

def simplify_variants(q: str) -> List[str]:
    q1 = q.replace("’", "'").strip()
    q2 = q1.replace("'", "")
    # also collapse multiple spaces
    q3 = re.sub(r"\s+", " ", q1)
    return list(dict.fromkeys([q3, q2]))

def bbox_from_center(lat: float, lon: float, radius_mi: float) -> Tuple[float,float,float,float]:
    km = radius_mi * 1.60934
    dlat = km / 111.32
    dlon = km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

@dataclass
class Place:
    name: str
    address: str
    lat: float
    lon: float
    distance_mi: float
    raw: Dict[str, Any]

# ---- UI ----
class StatusBar(Static):
    def set(self, msg: str) -> None:
        self.update(msg)

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
        Binding("j", "vi_down", show=False),
        Binding("k", "vi_up", show=False),
        Binding("g", "vi_top", show=False),
        Binding("G", "vi_bottom", show=False),
        Binding("o", "open_map", "Open Map"),
        Binding("q", "quit", "Quit"),
    ]

    query_text = reactive("")
    near_text = reactive("")
    radius_mi = reactive(3.0)
    origin: Optional[Tuple[float, float]] = None
    places: List[Place] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Find Places (OpenStreetMap)\nEnter fields and press Enter or click Search")

        with Vertical(id="inputs"):
            yield Label("Query (e.g., coffee shop, mexican restaurant, chili's):")
            self.query_input = Input(id="query")
            yield self.query_input

            yield Label("Near (e.g., Pasadena, CA or 91101):")
            self.near_input = Input(id="near")
            yield self.near_input

            yield Label("Radius mi (0.1–20):")
            self.radius_input = Input(value="3", id="radius")
            yield self.radius_input

            with Horizontal():
                self.search_btn = Button("Search", id="search_btn", variant="primary")
                yield self.search_btn

        self.table = DataTable(zebra_stripes=True)
        self.table.add_columns("#", "Name", "Address", "Distance", "Lat", "Lon")
        yield self.table
        self.status = StatusBar(id="status")
        yield self.status
        yield Footer()

    def on_mount(self):
        self.query_input.focus()

    # vi helpers
    def _row_count(self) -> int:
        return len(self.places)

    def _current_row(self) -> int:
        row = getattr(self.table, "cursor_row", None)
        if row is None:
            try:
                row = self.table.cursor_coordinate[0]
            except Exception:
                row = 0
        if self._row_count() == 0:
            return 0
        return max(0, min(row, self._row_count()-1))

    def _set_row(self, row: int) -> None:
        if self._row_count() == 0:
            return
        row = max(0, min(row, self._row_count()-1))
        try:
            if hasattr(self.table, "cursor_coordinate"):
                self.table.cursor_coordinate = (row, 0)
            elif hasattr(self.table, "move_cursor"):
                self.table.move_cursor(row=row)
        except Exception:
            pass

    # vi actions
    def action_vi_down(self) -> None: self._set_row(self._current_row() + 1)
    def action_vi_up(self) -> None: self._set_row(self._current_row() - 1)
    def action_vi_top(self) -> None: self._set_row(0)
    def action_vi_bottom(self) -> None: self._set_row(self._row_count() - 1)
    def action_focus_query(self) -> None: self.query_input.focus()

    # events
    def on_button_pressed(self, event: Button.Pressed):
        if event.button is self.search_btn:
            self._sync_inputs_and_search()

    def on_input_submitted(self, event: Input.Submitted):
        self._sync_inputs_and_search()

    def action_search(self):
        self._sync_inputs_and_search()

    def _sync_inputs_and_search(self):
        self.query_text = self.query_input.value
        self.near_text = self.near_input.value
        try:
            v = float((self.radius_input.value or "3").strip())
            v = max(0.1, min(20, v))
        except ValueError:
            v = 3
        self.radius_mi = v
        self.status.set("Searching…")
        self._do_search(self.query_text.strip(), self.near_text.strip(), self.radius_mi)

    @work(exclusive=True, thread=True)
    def _do_search(self, q: str, near: str, radius_mi: float):
        if not q or not near:
            self.call_from_thread(self.status.set, "Please enter both Query and Near location.")
            return
        try:
            lat, lon, display_name = geocode_cached_disk_first(near)
        except Exception as e:
            self.call_from_thread(self.status.set, f"Geocoding failed: {e}")
            return

        # Progressive radius: start smaller, then expand if needed
        r_small = max(0.25, min(radius_mi, radius_mi * 0.5))
        radii_m = [int(r_small * 1609.34), int(radius_mi * 1609.34)]

        results: List[Dict[str, Any]] = []
        # 1) Exact anchored brand/name (fast path)
        for r in radii_m:
            try:
                results = overpass_cached(self._overpass_exact_first, q, lat, lon, r, 80)
                if results:
                    break
            except Exception:
                pass

        # 2) If still few results, broaden to lightweight regex over name/brand only
        if len(results) < 8:
            for r in radii_m:
                try:
                    broadened = overpass_cached(self._overpass_name_brand_regex, q, lat, lon, r, 120)
                    # merge without dupes
                    bykey = {(e.get("type",""), e.get("id")) for e in results if "id" in e}
                    for e in broadened:
                        k = (e.get("type",""), e.get("id"))
                        if "id" in e and k not in bykey:
                            results.append(e); bykey.add(k)
                    if len(results) >= 8:
                        break
                except Exception:
                    pass

        # 3) Nominatim bounded fallback only if we still have nothing (also cached)
        if not results:
            try:
                results = nominatim_bounded_cached(q, lat, lon, radius_mi)
            except Exception:
                results = []

        places: List[Place] = []
        for o in results:
            tags = o.get("tags") or {}
            name = pick_name(tags) if tags else o.get("display_name", "(unnamed)")
            if "lat" in o and "lon" in o:
                plat, plon = float(o["lat"]), float(o["lon"])
            elif "center" in o:
                plat, plon = float(o["center"]["lat"]), float(o["center"]["lon"])
            elif "latlon" in o:
                plat, plon = o["latlon"]
            else:
                continue
            addr = make_address(tags) if tags else o.get("address", "(no address)")
            dist = haversine_miles(lat, lon, plat, plon)
            places.append(Place(name, addr, plat, plon, dist, o))

        places.sort(key=lambda p: p.distance_mi)

        def apply():
            self.places = places[:100]
            self._render_table()
            msg = f"Found {len(self.places)} place(s) near {near}." if self.places else "No results."
            self.status.set(msg + "  Use ↑/↓ (or j/k), g/G for top/bottom, 'o' to open.")
        self.call_from_thread(apply)

    def _render_table(self):
        self.table.clear()
        for i, p in enumerate(self.places, 1):
            self.table.add_row(str(i), p.name, p.address, fmt_miles(p.distance_mi), f"{p.lat:.5f}", f"{p.lon:.5f}")
        try:
            if self.places and hasattr(self.table, "cursor_coordinate"):
                self.table.cursor_coordinate = (0, 0)
        except Exception:
            pass

    def action_open_map(self):
        row = getattr(self.table, "cursor_row", None)
        if row is None:
            try:
                row = self.table.cursor_coordinate[0]
            except Exception:
                row = 0
        if row < 0 or row >= len(self.places):
            return
        p = self.places[row]
        url = f"https://www.openstreetmap.org/?mlat={p.lat:.6f}&mlon={p.lon:.6f}#map=18/{p.lat:.6f}/{p.lon:.6f}"
        webbrowser.open(url)
        self.status.set(f"Opened {p.name} in browser.")

    # ---- network helpers ----
    def _overpass_exact_first(self, query, lat, lon, radius_m, limit=80) -> List[Dict[str, Any]]:
        """Try exact brand/name/oper match with anchored regex (fast + low cardinality)."""
        variants = simplify_variants(query)
        blocks: List[str] = []
        for qv in variants:
            q = escape_for_overpass_regex(qv)
            rx = f'"^{q}$",i'   # exact, case-insensitive
            blocks += [
                f'node(around:{radius_m},{lat},{lon})["name"~{rx}];',
                f'way(around:{radius_m},{lat},{lon})["name"~{rx}];',
                f'relation(around:{radius_m},{lat},{lon})["name"~{rx}];',
                f'node(around:{radius_m},{lat},{lon})["brand"~{rx}];',
                f'way(around:{radius_m},{lat},{lon})["brand"~{rx}];',
                f'relation(around:{radius_m},{lat},{lon})["brand"~{rx}];',
                f'node(around:{radius_m},{lat},{lon})["operator"~{rx}];',
                f'way(around:{radius_m},{lat},{lon})["operator"~{rx}];',
                f'relation(around:{radius_m},{lat},{lon})["operator"~{rx}];',
            ]
        ql = f"[out:json][timeout:20];({''.join(blocks)});out center tags {int(limit)};"
        return call_overpass_parallel_cached(ql)

    def _overpass_name_brand_regex(self, query, lat, lon, radius_m, limit=120) -> List[Dict[str, Any]]:
        """Broader but still lightweight: only `name` and `brand` regex (skip operator/cuisine to reduce work)."""
        variants = simplify_variants(query)
        blocks: List[str] = []
        for qv in variants:
            q = escape_for_overpass_regex(qv)
            rx = f'"{q}",i'
            blocks += [
                f'node(around:{radius_m},{lat},{lon})["name"~{rx}];',
                f'way(around:{radius_m},{lat},{lon})["name"~{rx}];',
                f'relation(around:{radius_m},{lat},{lon})["name"~{rx}];',
                f'node(around:{radius_m},{lat},{lon})["brand"~{rx}];',
                f'way(around:{radius_m},{lat},{lon})["brand"~{rx}];',
                f'relation(around:{radius_m},{lat},{lon})["brand"~{rx}];',
            ]
        ql = f"[out:json][timeout:25];({''.join(blocks)});out center tags {int(limit)};"
        return call_overpass_parallel_cached(ql)

    def _nominatim_bounded(self, query: str, lat: float, lon: float, radius_mi: float) -> List[Dict[str, Any]]:
        left, bottom, right, top = bbox_from_center(lat, lon, radius_mi)
        params = {
            "format": "json",
            "q": query,
            "viewbox": f"{left},{top},{right},{bottom}",
            "bounded": 1,
            "limit": 40,
            "addressdetails": 1,
            "countrycodes": "us"
        }
        r = SESSION.get(NOMINATIM_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out = []
        for d in data:
            try:
                plat, plon = float(d["lat"]), float(d["lon"])
            except Exception:
                continue
            out.append({
                "display_name": d.get("display_name", "(unnamed)"),
                "lat": plat,
                "lon": plon,
                "address": d.get("display_name", ""),
                "tags": {},
                "latlon": (plat, plon),
            })
        return out

# ---- shared helpers (module level so we can cache them) ----
@lru_cache(maxsize=256)
def geocode_cached_mem(near: str) -> Tuple[float, float, str]:
    """In-memory small LRU (still used after disk cache to avoid JSON decode repeatedly)."""
    return _geocode_network(near)

def geocode_cached_disk_first(near: str) -> Tuple[float, float, str]:
    """Disk-first cache for geocoding."""
    key = f"geo:{near.strip().lower()}"
    hit = cache_get_json(key, GEOCODE_TTL_S)
    if hit and isinstance(hit, dict) and {"lat","lon","name"} <= hit.keys():
        return float(hit["lat"]), float(hit["lon"]), str(hit["name"])
    # Miss → network (via small LRU helper), then store
    lat, lon, name = geocode_cached_mem(near)
    cache_set_json(key, {"lat": lat, "lon": lon, "name": name})
    return lat, lon, name

def _geocode_network(near: str) -> Tuple[float, float, str]:
    s = near.strip()
    m = ZIP_RE.match(s)
    if m:
        zip5 = m.group(1)
        params = {"format": "json", "q": zip5, "countrycodes": "us", "limit": 1, "addressdetails": 1}
    else:
        params = {"format": "json", "q": s, "limit": 1, "addressdetails": 1}
    r = SESSION.get(NOMINATIM_URL, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("No match for location.")
    d = data[0]
    return float(d["lat"]), float(d["lon"]), d.get("display_name", s)

def call_overpass_parallel(ql: str) -> List[Dict[str, Any]]:
    """POST the same query to multiple Overpass mirrors **in parallel**; return the first success."""
    def post(url: str):
        resp = SESSION.post(url, data={"data": ql}, timeout=20)
        resp.raise_for_status()
        return resp.json().get("elements", [])
    with ThreadPoolExecutor(max_workers=min(3, len(OVERPASS_ENDPOINTS))) as ex:
        futures = {ex.submit(post, url): url for url in OVERPASS_ENDPOINTS}
        for fut in as_completed(futures):
            try:
                return fut.result()
            except Exception:
                continue
    return []

def call_overpass_parallel_cached(ql: str) -> List[Dict[str, Any]]:
    """Same as above, but with a disk cache keyed by the full Overpass QL."""
    key = "ovp:" + str(hash(ql))  # compact key, hash is fine (collisions practically negligible here)
    hit = cache_get_json(key, OVERPASS_TTL_S)
    if hit is not None and isinstance(hit, list):
        return hit
    elems = call_overpass_parallel(ql)
    cache_set_json(key, elems)
    return elems

def overpass_cached(fn, q: str, lat: float, lon: float, radius_m: int, limit: int) -> List[Dict[str, Any]]:
    """Cache wrapper for the two Overpass builders (exact / name-brand regex)."""
    # key must capture query + parameters + which function (different QL)
    sig = f"{fn.__name__}|{q}|{lat:.6f}|{lon:.6f}|{radius_m}|{limit}"
    key = "ovp:" + str(hash(sig))
    hit = cache_get_json(key, OVERPASS_TTL_S)
    if hit is not None and isinstance(hit, list):
        return hit
    elems = fn(q, lat, lon, radius_m, limit)  # will itself call call_overpass_parallel_cached
    cache_set_json(key, elems)
    return elems

def nominatim_bounded_cached(query: str, lat: float, lon: float, radius_mi: float) -> List[Dict[str, Any]]:
    sig = f"nomi:{query}|{lat:.6f}|{lon:.6f}|{radius_mi:.3f}"
    hit = cache_get_json(sig, OVERPASS_TTL_S)
    if hit is not None and isinstance(hit, list):
        return hit
    # network
    left, bottom, right, top = bbox_from_center(lat, lon, radius_mi)
    params = {
        "format": "json",
        "q": query,
        "viewbox": f"{left},{top},{right},{bottom}",
        "bounded": 1,
        "limit": 40,
        "addressdetails": 1,
        "countrycodes": "us"
    }
    r = SESSION.get(NOMINATIM_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    out = []
    for d in data:
        try:
            plat, plon = float(d["lat"]), float(d["lon"])
        except Exception:
            continue
        out.append({
            "display_name": d.get("display_name", "(unnamed)"),
            "lat": plat,
            "lon": plon,
            "address": d.get("display_name", ""),
            "tags": {},
            "latlon": (plat, plon),
        })
    cache_set_json(sig, out)
    return out

if __name__ == "__main__":
    try:
        PlacesTUI().run()
    except KeyboardInterrupt:
        sys.exit(0)

