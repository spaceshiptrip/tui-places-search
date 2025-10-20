#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import sys
import textwrap
import webbrowser
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
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
USER_AGENT = "places-tui/1.4 (personal use) https://openstreetmap.org"

ZIP_RE = re.compile(r"^\s*(\d{5})(?:-\d{4})?\s*$")  # capture 5-digit ZIP

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
    return list(dict.fromkeys([q1, q2]))

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
            lat, lon, display_name = self._geocode(near)
        except Exception as e:
            self.call_from_thread(self.status.set, f"Geocoding failed: {e}")
            return

        radius_m = radius_mi * 1609.34  # miles -> meters

        results: List[Dict[str, Any]] = []
        try:
            results = self._overpass_search_all(q, lat, lon, int(radius_m))
        except Exception as e:
            self.call_from_thread(self.status.set, f"Overpass issue, trying fallback… ({e})")

        if not results:
            try:
                results = self._nominatim_bounded(q, lat, lon, radius_mi)
            except Exception as e:
                self.call_from_thread(self.status.set, f"Fallback failed: {e}")
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
            self.status.set(msg + "  Use ↑/↓, press 'o' to open in map.")
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
    def _geocode(self, near: str):
        s = near.strip()
        m = ZIP_RE.match(s)
        if m:
            zip5 = m.group(1)
            params = {"format": "json", "q": zip5, "countrycodes": "us", "limit": 1, "addressdetails": 1}
        else:
            params = {"format": "json", "q": s, "limit": 1, "addressdetails": 1}
        r = requests.get(NOMINATIM_URL, params=params,
                         headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                         timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError("No match for location.")
        d = data[0]
        return float(d["lat"]), float(d["lon"]), d.get("display_name", s)

    def _overpass_search_all(self, query, lat, lon, radius_m) -> List[Dict[str, Any]]:
        variants = simplify_variants(query)
        blocks: List[str] = []
        for qv in variants:
            q = escape_for_overpass_regex(qv)
            blocks += [
                f'node(around:{radius_m},{lat},{lon})["name"~"{q}",i];',
                f'way(around:{radius_m},{lat},{lon})["name"~"{q}",i];',
                f'relation(around:{radius_m},{lat},{lon})["name"~"{q}",i];',
                f'node(around:{radius_m},{lat},{lon})["brand"~"{q}",i];',
                f'way(around:{radius_m},{lat},{lon})["brand"~"{q}",i];',
                f'relation(around:{radius_m},{lat},{lon})["brand"~"{q}",i];',
            ]
        ql = f"[out:json][timeout:25];({''.join(blocks)});out center tags;"
        elems = self._call_overpass(ql)
        return elems or []

    def _call_overpass(self, ql: str) -> List[Dict[str, Any]]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        for url in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(url, data={"data": ql}, headers=headers, timeout=60)
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except requests.RequestException:
                continue
        return []

    def _nominatim_bounded(self, query: str, lat: float, lon: float, radius_mi: float) -> List[Dict[str, Any]]:
        left, bottom, right, top = bbox_from_center(lat, lon, radius_mi)
        params = {
            "format": "json",
            "q": query,
            "viewbox": f"{left},{top},{right},{bottom}",
            "bounded": 1,
            "limit": 50,
            "addressdetails": 1,
            "countrycodes": "us"  # gently bias when searching US ZIPs / cities
        }
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
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

if __name__ == "__main__":
    try:
        PlacesTUI().run()
    except KeyboardInterrupt:
        sys.exit(0)

