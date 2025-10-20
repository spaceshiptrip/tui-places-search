# Places TUI (OpenStreetMap)

A lightweight terminal UI to search for nearby places by **query** (e.g., `coffee shop`, `mexican restaurant`, `starbucks`) around a **city / state** or **US ZIP code**.  
Uses free OpenStreetMap services — **Nominatim** for geocoding and **Overpass** for places. No API keys required.

---

## ✨ Features

- Text-based UI (built with **Textual**)
- Search by **city/state** or **US ZIP code** (e.g., `91101`)
- Radius in **miles**
- Results show **name, street address, distance, lat/lon**
- Open the selected place in your browser (**o**)
- Works even when Overpass is sparse: falls back to a **bounded Nominatim** search
- **Fuzzy suggestions** (e.g., `Startbucks` → _“Did you mean **Starbucks**?”_)

---

## 🛠️ Requirements

- Python **3.10+**
- Packages: `textual` (>=0.89), `requests`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "textual>=0.89" requests
```

> If you see import errors or CSS warnings, you might be on an older/newer Textual.  
> This repo targets **Textual 0.89.x** and uses only widely available widgets.

---

## ▶️ Run

```bash
source .venv/bin/activate
python places_tui.py
```

Fill the fields and press **Enter** or click **Search** (or **Ctrl+S**).

**Examples**
- Query: `coffee shop` · Near: `Pasadena, CA` · Radius mi: `5`
- Query: `starbuks` (typo) · Near: `91101` · Radius mi: `3` → will suggest **Starbucks**

---

## 🧭 Controls

- **Enter**: search (from any field)
- **Ctrl+S**: search now
- **o**: open selected result on OpenStreetMap
- **↑/↓**: move selection
- **q**: quit

---

## ⚙️ How it works (high level)

1. **Geocode** the “Near” input via Nominatim (ZIPs auto-detected and US-biased).
2. Query **Overpass** within the radius (converted to meters):
   - Search across `name`, `brand`, `operator` (case-insensitive).
   - If nothing, try **category fallbacks**: `restaurant`, `fast_food`, `cafe`, `bar`, `pub`, `food_court` by **name**.
3. If Overpass returns nothing, run a **bounded Nominatim** search within the same area.
4. If still empty (or the query looks misspelled), compute **fuzzy suggestions** from nearby names and offer “Did you mean …”.

Distances are calculated with a great‑circle formula and shown in **miles**.

---

## ❓ Troubleshooting

- **“Search failed: 400 Bad Request”**  
  Overpass can reject some regex; the app already uses case-insensitive flags the way Overpass expects. Try simplifying the query (e.g., `starbucks` → `starb`).
- **Rate limiting / intermittent errors**  
  Overpass is a shared public service. The app rotates through multiple public endpoints, but you might still hit limits. Try again later or reduce radius.
- **No results**  
  OSM data quality varies. Try a broader query (`coffee`, `cafe`), increase radius, or search a nearby ZIP.
- **Textual errors**  
  Ensure `textual --version` shows ~0.89.x and that `from textual.widgets import ...` matches the app.

---

## 📝 License

MIT — see `LICENSE` if present. OpenStreetMap data © OpenStreetMap contributors.
