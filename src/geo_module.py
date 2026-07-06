"""
geo_module.py – Place-name extraction and offline map generation
----------------------------------------------------------------
Strategy:
  1. spaCy en_core_web_sm extracts GPE/LOC/FAC entities from English translation
  2. Match entities against:
       a. SOUTH_ASIA_GAZETTEER  – embedded ~300 militarily-relevant locations
       b. geonamescache         – 32k world cities fallback
  3. MGRS / decimal / DMS hard coordinates still parsed as before
  4. folium map with OpenStreetMap tiles (shows city/village labels like Google Maps)

No internet required at runtime. spaCy model must be downloaded once:
    python -m spacy download en_core_web_sm
"""

import re
import folium


# ── Threat colour palette ──────────────────────────────────────────────────────

THREAT_COLORS = {
    "CRITICAL": "#e60026",
    "HIGH":     "#ff6600",
    "MEDIUM":   "#ffaa00",
    "LOW":      "#00aaff",
    "CLEAR":    "#00cc66",
}


# ── Regional gazetteer: South & Central Asia ───────────────────────────────────
# Format: "Place Name": (lat, lon, "Country/Region")
# Covers Afghanistan, Pakistan, India (J&K/NE), Myanmar, Central Asia, China (Tibet/Xinjiang)

SOUTH_ASIA_GAZETTEER: dict = {
    # ── Common ASR/NLLB transliteration variants (alias → canonical coords) ────
    "shinagar":         (34.0857, 74.8055, "India J&K"),
    "shi nagar":        (34.0857, 74.8055, "India J&K"),
    "sri nagar":        (34.0857, 74.8055, "India J&K"),
    "anantnak":         (33.7311, 75.1487, "India J&K"),
    "anantnad":         (33.7311, 75.1487, "India J&K"),
    "anant nag":        (33.7311, 75.1487, "India J&K"),
    "baramula":         (34.1979, 74.3629, "India J&K"),
    "kupwada":          (34.5260, 74.2546, "India J&K"),
    "muzafarabad":      (34.3700, 73.4711, "Pakistan AJK"),

    # ── Afghanistan ──────────────────────────────────────────────────────────
    "kabul":            (34.5281, 69.1723, "Afghanistan"),
    "kandahar":         (31.6258, 65.7231, "Afghanistan"),
    "herat":            (34.3482, 62.2040, "Afghanistan"),
    "mazar-i-sharif":   (36.7069, 67.1100, "Afghanistan"),
    "mazar i sharif":   (36.7069, 67.1100, "Afghanistan"),
    "jalalabad":        (34.4301, 70.4483, "Afghanistan"),
    "kunduz":           (36.7270, 68.8586, "Afghanistan"),
    "lashkar gah":      (31.5933, 64.3700, "Afghanistan"),
    "lashkargah":       (31.5933, 64.3700, "Afghanistan"),
    "ghazni":           (33.5535, 68.4201, "Afghanistan"),
    "khost":            (33.3338, 69.9200, "Afghanistan"),
    "paktia":           (33.7000, 69.4667, "Afghanistan"),
    "kunar":            (34.8467, 71.0969, "Afghanistan"),
    "asadabad":         (34.8731, 71.1508, "Afghanistan"),
    "bamyan":           (34.8221, 67.8270, "Afghanistan"),
    "helmand":          (31.5000, 64.0000, "Afghanistan"),
    "farah":            (32.3753, 62.1167, "Afghanistan"),
    "panjshir":         (35.1224, 69.5160, "Afghanistan"),
    "baghlan":          (36.1302, 68.7090, "Afghanistan"),
    "logar":            (33.9800, 69.1900, "Afghanistan"),
    "wardak":           (34.2000, 68.5167, "Afghanistan"),
    "nangarhar":        (34.2890, 70.4370, "Afghanistan"),
    "nuristan":         (35.3000, 70.8333, "Afghanistan"),
    "zabul":            (32.0000, 67.3333, "Afghanistan"),
    "uruzgan":          (32.9316, 65.8530, "Afghanistan"),
    "nimroz":           (31.0000, 62.3333, "Afghanistan"),
    "ghor":             (34.0833, 64.9833, "Afghanistan"),
    "samangan":         (36.1544, 67.9561, "Afghanistan"),
    "faryab":           (35.9167, 64.5167, "Afghanistan"),
    "badakhshan":       (36.7392, 70.8117, "Afghanistan"),
    "takhar":           (36.7348, 69.5342, "Afghanistan"),
    "tora bora":        (34.0000, 70.3333, "Afghanistan"),
    "spin boldak":      (30.9757, 66.8406, "Afghanistan"),
    "chaman":           (30.9213, 66.4503, "Afghanistan / Pakistan"),
    "torkham":          (34.0994, 71.1116, "Afghanistan / Pakistan"),
    # ── Pakistan ─────────────────────────────────────────────────────────────
    "islamabad":        (33.6844, 73.0479, "Pakistan"),
    "rawalpindi":       (33.5651, 73.0169, "Pakistan"),
    "peshawar":         (34.0080, 71.5785, "Pakistan"),
    "quetta":           (30.1841, 67.0014, "Pakistan"),
    "lahore":           (31.5580, 74.3507, "Pakistan"),
    "karachi":          (24.8608, 67.0104, "Pakistan"),
    "multan":           (30.1978, 71.4711, "Pakistan"),
    "faisalabad":       (31.4180, 73.0790, "Pakistan"),
    "gwadar":           (25.1216, 62.3254, "Pakistan"),
    "sialkot":          (32.4945, 74.5229, "Pakistan"),
    "swat":             (35.2227, 72.4258, "Pakistan"),
    "mingora":          (34.7744, 72.3601, "Pakistan"),
    "waziristan":       (32.3000, 69.8000, "Pakistan"),
    "north waziristan": (33.0000, 70.0667, "Pakistan"),
    "south waziristan": (32.3000, 69.8000, "Pakistan"),
    "khyber":           (34.0667, 71.2000, "Pakistan"),
    "bajaur":           (34.8167, 71.5167, "Pakistan"),
    "mohmand":          (34.5000, 71.2667, "Pakistan"),
    "kurram":           (33.5833, 70.0167, "Pakistan"),
    "chitral":          (35.8511, 71.7889, "Pakistan"),
    "dir":              (35.2000, 71.9000, "Pakistan"),
    "abbottabad":       (34.1558, 73.2194, "Pakistan"),
    "mansehra":         (34.3335, 73.1999, "Pakistan"),
    "gilgit":           (35.9220, 74.3089, "Pakistan"),
    "skardu":           (35.2971, 75.6333, "Pakistan"),
    "muzaffarabad":     (34.3700, 73.4700, "Pakistan AJK"),
    "mirpur":           (33.1450, 73.7513, "Pakistan AJK"),
    "khuzdar":          (27.8000, 66.6167, "Pakistan"),
    "turbat":           (26.0028, 63.0417, "Pakistan"),
    "dera ismail khan": (31.8309, 70.9012, "Pakistan"),
    "tank":             (32.2248, 70.3785, "Pakistan"),
    "bannu":            (32.9862, 70.6033, "Pakistan"),
    "kohat":            (33.5855, 71.4420, "Pakistan"),
    "nowshera":         (34.0153, 71.9748, "Pakistan"),
    "mardan":           (34.1981, 72.0445, "Pakistan"),
    "hyderabad":        (25.3792, 68.3683, "Pakistan"),
    "sukkur":           (27.6995, 68.8673, "Pakistan"),
    "larkana":          (27.5570, 68.2186, "Pakistan"),
    # ── India – Kashmir & Northeast ───────────────────────────────────────────
    "srinagar":         (34.0857, 74.8055, "India J&K"),
    "jammu":            (32.7266, 74.8570, "India J&K"),
    "baramulla":        (34.1979, 74.3629, "India J&K"),
    "sopore":           (34.3000, 74.4667, "India J&K"),
    "kupwara":          (34.5260, 74.2649, "India J&K"),
    "pulwama":          (33.8693, 74.8987, "India J&K"),
    "shopian":          (33.7163, 74.8389, "India J&K"),
    "anantnag":         (33.7311, 75.1487, "India J&K"),
    "kargil":           (34.5539, 76.1348, "India J&K"),
    "leh":              (34.1526, 77.5771, "India Ladakh"),
    "siachen":          (35.4000, 77.0000, "India Ladakh"),
    "imphal":           (24.8170, 93.9368, "India Manipur"),
    "kohima":           (25.6701, 94.1077, "India Nagaland"),
    "dimapur":          (25.9071, 93.7276, "India Nagaland"),
    "aizawl":           (23.7271, 92.7176, "India Mizoram"),
    "agartala":         (23.8315, 91.2868, "India Tripura"),
    "shillong":         (25.5788, 91.8933, "India Meghalaya"),
    "guwahati":         (26.1445, 91.7362, "India Assam"),
    "tawang":           (27.5844, 91.8597, "India Arunachal"),
    "chandigarh":       (30.7333, 76.7794, "India Punjab"),
    "amritsar":         (31.6340, 74.8723, "India Punjab"),
    "pathankot":        (32.2743, 75.6508, "India Punjab"),
    # ── Nepal ─────────────────────────────────────────────────────────────────
    "kathmandu":        (27.7172, 85.3240, "Nepal"),
    "pokhara":          (28.2096, 83.9856, "Nepal"),
    "biratnagar":       (26.4541, 87.2718, "Nepal"),
    # ── Myanmar ───────────────────────────────────────────────────────────────
    "yangon":           (16.8661, 96.1951, "Myanmar"),
    "mandalay":         (21.9588, 96.0891, "Myanmar"),
    "naypyidaw":        (19.7633, 96.0785, "Myanmar"),
    "myitkyina":        (25.3814, 97.3940, "Myanmar"),
    "lashio":           (22.9333, 97.7500, "Myanmar"),
    "bago":             (17.3400, 96.4800, "Myanmar"),
    "taunggyi":         (20.7878, 97.0369, "Myanmar"),
    "mawlamyine":       (16.4833, 97.6167, "Myanmar"),
    "rakhine":          (20.0833, 93.0000, "Myanmar"),
    "chin state":       (22.0000, 93.5000, "Myanmar"),
    "kachin":           (25.5000, 97.5000, "Myanmar"),
    "shan":             (21.0000, 98.0000, "Myanmar"),
    # ── China border regions ──────────────────────────────────────────────────
    "kashgar":          (39.4704, 75.9895, "China Xinjiang"),
    "hotan":            (37.1148, 79.9299, "China Xinjiang"),
    "urumqi":           (43.8256, 87.6168, "China Xinjiang"),
    "lhasa":            (29.6500, 91.1000, "China Tibet"),
    "shigatse":         (29.2678, 88.8803, "China Tibet"),
    "nyingchi":         (29.6504, 94.3622, "China Tibet"),
    # ── Central Asia ─────────────────────────────────────────────────────────
    "dushanbe":         (38.5598, 68.7870, "Tajikistan"),
    "khorog":           (37.4900, 71.5600, "Tajikistan"),
    "tashkent":         (41.2995, 69.2401, "Uzbekistan"),
    "samarkand":        (39.6270, 66.9750, "Uzbekistan"),
    "andijan":          (40.7821, 72.3442, "Uzbekistan"),
    "fergana":          (40.3864, 71.7864, "Uzbekistan"),
    "termez":           (37.2244, 67.2783, "Uzbekistan"),
    "almaty":           (43.2220, 76.8512, "Kazakhstan"),
    "bishkek":          (42.8746, 74.5698, "Kyrgyzstan"),
    "osh":              (40.5283, 72.7985, "Kyrgyzstan"),
    "ashgabat":         (37.9601, 58.3261, "Turkmenistan"),
    "mary":             (37.5931, 61.8300, "Turkmenistan"),
    # ── Iran / Iraq border ────────────────────────────────────────────────────
    "tehran":           (35.6892, 51.3890, "Iran"),
    "mashhad":          (36.2605, 59.6168, "Iran"),
    "zahedan":          (29.4963, 60.8629, "Iran"),
    "zaranj":           (30.9589, 61.8628, "Afghanistan / Iran border"),
    "baghdad":          (33.3152, 44.3661, "Iraq"),
    "mosul":            (36.3350, 43.1189, "Iraq"),
    # ── UAE / Gulf ────────────────────────────────────────────────────────────
    "dubai":            (25.2048, 55.2708, "UAE"),
    "abu dhabi":        (24.4539, 54.3773, "UAE"),
}

# Alias map (common alternate names / transliterations)
_ALIASES: dict = {
    "mazar":        "mazar-i-sharif",
    "mazar sharif": "mazar-i-sharif",
    "k'bul":        "kabul",
    "peshwar":      "peshawar",
    "wana":         "south waziristan",
    "miranshah":    "north waziristan",
    "mirali":       "north waziristan",
    "islamabad":    "islamabad",
    "rwlpindi":     "rawalpindi",
    "pindi":        "rawalpindi",
    "khi":          "karachi",
    "lhr":          "lahore",
    "isk":          "islamabad",
    "jlb":          "jalalabad",
    "kdh":          "kandahar",
    "hkm":          "helmand",
    "helmand river": "helmand",
}


# ── Hard-coordinate regexes (unchanged from before) ───────────────────────────

_MGRS_RE = re.compile(
    r'\b(\d{1,2}[C-HJ-NP-X])\s*([A-HJ-NP-Z]{2})\s*(\d{2,5})\s*(\d{2,5})\b',
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(
    r'(-?\d{1,3}\.\d{2,6})\s*[°,\s]\s*(-?\d{1,3}\.\d{2,6})'
)
_DMS_RE = re.compile(
    r'(\d{1,3})[°d]\s*(\d{1,2})[\'m]\s*(\d{0,2}(?:\.\d+)?)[\"s]?\s*([NS])'
    r'\s+'
    r'(\d{1,3})[°d]\s*(\d{1,2})[\'m]\s*(\d{0,2}(?:\.\d+)?)[\"s]?\s*([EW])',
    re.IGNORECASE,
)


def _dms_to_decimal(deg, min_, sec, hemi):
    val = float(deg) + float(min_) / 60.0 + float(sec or 0) / 3600.0
    return round(-val if hemi.upper() in ("S", "W") else val, 6)


# ── spaCy loader (cached at module level) ──────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            _nlp = False   # mark as unavailable so we don't retry
    return _nlp if _nlp else None


# ── geonamescache lookup (built once) ─────────────────────────────────────────

_gnc_index: dict | None = None

def _get_gnc_index() -> dict:
    global _gnc_index
    if _gnc_index is None:
        try:
            import geonamescache
            gc = geonamescache.GeonamesCache()
            cities = gc.get_cities()
            idx = {}
            for c in cities.values():
                for field in ("name", "asciiname"):
                    key = (c.get(field) or "").lower().strip()
                    if len(key) >= 3:
                        if key not in idx or c["population"] > idx[key]["population"]:
                            idx[key] = c
            _gnc_index = idx
        except Exception:
            _gnc_index = {}
    return _gnc_index


# ── Geocoding ─────────────────────────────────────────────────────────────────

def _geocode_name(name: str) -> dict | None:
    """
    Resolve a place name to {label, lat, lon, source, region}.
    Checks: alias map → gazetteer → geonamescache.
    """
    key = name.lower().strip()

    # Alias resolution
    key = _ALIASES.get(key, key)

    # Custom gazetteer (highest priority for our region)
    if key in SOUTH_ASIA_GAZETTEER:
        lat, lon, region = SOUTH_ASIA_GAZETTEER[key]
        return {"label": name.title(), "lat": lat, "lon": lon,
                "source": "place", "region": region}

    # geonamescache fallback
    gnc = _get_gnc_index()
    c = gnc.get(key)
    if c:
        cc = c.get("countrycode", "")
        return {
            "label":  name.title(),
            "lat":    float(c["latitude"]),
            "lon":    float(c["longitude"]),
            "source": "place",
            "region": cc,
        }

    return None


# ── Main extraction ────────────────────────────────────────────────────────────

def extract_locations(text: str) -> list:
    """
    Extract all geographic references from text.
    Returns list of dicts: {label, lat, lon, source, region(optional)}
    source: "mgrs" | "decimal" | "dms" | "place"
    """
    if not text:
        return []

    locations: list = []
    seen: set = set()

    def _add(loc: dict):
        key = f"{loc['lat']:.3f},{loc['lon']:.3f}"
        if key not in seen:
            seen.add(key)
            locations.append(loc)

    # ── MGRS ──────────────────────────────────────────────────────────────────
    try:
        import mgrs as _mgrs_lib
        _m = _mgrs_lib.MGRS()
        for match in _MGRS_RE.finditer(text):
            raw = match.group(0).replace(" ", "").upper()
            try:
                lat, lon = _m.toLatLon(raw)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    _add({"label": raw, "lat": round(lat, 6),
                          "lon": round(lon, 6), "source": "mgrs"})
            except Exception:
                pass
    except ImportError:
        pass

    # ── Decimal degrees ────────────────────────────────────────────────────────
    for match in _DECIMAL_RE.finditer(text):
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            _add({"label": f"{lat:.4f}, {lon:.4f}",
                  "lat": lat, "lon": lon, "source": "decimal"})

    # ── DMS ────────────────────────────────────────────────────────────────────
    for match in _DMS_RE.finditer(text):
        d1, m1, s1, h1, d2, m2, s2, h2 = match.groups()
        lat = _dms_to_decimal(d1, m1, s1, h1)
        lon = _dms_to_decimal(d2, m2, s2, h2)
        _add({"label": match.group(0).strip(),
              "lat": lat, "lon": lon, "source": "dms"})

    # ── Direct gazetteer scan (most reliable for South Asian names) ───────────
    # Scan the full text for every entry in the regional gazetteer.
    # This catches names spaCy doesn't know (Lashkar Gah, Swat, Bajaur…).
    text_lower = text.lower()
    for gaz_key, (lat, lon, region) in SOUTH_ASIA_GAZETTEER.items():
        # Word-boundary match to avoid "iran" in "terrain" etc.
        pattern = r'(?<![a-z])' + re.escape(gaz_key) + r'(?![a-z])'
        if re.search(pattern, text_lower):
            _add({"label":  gaz_key.title(),
                  "lat":    lat, "lon": lon,
                  "source": "place", "region": region})

    # ── spaCy NER for places NOT in gazetteer (world cities via geonamescache) ─
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:10000])
        candidates = [ent.text for ent in doc.ents
                      if ent.label_ in ("GPE", "LOC", "FAC")
                      and len(ent.text) >= 3]
    else:
        candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    for name in candidates:
        result = _geocode_name(name)
        if result:
            _add(result)

    return locations


# ── Map builders ───────────────────────────────────────────────────────────────

def _base_map(center_lat: float, center_lon: float,
              zoom: int = 10, height: int = 400) -> folium.Map:
    """OpenStreetMap base — shows city/village/road labels like Google Maps."""
    return folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        width="100%",
        height=height,
        prefer_canvas=True,
    )


def _marker_icon(threat_level: str, source: str) -> folium.Icon:
    colors = {
        "CRITICAL": "red",
        "HIGH":     "orange",
        "MEDIUM":   "beige",
        "LOW":      "blue",
        "CLEAR":    "green",
    }
    icons = {"mgrs": "crosshairs", "decimal": "map-marker",
             "dms": "map-marker", "place": "map-pin"}
    return folium.Icon(
        color=colors.get(threat_level.upper(), "gray"),
        icon=icons.get(source, "map-marker"),
        prefix="fa",
    )


def build_single_map(
    locations: list,
    report_id: str    = "",
    threat_level: str = "CLEAR",
    height: int       = 400,
) -> str:
    """
    Build an OSM map HTML for one intercept.
    Markers show place name / coordinate; popup shows report metadata.
    Returns empty string if no locations.
    """
    if not locations:
        return ""

    center_lat = sum(l["lat"] for l in locations) / len(locations)
    center_lon = sum(l["lon"] for l in locations) / len(locations)
    zoom       = 8 if len(locations) > 1 else 10

    fmap = _base_map(center_lat, center_lon, zoom=zoom, height=height)

    hex_col = THREAT_COLORS.get(threat_level.upper(), "#888")

    for loc in locations:
        region_line = (f'<br><span style="color:#888">{loc["region"]}</span>'
                       if loc.get("region") else "")
        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=12,
            color=hex_col,
            fill=True,
            fill_color=hex_col,
            fill_opacity=0.75,
            weight=2,
            tooltip=f'<b>{loc["label"]}</b>',
            popup=folium.Popup(
                f'<div style="font-family:sans-serif;min-width:160px">'
                f'<b>{loc["label"]}</b>{region_line}<br>'
                f'<span style="font-size:0.8em;color:#555">'
                f'{loc["lat"]:.4f}, {loc["lon"]:.4f}</span><br>'
                f'<span style="font-size:0.75em;color:#777">'
                f'Report: {report_id} &nbsp;|&nbsp; {threat_level}</span>'
                f'</div>',
                max_width=240,
            ),
        ).add_to(fmap)

    return fmap._repr_html_()


def build_aggregate_map(
    intercepts: list,
    height: int = 460,
) -> tuple:
    """
    Aggregate map across all intercepts.
    Each intercept dict should have: report_id, threat_level,
    transcript, translation, where_field, isum_assessment.
    Returns (html_string, total_point_count).
    """
    all_points: list = []

    for ic in intercepts:
        combined = " ".join(filter(None, [
            ic.get("where_field") or "",
            ic.get("transcript", ""),
            ic.get("translation", ""),
            ic.get("isum_assessment", ""),
        ]))
        locs = extract_locations(combined)
        if not locs:
            continue
        thr = (ic.get("threat_level") or "CLEAR").upper()
        rid = ic.get("report_id", "")
        ts  = (ic.get("timestamp_utc") or "")[:16].replace("T", " ")
        for loc in locs:
            all_points.append((loc, thr, rid, ts))

    if not all_points:
        return "", 0

    center_lat = sum(p[0]["lat"] for p in all_points) / len(all_points)
    center_lon = sum(p[0]["lon"] for p in all_points) / len(all_points)

    fmap = _base_map(center_lat, center_lon, zoom=6, height=height)

    for loc, thr, rid, ts in all_points:
        hex_col = THREAT_COLORS.get(thr, "#888")
        region_line = (f'<br><span style="color:#888">{loc["region"]}</span>'
                       if loc.get("region") else "")
        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=10,
            color=hex_col,
            fill=True,
            fill_color=hex_col,
            fill_opacity=0.70,
            weight=2,
            tooltip=f'<b>{loc["label"]}</b> [{thr}]',
            popup=folium.Popup(
                f'<div style="font-family:sans-serif;min-width:160px">'
                f'<b>{loc["label"]}</b>{region_line}<br>'
                f'<span style="font-size:0.8em;color:#555">{ts}</span><br>'
                f'<span style="font-size:0.75em;color:#777">'
                f'Report: {rid} &nbsp;|&nbsp; {thr}</span>'
                f'</div>',
                max_width=240,
            ),
        ).add_to(fmap)

    return fmap._repr_html_(), len(all_points)
