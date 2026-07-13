"""
network_module.py – Multi-type relationship graph
--------------------------------------------------
Node types
  • ACTOR    — callsigns / unit designators (from who_field)
  • LOCATION — place names extracted from where_field
  • VOICE    — persistent speaker voice IDs (from speaker_voices table)

Edge types
  • co-occurrence  — two actors appeared in the same intercept
  • at-location    — actor mentioned in intercept where location appears
  • voice-link     — voice ID co-occurred with actor in same intercept
"""

import re
import sqlite3
from collections import defaultdict
from typing import List, Dict, Optional, Set, Tuple

import networkx as nx

# ── Styling ────────────────────────────────────────────────────────────────────
THREAT_COLOR = {
    "CRITICAL": "#ff3355",
    "HIGH":     "#ff8c00",
    "MEDIUM":   "#ffaa00",
    "LOW":      "#88cc00",
    "CLEAR":    "#00ff88",
}

NODE_COLOR = {
    "actor":    None,           # uses threat colour
    "location": "#e65100",      # amber
    "voice":    "#7c4dff",      # purple
    "codeword": "#00bcd4",      # cyan
}

EDGE_COLOR = {
    "co-occurrence": "#2a4060",
    "at-location":   "#5d4037",
    "voice-link":    "#4a148c",
    "uses-term":     "#006064",
}

CTYPE_SYMBOL = {
    "callsign":        "circle",
    "unit_designator": "diamond",
    "unit":            "square",
    "rank":            "triangle-up",
    "force_indicator": "hexagram",
    "unknown":         "circle",
    "location":        "star",
    "voice":           "pentagon",
    "codeword":        "cross",
}

CTYPE_LABEL = {
    "callsign":        "CALLSIGN",
    "unit_designator": "UNIT DESIGNATOR",
    "unit":            "UNIT",
    "rank":            "RANK",
    "force_indicator": "FORCE",
    "unknown":         "UNKNOWN",
    "location":        "LOCATION",
    "voice":           "VOICE ID",
    "codeword":        "CODEWORD",
}

_SKIP_TYPES = {"rank", "force_indicator"}

# Phrases that are NOT real location names
_LOCATION_NOISE = re.compile(
    r"^(no specific|not identified|grid ref|direction|location indicator|"
    r"no location|unknown|n/a|none|not mentioned)",
    re.IGNORECASE,
)


# ── Location parsing ───────────────────────────────────────────────────────────

def _parse_locations(where_field: str) -> List[str]:
    """Extract location names from a where_field string."""
    locs = []
    for part in where_field.split(";"):
        part = part.strip()
        if not part or _LOCATION_NOISE.match(part):
            continue
        # "Kandahar (origin)" → "Kandahar"
        name = re.sub(r"\s*\(.*?\)", "", part).strip().strip("-").strip()
        if name and len(name) >= 3:
            locs.append(name)
    return locs


# ── Graph building ─────────────────────────────────────────────────────────────

def build_actor_graph(
    profiles:        List[Dict],
    min_appearances: int           = 1,
    include_types:   Optional[Set] = None,
) -> nx.Graph:
    """
    Backward-compatible actor-only co-occurrence graph.
    Delegates to build_full_graph() with location/voice disabled.
    """
    return build_full_graph(
        profiles,
        min_appearances=min_appearances,
        include_types=include_types,
        db_path=None,
        include_locations=False,
        include_voices=False,
    )


def build_full_graph(
    profiles:          List[Dict],
    min_appearances:   int           = 1,
    include_types:     Optional[Set] = None,
    db_path:           Optional[str] = None,
    include_locations: bool          = True,
    include_voices:    bool          = True,
    include_codewords: bool          = True,
    aliases:           Optional[Dict] = None,
) -> nx.Graph:
    """
    Build a multi-type relationship graph.

    Parameters
    ----------
    profiles          : from db.get_actor_profiles()
    min_appearances   : minimum intercept count for actor nodes
    include_types     : actor ctype filter (None = all except _SKIP_TYPES)
    db_path           : path to transcripts.db for location/voice data
    include_locations : add location nodes and at-location edges
    include_voices    : add voice nodes and voice-link edges
    """
    if include_types is None:
        include_types = {k for k in CTYPE_SYMBOL
                         if k not in _SKIP_TYPES and k not in ("location", "voice")}

    G = nx.Graph()

    # ── 1. Actor nodes ─────────────────────────────────────────────────────────
    for p in profiles:
        if p["count"] < min_appearances:
            continue
        if p["callsign_type"] not in include_types:
            continue
        G.add_node(
            p["name"],
            node_type = "actor",
            count     = p["count"],
            threat    = p["top_threat"],
            ctype     = p["callsign_type"],
            langs     = p.get("languages", ""),
            first     = p.get("first_seen", ""),
            last      = p.get("last_seen",  ""),
        )

    if not G.nodes:
        return G

    # ── 2. Actor ↔ Actor co-occurrence edges ───────────────────────────────────
    report_actors: Dict[str, List[str]] = defaultdict(list)
    for p in profiles:
        if p["name"] not in G.nodes:
            continue
        for rid in p["report_ids"]:
            report_actors[rid].append(p["name"])

    for rid, actors in report_actors.items():
        actors = list(dict.fromkeys(actors))
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                a, b = actors[i], actors[j]
                if not (G.has_node(a) and G.has_node(b)):
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                    G[a][b]["report_ids"].append(rid)
                else:
                    G.add_edge(a, b,
                               weight=1, report_ids=[rid],
                               edge_type="co-occurrence")

    if not db_path:
        return G

    # ── 3. Location and voice nodes from database ──────────────────────────────
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row

        # Build report_id → actor names lookup for edge creation
        report_actor_set: Dict[str, Set[str]] = defaultdict(set)
        for p in profiles:
            if p["name"] in G.nodes:
                for rid in p["report_ids"]:
                    report_actor_set[rid].add(p["name"])

        if include_locations:
            rows = conn.execute(
                "SELECT i.report_id, s.where_field "
                "FROM intercepts i "
                "JOIN isums s ON s.intercept_id = i.id "
                "WHERE s.where_field IS NOT NULL AND s.where_field != ''"
            ).fetchall()

            for row in rows:
                rid  = row["report_id"]
                locs = _parse_locations(row["where_field"] or "")
                if not locs:
                    continue

                actors_in_report = report_actor_set.get(rid, set())
                if not actors_in_report and include_locations:
                    # Add location even with no actor link (if meaningful)
                    pass

                for loc in locs:
                    loc_key = f"LOC:{loc}"
                    if not G.has_node(loc_key):
                        G.add_node(loc_key,
                                   node_type = "location",
                                   label     = loc,
                                   count     = 0,
                                   ctype     = "location")
                    G.nodes[loc_key]["count"] = G.nodes[loc_key].get("count", 0) + 1

                    for actor in actors_in_report:
                        if not G.has_node(actor):
                            continue
                        edge_key = (actor, loc_key)
                        if G.has_edge(*edge_key):
                            G[actor][loc_key]["weight"] += 1
                            G[actor][loc_key]["report_ids"].append(rid)
                        else:
                            G.add_edge(actor, loc_key,
                                       weight=1, report_ids=[rid],
                                       edge_type="at-location")

        if include_voices:
            try:
                voice_rows = conn.execute(
                    "SELECT voice_id, intercept_ids, color FROM speaker_voices"
                ).fetchall()

                for vrow in voice_rows:
                    vid       = vrow["voice_id"]
                    int_ids   = __import__("json").loads(vrow["intercept_ids"] or "[]")
                    vid_color = vrow["color"] or "#7c4dff"

                    vid_count = len(int_ids)
                    if vid_count == 0:
                        continue

                    vid_key = f"VOICE:{vid}"
                    if not G.has_node(vid_key):
                        G.add_node(vid_key,
                                   node_type  = "voice",
                                   label      = vid,
                                   count      = vid_count,
                                   ctype      = "voice",
                                   node_color = vid_color)

                    for rid in int_ids:
                        for actor in report_actor_set.get(rid, set()):
                            if not G.has_node(actor):
                                continue
                            if G.has_edge(actor, vid_key):
                                G[actor][vid_key]["weight"] += 1
                            else:
                                G.add_edge(actor, vid_key,
                                           weight=1, report_ids=[rid],
                                           edge_type="voice-link")
            except Exception:
                pass   # speaker_voices table doesn't exist yet

        # ── Codeword / threat-term nodes from keyword_alerts ──────────────────
        # Coded terminology (aloo->grenades, mehmaan->infiltrators, ...) linked
        # to the actors that used them — surfaces "who is talking in code".
        if include_codewords:
            try:
                cw_rows = conn.execute(
                    "SELECT i.report_id, k.matched_word "
                    "FROM keyword_alerts k "
                    "JOIN intercepts i ON i.id = k.intercept_id "
                    "WHERE k.category = 'coded_terminology'"
                ).fetchall()
                for row in cw_rows:
                    rid  = row["report_id"]
                    word = (row["matched_word"] or "").strip()
                    if not word:
                        continue
                    cw_key = f"TERM:{word.lower()}"
                    if not G.has_node(cw_key):
                        G.add_node(cw_key,
                                   node_type = "codeword",
                                   label     = word,
                                   count     = 0,
                                   ctype     = "codeword")
                    G.nodes[cw_key]["count"] = G.nodes[cw_key].get("count", 0) + 1
                    for actor in report_actor_set.get(rid, set()):
                        if not G.has_node(actor):
                            continue
                        if G.has_edge(actor, cw_key):
                            G[actor][cw_key]["weight"] += 1
                            G[actor][cw_key]["report_ids"].append(rid)
                        else:
                            G.add_edge(actor, cw_key,
                                       weight=1, report_ids=[rid],
                                       edge_type="uses-term")
            except Exception:
                pass

        conn.close()

    except Exception:
        pass   # DB unavailable — return actor-only graph

    # ── Operator aliases: relabel any node the analyst has resolved ────────────
    if aliases:
        _al = {k.lower(): v for k, v in aliases.items() if v}
        for node, attrs in G.nodes(data=True):
            base = (attrs.get("label", node) or "").lower()
            key  = node.split(":", 1)[-1].lower()   # strip LOC:/VOICE:/TERM: prefix
            alias = _al.get(base) or _al.get(key)
            if alias:
                attrs["alias"] = alias
                attrs["label"] = f"{attrs.get('label', node)} = {alias}"

    return G


# ── Graph stats ────────────────────────────────────────────────────────────────

def graph_stats(G: nx.Graph) -> Dict:
    if not G.nodes:
        return {"nodes": 0, "edges": 0, "top_nodes": [], "components": 0, "isolated": 0}
    deg = sorted(
        [(n, d) for n, d in G.degree() if G.nodes[n].get("node_type") == "actor"],
        key=lambda x: x[1], reverse=True,
    )
    return {
        "nodes":      G.number_of_nodes(),
        "edges":      G.number_of_edges(),
        "top_nodes":  [(n, d) for n, d in deg[:5]],
        "isolated":   len(list(nx.isolates(G))),
        "components": nx.number_connected_components(G),
        "by_type":    _count_by_type(G),
    }


def _count_by_type(G: nx.Graph) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for _, attrs in G.nodes(data=True):
        counts[attrs.get("node_type", "actor")] += 1
    return dict(counts)


# ── Plotly rendering ───────────────────────────────────────────────────────────

def render_network_figure(G: nx.Graph, title: str = "") -> Optional[object]:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not G.nodes:
        return None

    n = G.number_of_nodes()
    if n <= 20:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    else:
        k = 2.5 / max(n ** 0.5, 1)
        pos = nx.spring_layout(G, k=k, iterations=120, seed=42, weight="weight")

    # ── Edge traces — one per edge type ───────────────────────────────────────
    edge_groups: Dict[str, Tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
    max_weight = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)

    for u, v, data in G.edges(data=True):
        etype  = data.get("edge_type", "co-occurrence")
        weight = data.get("weight", 1)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_groups[etype][0].extend([x0, x1, None])
        edge_groups[etype][1].extend([y0, y1, None])
        edge_groups[etype][2].append(weight)

    edge_traces = []
    for etype, (ex, ey, ews) in edge_groups.items():
        avg_w = sum(ews) / max(len(ews), 1)
        width = max(0.8, min(4.0, avg_w / max_weight * 4.0))
        edge_traces.append(go.Scatter(
            x=ex, y=ey,
            mode="lines",
            line=dict(width=width, color=EDGE_COLOR.get(etype, "#2a4060")),
            hoverinfo="none",
            showlegend=True,
            name=etype,
            legendgroup="edges",
            legendgrouptitle_text="EDGES" if etype == list(edge_groups)[0] else None,
        ))

    # ── Node traces — grouped by node_type × threat ───────────────────────────
    node_groups: Dict[Tuple, list] = defaultdict(list)
    for name, attrs in G.nodes(data=True):
        ntype  = attrs.get("node_type", "actor")
        threat = attrs.get("threat", "CLEAR") if ntype == "actor" else ntype
        node_groups[(ntype, threat)].append((name, attrs))

    node_traces = []
    threat_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAR",
                    "location", "codeword", "voice"]

    def _sort_key(k):
        ntype, threat = k
        ti = threat_order.index(threat) if threat in threat_order else 99
        return (0 if ntype == "actor" else 1 if ntype == "location" else 2, ti)

    for (ntype, threat), nodes in sorted(node_groups.items(), key=_sort_key):
        xs, ys, sizes, texts, hover, symbols, colors = [], [], [], [], [], [], []

        for name, attrs in nodes:
            x, y = pos[name]
            xs.append(x); ys.append(y)

            cnt = attrs.get("count", 1)
            if ntype == "actor":
                sizes.append(max(14, min(44, 10 + cnt * 4)))
                color = THREAT_COLOR.get(threat, "#00aaff")
            elif ntype == "location":
                sizes.append(max(10, min(28, 8 + cnt * 3)))
                color = NODE_COLOR["location"]
            elif ntype == "codeword":
                sizes.append(max(11, min(30, 9 + cnt * 3)))
                color = NODE_COLOR["codeword"]
            else:  # voice
                sizes.append(max(12, min(32, 10 + cnt * 3)))
                color = attrs.get("node_color", NODE_COLOR["voice"])

            colors.append(color)
            symbols.append(CTYPE_SYMBOL.get(attrs.get("ctype", "unknown"), "circle"))
            label = attrs.get("label", name)
            texts.append(label)

            deg = G.degree(name)
            rids = list({
                rid
                for _, _, d in G.edges(name, data=True)
                for rid in d.get("report_ids", [])
            })[:5]
            hover.append(
                f"<b>{label}</b><br>"
                f"Type: {CTYPE_LABEL.get(attrs.get('ctype', ntype), ntype.upper())}<br>"
                f"Appearances: {cnt}<br>"
                f"Connections: {deg}<br>"
                + (f"Threat: {threat}<br>" if ntype == 'actor' else "")
                + (f"Language(s): {attrs.get('langs','?')}<br>" if attrs.get('langs') else "")
                + (f"Intercepts: {', '.join(rids)}" if rids else "")
            )

        legend_name = (
            f"{threat}" if ntype == "actor"
            else f"LOCATION" if ntype == "location"
            else f"VOICE ID"
        )
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbols,
                line=dict(width=1.5, color="#0d1117"),
                opacity=0.92,
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=9, color="#8a9aaa",
                          family="Share Tech Mono, monospace"),
            hovertext=hover,
            hoverinfo="text",
            name=legend_name,
            showlegend=True,
            legendgroup=ntype,
            legendgrouptitle_text=ntype.upper() if nodes == list(node_groups.values())[0] else None,
        ))

    return go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(color="#8a9aaa", size=11, family="Share Tech Mono"),
                x=0.01, y=0.99,
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#8a9aaa", family="Share Tech Mono, monospace"),
            showlegend=True,
            legend=dict(
                bgcolor="#141c24",
                bordercolor="#2a3f55",
                borderwidth=1,
                font=dict(size=9),
                groupclick="toggleitem",
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       scaleanchor="y"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
            height=640,
        ),
    )


# ── Interactive (pyvis / vis.js) rendering ────────────────────────────────────

def render_network_html(G: nx.Graph, height_px: int = 640) -> Optional[str]:
    """Interactive vis.js network as a self-contained HTML string.

    Scroll to zoom, drag the background to pan, drag nodes to rearrange
    (physics re-settles the layout), hover for the intel tooltip, corner
    navigation buttons for zoom/fit. cdn_resources='in_line' embeds the
    vis.js bundle in the HTML, so this works fully offline.

    Returns None if pyvis is unavailable (caller falls back to plotly).
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return None
    if not G.nodes:
        return None

    _shape = {"actor": "dot", "location": "triangle",
              "codeword": "diamond", "voice": "square"}

    net = Network(height=f"{height_px}px", width="100%",
                  bgcolor="#0d1117", font_color="#8a9aaa",
                  cdn_resources="in_line", notebook=False, directed=False)

    for name, attrs in G.nodes(data=True):
        ntype  = attrs.get("node_type", "actor")
        threat = attrs.get("threat", "CLEAR")
        cnt    = attrs.get("count", 1)
        if ntype == "actor":
            size  = max(14, min(44, 10 + cnt * 4))
            color = THREAT_COLOR.get(threat, "#00aaff")
        elif ntype == "location":
            size  = max(10, min(28, 8 + cnt * 3))
            color = NODE_COLOR["location"]
        elif ntype == "codeword":
            size  = max(11, min(30, 9 + cnt * 3))
            color = NODE_COLOR["codeword"]
        else:
            size  = max(12, min(32, 10 + cnt * 3))
            color = attrs.get("node_color", NODE_COLOR["voice"])

        label = attrs.get("label", name)
        deg   = G.degree(name)
        rids  = list({rid for _, _, d in G.edges(name, data=True)
                      for rid in d.get("report_ids", [])})[:5]
        tooltip = (
            f"{label}\n"
            f"Type: {CTYPE_LABEL.get(attrs.get('ctype', ntype), ntype.upper())}\n"
            f"Appearances: {cnt}   Connections: {deg}"
            + (f"\nThreat: {threat}" if ntype == "actor" else "")
            + (f"\nLanguage(s): {attrs.get('langs')}" if attrs.get("langs") else "")
            + (f"\nIntercepts: {', '.join(rids)}" if rids else "")
        )
        net.add_node(name, label=label, title=tooltip, size=size, color=color,
                     shape=_shape.get(ntype, "dot"),
                     borderWidth=1, borderWidthSelected=3)

    max_weight = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        etype  = d.get("edge_type", "co-occurrence")
        weight = d.get("weight", 1)
        net.add_edge(u, v,
                     color=EDGE_COLOR.get(etype, "#2a4060"),
                     width=max(1.0, min(6.0, weight / max_weight * 6.0)),
                     title=f"{etype}  ·  {weight}×")

    # Valid JSON only (set_options is parsed as JSON by pyvis).
    net.set_options("""{
      "interaction": {
        "hover": true, "zoomView": true, "dragView": true,
        "navigationButtons": true, "multiselect": true, "tooltipDelay": 120
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -60, "springLength": 130,
          "springConstant": 0.06, "avoidOverlap": 0.5
        },
        "stabilization": {"iterations": 200, "fit": true}
      },
      "edges": {"smooth": false},
      "nodes": {"font": {"size": 12, "face": "Share Tech Mono, monospace",
                          "color": "#c8d8e8", "strokeWidth": 0}}
    }""")

    return net.generate_html()
