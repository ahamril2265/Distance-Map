import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import heapq
from shapely.geometry import Point, LineString
from shapely.ops import split
from shapely import wkt

# ---------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------
print("📂 Loading base road graph...")
G = nx.read_graphml("output/campus_graph_densified.graphml")

# ---------------------------------------------------
# SANITIZE NODE COORDINATES
# ---------------------------------------------------
for _, data in G.nodes(data=True):
    data["x"] = float(data["x"])
    data["y"] = float(data["y"])

# ---------------------------------------------------
# SANITIZE EDGES
# ---------------------------------------------------
for _, _, data in G.edges(data=True):
    data["length"] = float(data.get("length", 1.0))
    data["energy_base"] = float(data.get("energy_base", data["length"]))

    geom = data.get("geometry")
    if isinstance(geom, str):
        try:
            data["geometry"] = wkt.loads(geom)
        except Exception:
            data["geometry"] = None

# ---------------------------------------------------
# FORCE BIDIRECTIONAL ROADS (CAMPUS ASSUMPTION)
# ---------------------------------------------------
print("🔄 Forcing bidirectional roads...")
G = G.to_undirected(as_view=False)

# ---------------------------------------------------
# LOAD LOCATIONS
# ---------------------------------------------------
locations = pd.read_csv("output/locations.csv")

# ---------------------------------------------------
# VEHICLE CONFIG
# ---------------------------------------------------
VEHICLES = {
    "E-Scooter": {"battery": 500, "factor": 1.0},
    "E-Bike": {"battery": 750, "factor": 1.2},
    "E-Shuttle": {"battery": 1500, "factor": 2.0},
}

# ---------------------------------------------------
# EDGE-SAFE LOCATION INSERTION (MAIN-ROAD BIASED)
# ---------------------------------------------------
def insert_location_node(G, name, lat, lon, max_search_m=80):
    point = Point(lon, lat)
    best = None
    best_score = float("inf")

    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if not isinstance(geom, LineString):
            continue

        highway = str(data.get("highway", ""))
        if "service" in highway:
            continue  # deprioritize service roads

        d = geom.distance(point)
        score = d / max(geom.length, 1e-6)  # main-road bias

        if score < best_score:
            best_score = score
            best = (u, v, data, geom)

    if best and best_score < (max_search_m / 111000):
        u, v, data, geom = best
        snapped = geom.interpolate(geom.project(point))

        new_node = f"loc_{name}"
        G.add_node(new_node, x=snapped.x, y=snapped.y, location=name)

        try:
            parts = split(geom, snapped)
            if len(parts) == 2:
                g1, g2 = parts
                base_energy = data["energy_base"]

                G.remove_edge(u, v)

                G.add_edge(u, new_node,
                           length=g1.length,
                           energy_base=base_energy * (g1.length / geom.length),
                           geometry=g1)

                G.add_edge(new_node, v,
                           length=g2.length,
                           energy_base=base_energy * (g2.length / geom.length),
                           geometry=g2)

                return new_node
        except Exception:
            pass

    # fallback: nearest routable node
    candidates = [
        ((d["x"] - lon) ** 2 + (d["y"] - lat) ** 2, n)
        for n, d in G.nodes(data=True)
        if G.degree(n) > 0
    ]
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]

    return list(G.nodes)[0]

# ---------------------------------------------------
# BUILD LOCATION NODES
# ---------------------------------------------------
LOCATION_NODE = {
    row["name"]: insert_location_node(G, row["name"], row["lat"], row["lon"])
    for _, row in locations.iterrows()
}

# ---------------------------------------------------
# BATTERY-CONSTRAINED SHORTEST PATH
# ---------------------------------------------------
def battery_constrained_shortest_path(G, source, target, max_energy, factor):
    pq = [(0.0, source, [source])]
    best = {source: 0.0}

    while pq:
        energy, u, path = heapq.heappop(pq)

        if u == target:
            return path, energy

        for v in G[u]:
            data = G[u][v]

            # --- CASE 1: Multi-edge (dict of dicts) ---
            if isinstance(data, dict) and all(isinstance(val, dict) for val in data.values()):
                edge = min(
                    data.values(),
                    key=lambda d: float(d.get("energy_base", float("inf")))
                )

            # --- CASE 2: Single-edge dict ---
            elif isinstance(data, dict):
                edge = data

            else:
                continue  # safety fallback

            cost = float(edge.get("energy_base", float("inf"))) * factor
            new_energy = energy + cost

            if new_energy > max_energy:
                continue

            if v not in best or new_energy < best[v]:
                best[v] = new_energy
                heapq.heappush(pq, (new_energy, v, path + [v]))

    return None, float("inf")


# ---------------------------------------------------
# CENTER OUT THE GRAPH
# ---------------------------------------------------

def route_center(lats, lons):
    return sum(lats) / len(lats), sum(lons) / len(lons)

# ---------------------------------------------------
# ROUTE → ROAD GEOMETRY ONLY
# ---------------------------------------------------
def route_to_geometry(G, route):
    lats, lons = [], []

    for u, v in zip(route[:-1], route[1:]):
        data = G[u][v]

        # --- CASE 1: Multi-edge (dict of dicts) ---
        if isinstance(data, dict) and all(isinstance(val, dict) for val in data.values()):
            edge = min(
                data.values(),
                key=lambda d: float(d.get("length", float("inf")))
            )

        # --- CASE 2: Single-edge dict ---
        elif isinstance(data, dict):
            edge = data

        else:
            continue  # safety fallback

        geom = edge.get("geometry")

        if isinstance(geom, LineString):
            xs, ys = geom.xy
            lats.extend(ys)
            lons.extend(xs)
        else:
            # fallback: straight segment between nodes
            lats.extend([G.nodes[u]["y"], G.nodes[v]["y"]])
            lons.extend([G.nodes[u]["x"], G.nodes[v]["x"]])

    return lats, lons

# ---------------------------------------------------
# DASH APP
# ---------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Campus EV Route Optimization"),

    html.Div([
        dcc.Dropdown(list(LOCATION_NODE.keys()), id="start", placeholder="Start"),
        dcc.Dropdown(list(LOCATION_NODE.keys()), id="end", placeholder="End"),
        dcc.Dropdown(list(VEHICLES.keys()), "E-Shuttle", id="vehicle"),
        html.Button("Compute Route", id="run"),
    ], style={"width": "30%", "display": "inline-block"}),

    html.Div([
        dcc.Graph(id="map"),
        html.Div(id="metrics")
    ], style={"width": "65%", "display": "inline-block"})
])

# ---------------------------------------------------
# CALLBACK
# ---------------------------------------------------
@app.callback(
    [Output("map", "figure"), Output("metrics", "children")],
    Input("run", "n_clicks"),
    State("start", "value"),
    State("end", "value"),
    State("vehicle", "value")
)
def update(_, start_name, end_name, vehicle):
    if not start_name or not end_name:
        return go.Figure(), ""

    start = LOCATION_NODE[start_name]
    end = LOCATION_NODE[end_name]

    battery = VEHICLES[vehicle]["battery"]
    factor = VEHICLES[vehicle]["factor"]

    dist_route = nx.shortest_path(G, start, end, weight="length")
    energy_route, used = battery_constrained_shortest_path(G, start, end, battery * 0.95, factor)

    fig = go.Figure()

    for route, color, label in [
        (dist_route, "blue", "Distance-optimal"),
        (energy_route, "red", "Energy-optimal")
    ]:
        if route:
            lats, lons = route_to_geometry(G, route)
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="lines",
                line=dict(width=5, color=color),
                name=label
            ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=16,
        mapbox_center=dict(
            lat=float(G.nodes[start]["y"]),
            lon=float(G.nodes[start]["x"])
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig, html.Div([
        html.P(f"Vehicle: {vehicle}"),
        html.P(f"Battery: {battery} Wh"),
        html.P(f"Energy used: {used:.2f} Wh"),
        html.P(f"Remaining: {battery-used:.2f} Wh")
    ])

# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
