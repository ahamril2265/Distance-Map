import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import heapq

# ---------------------------------------------------
# LOAD GRAPH (WITH LOCATIONS)
# ---------------------------------------------------
print("📂 Loading graph with locations...")
G = nx.read_graphml("output/campus_graph_with_locations.graphml")

# ---------------------------------------------------
# SANITIZE EDGES
# ---------------------------------------------------
for _, _, data in G.edges(data=True):
    data["length"] = float(data.get("length", 1.0))
    data["energy_base"] = float(data.get("energy_base", data["length"]))

# ---------------------------------------------------
# LOAD LOCATIONS
# ---------------------------------------------------
locations_df = pd.read_csv("output/locations.csv")
location_names = locations_df["name"].tolist()

LOCATION_NODE = {
    row["name"]: f"loc_{row['name']}"
    for _, row in locations_df.iterrows()
}

# ---------------------------------------------------
# VEHICLES
# ---------------------------------------------------
VEHICLES = {
    "E-Scooter": {"battery": 500, "factor": 1.0},
    "E-Bike": {"battery": 750, "factor": 1.2},
    "E-Shuttle": {"battery": 1500, "factor": 2.0},
}

# ---------------------------------------------------
# BATTERY-CONSTRAINED ROUTING
# ---------------------------------------------------

def battery_constrained_shortest_path(G, source, target, max_energy, factor):
    """
    Battery-constrained shortest path.
    Works for Graph, DiGraph, and MultiDiGraph loaded from GraphML.
    """

    pq = [(0.0, source, [source])]
    best = {source: 0.0}

    while pq:
        energy, u, path = heapq.heappop(pq)

        if u == target:
            return path, energy

        for v in G[u]:

            edge_data = G[u][v]

            # ---- CASE 1: MultiDiGraph (dict of dicts) ----
            if isinstance(edge_data, dict) and all(isinstance(x, dict) for x in edge_data.values()):
                edge = min(
                    edge_data.values(),
                    key=lambda d: float(d.get("energy_base", float("inf")))
                )

            # ---- CASE 2: Graph / DiGraph (single dict) ----
            elif isinstance(edge_data, dict):
                edge = edge_data

            # ---- SAFETY FALLBACK (should never trigger) ----
            else:
                continue

            e_cost = float(edge.get("energy_base", float("inf"))) * factor
            new_energy = energy + e_cost

            if new_energy > max_energy:
                continue

            if v not in best or new_energy < best[v]:
                best[v] = new_energy
                heapq.heappush(pq, (new_energy, v, path + [v]))

    return None, float("inf")

# ---------------------------------------------------
# DASH APP
# ---------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Campus EV Route Optimization"),

    html.Div([
        html.Label("Start Location"),
        dcc.Dropdown(location_names, location_names[0], id="start"),

        html.Label("End Location"),
        dcc.Dropdown(location_names, location_names[-1], id="end"),

        html.Label("Vehicle"),
        dcc.Dropdown(list(VEHICLES.keys()), "E-Shuttle", id="vehicle"),

        html.Button("Compute Route", id="run", n_clicks=0),
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
def update(n, start_name, end_name, vehicle):
    if n == 0:
        return go.Figure(), ""

    start = LOCATION_NODE[start_name]
    end = LOCATION_NODE[end_name]

    battery = VEHICLES[vehicle]["battery"]
    factor = VEHICLES[vehicle]["factor"]

    dist_route = nx.shortest_path(G, start, end, weight="length")
    energy_route, used = battery_constrained_shortest_path(
        G, start, end, battery * 0.95, factor
    )

    fig = go.Figure()

    def draw(route, color, label):
        fig.add_trace(go.Scattermapbox(
            lat=[float(G.nodes[n]["y"]) for n in route],
            lon=[float(G.nodes[n]["x"]) for n in route],
            mode="lines",
            line=dict(width=5, color=color),
            name=label
        ))

    draw(dist_route, "blue", "Distance-optimal")
    if energy_route:
        draw(energy_route, "red", "Energy-optimal")

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=16,
        mapbox_center=dict(
            lat=float(G.nodes[start]["y"]),
            lon=float(G.nodes[start]["x"])
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    metrics = html.Div([
        html.P(f"Vehicle: {vehicle}"),
        html.P(f"Battery: {battery} Wh"),
        html.P(f"Energy used: {used:.2f} Wh"),
        html.P(f"Remaining: {battery-used:.2f} Wh")
    ])

    return fig, metrics

# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
