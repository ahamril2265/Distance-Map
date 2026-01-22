import joblib
import pandas as pd
import networkx as nx
import heapq
import folium


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
GRAPH_FILE = "output/campus_graph.graphml"
MODEL_FILE = "output/energy_model.joblib"
EDGE_DATA_FILE = "output/edges_with_energy.csv"

# Battery configuration (Wh)
BATTERY_CAPACITY_WH = 1500
ENERGY_MARGIN = 0.95
MAX_ENERGY = BATTERY_CAPACITY_WH * ENERGY_MARGIN

# ---------------------------------------------------
# BATTERY-CONSTRAINED SHORTEST PATH
# ---------------------------------------------------
def battery_constrained_shortest_path(G, source, target, max_energy):
    pq = [(0.0, source, [source])]
    best_energy = {source: 0.0}

    while pq:
        energy_used, u, path = heapq.heappop(pq)

        if u == target:
            return path, energy_used

        for v, edge_data in G[u].items():

            # DiGraph vs MultiDiGraph safe handling
            if isinstance(G, nx.MultiDiGraph):
                edge = min(
                    edge_data.values(),
                    key=lambda d: float(d.get("energy_weight", float("inf")))
                )
            else:
                edge = edge_data

            e_cost = float(edge.get("energy_weight", 0.0))
            new_energy = energy_used + e_cost

            if new_energy > max_energy:
                continue

            if v not in best_energy or new_energy < best_energy[v]:
                best_energy[v] = new_energy
                heapq.heappush(pq, (new_energy, v, path + [v]))

    return None, float("inf")

# ---------------------------------------------------
# PATH COST UTILITY
# ---------------------------------------------------
def path_cost(G, path, weight):
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):

        edge_data = G.get_edge_data(u, v)

        if isinstance(G, nx.MultiDiGraph):
            edge = min(
                edge_data.values(),
                key=lambda d: float(d.get(weight, 0.0))
            )
        else:
            edge = edge_data

        cost += float(edge.get(weight, 0.0))

    return cost

# ---------------------------------------------------
# VISUALIZATION UTILITY
# ---------------------------------------------------

def visualize_routes(G, route_distance, route_energy, output_file):
    """
    Visualize distance-optimal and energy-optimal routes on a Folium map.
    """

    # Get node coordinates
    def get_coords(node):
        data = G.nodes[node]
        return (float(data["y"]), float(data["x"]))

    # Map center
    center = get_coords(route_distance[0])

    m = folium.Map(
        location=center,
        zoom_start=16,
        tiles="cartodbpositron"
    )

    # Distance-optimal route (BLUE)
    dist_coords = [get_coords(n) for n in route_distance]
    folium.PolyLine(
        dist_coords,
        color="blue",
        weight=5,
        opacity=0.8,
        tooltip="Distance-optimal route"
    ).add_to(m)

    # Energy-optimal route (RED)
    energy_coords = [get_coords(n) for n in route_energy]
    folium.PolyLine(
        energy_coords,
        color="red",
        weight=5,
        opacity=0.8,
        tooltip="Energy-optimal (battery-feasible) route"
    ).add_to(m)

    # Start & End markers
    folium.Marker(
        dist_coords[0],
        icon=folium.Icon(color="green"),
        tooltip="Start"
    ).add_to(m)

    folium.Marker(
        dist_coords[-1],
        icon=folium.Icon(color="black"),
        tooltip="Destination"
    ).add_to(m)

    m.save(output_file)
    print(f"🗺️ Route visualization saved → {output_file}")


# ---------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------
print("📂 Loading graph (NetworkX-safe)...")
G = nx.read_graphml(GRAPH_FILE)

# Ensure numeric edge attributes
for _, _, data in G.edges(data=True):
    data["length"] = float(data.get("length", 1.0))
    data["energy_weight"] = float(data.get("energy_weight", data["length"]))

# ---------------------------------------------------
# LOAD ENERGY MODEL
# ---------------------------------------------------
print("🤖 Loading trained energy model...")
model = joblib.load(MODEL_FILE)

# ---------------------------------------------------
# LOAD EDGE DATASET
# ---------------------------------------------------
edges_df = pd.read_csv(EDGE_DATA_FILE)

FEATURES = [
    "length",
    "elevation_gain",
    "elevation_loss",
    "grade_abs",
    "slope_percent",
    "highway"
]

X = edges_df[FEATURES]

print("⚡ Predicting energy for edges...")
edges_df["energy_pred_wh"] = model.predict(X)

# ---------------------------------------------------
# ATTACH ENERGY TO GRAPH EDGES (FULLY SAFE)
# ---------------------------------------------------
print("🔗 Attaching energy weights to graph...")

for _, row in edges_df.iterrows():
    u = str(row["u"])
    v = str(row["v"])

    if not G.has_edge(u, v):
        continue

    edge_data = G.get_edge_data(u, v)
    energy_val = float(row["energy_pred_wh"])

    if isinstance(G, nx.MultiDiGraph):
        for _, attr in edge_data.items():
            if isinstance(attr, dict):
                attr["energy_weight"] = energy_val
    else:
        if isinstance(edge_data, dict):
            edge_data["energy_weight"] = energy_val

# ---------------------------------------------------
# SELECT SOURCE & TARGET
# ---------------------------------------------------
nodes = list(G.nodes)

SOURCE_NODE = nodes[len(nodes) // 4]
TARGET_NODE = nodes[3 * len(nodes) // 4]

print(f"🚗 Routing from {SOURCE_NODE} → {TARGET_NODE}")

# ---------------------------------------------------
# ROUTING
# ---------------------------------------------------
print("📏 Computing shortest-distance route...")
route_distance = nx.shortest_path(
    G,
    SOURCE_NODE,
    TARGET_NODE,
    weight="length"
)

print("🔋 Computing battery-feasible energy-optimal route...")
route_energy, energy_used = battery_constrained_shortest_path(
    G,
    SOURCE_NODE,
    TARGET_NODE,
    MAX_ENERGY
)

if route_energy is None:
    print("❌ No feasible route within battery capacity")
    exit()

# ---------------------------------------------------
# COST COMPARISON
# ---------------------------------------------------
dist_cost = path_cost(G, route_distance, "length")
energy_cost = path_cost(G, route_energy, "energy_weight")

print("\n📊 ROUTE COMPARISON")
print("-----------------------------")
print(f"Distance-optimal length : {dist_cost:.2f} m")
print(f"Energy-optimal energy   : {energy_cost:.2f} Wh")
print(f"Battery capacity        : {BATTERY_CAPACITY_WH:.0f} Wh")
print(f"Energy used             : {energy_used:.2f} Wh")
print(f"Battery remaining       : {BATTERY_CAPACITY_WH - energy_used:.2f} Wh")
print(f"Nodes (distance route)  : {len(route_distance)}")
print(f"Nodes (energy route)    : {len(route_energy)}")

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------
print("🗺️ Generating route visualization...")

visualize_routes(
    G,
    route_distance,
    route_energy,
    output_file="output/route_comparison.html"
)
