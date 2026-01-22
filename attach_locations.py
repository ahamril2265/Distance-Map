import networkx as nx
import pandas as pd
import math

GRAPH_FILE = "output/campus_graph.graphml"
LOCATIONS_FILE = "output/locations.csv"
OUT_GRAPH_FILE = "output/campus_graph_with_locations.graphml"

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def euclidean_m(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000

def nearest_routable_node(G, lat, lon):
    best = None
    best_dist = float("inf")

    for n, d in G.nodes(data=True):
        try:
            if G.degree(n) == 0:
                continue
            dy = float(d["y"]) - lat
            dx = float(d["x"]) - lon
            dist = dx*dx + dy*dy
            if dist < best_dist:
                best = n
                best_dist = dist
        except:
            continue

    return best

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
print("📂 Loading graph...")
G = nx.read_graphml(GRAPH_FILE)

print("📍 Loading locations...")
locations = pd.read_csv(LOCATIONS_FILE)

# ---------------------------------------------------
# ATTACH LOCATIONS AS VIRTUAL NODES
# ---------------------------------------------------
print("🔗 Attaching location nodes...")

for _, row in locations.iterrows():
    name = row["name"]
    lat = float(row["lat"])
    lon = float(row["lon"])

    loc_node = f"loc_{name}"

    if loc_node in G:
        continue

    road_node = nearest_routable_node(G, lat, lon)
    if road_node is None:
        continue

    dist = euclidean_m(
        lat, lon,
        float(G.nodes[road_node]["y"]),
        float(G.nodes[road_node]["x"])
    )

    # --- add virtual node ---
    G.add_node(
        loc_node,
        x=lon,
        y=lat,
        node_type="location",
        label=name
    )

    # --- connector edges (bidirectional) ---
    G.add_edge(
        loc_node,
        road_node,
        length=dist,
        energy_base=dist * 0.2,
        edge_type="connector"
    )

    G.add_edge(
        road_node,
        loc_node,
        length=dist,
        energy_base=dist * 0.2,
        edge_type="connector"
    )

    print(f"   ✔ {name} → {road_node}")

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------
nx.write_graphml(G, OUT_GRAPH_FILE)
print(f"\n✅ Graph saved → {OUT_GRAPH_FILE}")
