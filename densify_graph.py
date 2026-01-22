import networkx as nx
from shapely.geometry import LineString
from shapely import wkt
import math

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
INPUT_GRAPH = "output/campus_graph.graphml"
OUTPUT_GRAPH = "output/campus_graph_densified.graphml"

STEP_METERS = 10
METERS_PER_DEGREE = 111_000

# ---------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------
print("📂 Loading graph...")
G = nx.read_graphml(INPUT_GRAPH)

# sanitize nodes
for _, data in G.nodes(data=True):
    data["x"] = float(data["x"])
    data["y"] = float(data["y"])

# ---------------------------------------------------
# CREATE NEW GRAPH
# ---------------------------------------------------
G2 = nx.DiGraph()

# copy original nodes
for n, data in G.nodes(data=True):
    G2.add_node(n, **data)

node_id = 0

# ---------------------------------------------------
# DENSIFY EDGES
# ---------------------------------------------------
print("🔧 Densifying edges...")

for u, v, data in G.edges(data=True):

    length = float(data.get("length", 1.0))
    energy = float(data.get("energy_base", length))

    geom = data.get("geometry")

    # load geometry safely
    if isinstance(geom, str):
        try:
            geom = wkt.loads(geom)
        except Exception:
            geom = None

    if not isinstance(geom, LineString):
        geom = LineString([
            (G.nodes[u]["x"], G.nodes[u]["y"]),
            (G.nodes[v]["x"], G.nodes[v]["y"])
        ])

    step_deg = STEP_METERS / METERS_PER_DEGREE
    n_segments = max(1, int(math.ceil(geom.length / step_deg)))

    points = [
        geom.interpolate(i / n_segments, normalized=True)
        for i in range(n_segments + 1)
    ]

    prev = u

    for p in points[1:-1]:
        nid = f"d_{node_id}"
        node_id += 1

        G2.add_node(nid, x=p.x, y=p.y)

        seg = LineString([
            (G2.nodes[prev]["x"], G2.nodes[prev]["y"]),
            (p.x, p.y)
        ])

        seg_len = seg.length
        ratio = seg_len / geom.length if geom.length > 0 else 0

        G2.add_edge(
            prev,
            nid,
            length=seg_len,
            energy_base=energy * ratio,
            geometry=seg.wkt      # ✅ WKT STRING
        )

        prev = nid

    last_seg = LineString([
        (G2.nodes[prev]["x"], G2.nodes[prev]["y"]),
        (G.nodes[v]["x"], G.nodes[v]["y"])
    ])

    seg_len = last_seg.length
    ratio = seg_len / geom.length if geom.length > 0 else 0

    G2.add_edge(
        prev,
        v,
        length=seg_len,
        energy_base=energy * ratio,
        geometry=last_seg.wkt      # ✅ WKT STRING
    )

print(f"✅ Added {node_id} intermediate nodes")

# ---------------------------------------------------
# SAVE GRAPH (GraphML-safe)
# ---------------------------------------------------
print("💾 Saving densified graph...")
nx.write_graphml(G2, OUTPUT_GRAPH)

print("🎉 DONE")
print(f"➡️ Output: {OUTPUT_GRAPH}")
