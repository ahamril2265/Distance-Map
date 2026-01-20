import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import folium

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
COLLEGE_NAME = "Kumaraguru College of Technology, Coimbatore, India"
DEM_FILE = "data/srtm.tif"
OUTPUT_DIR = "output"

GRAPH_FILE = f"{OUTPUT_DIR}/campus_graph.graphml"
EDGE_FILE = f"{OUTPUT_DIR}/edges.csv"
MAP_FILE = f"{OUTPUT_DIR}/campus_map.html"

NETWORK_TYPE = "walk"

# ---------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.log_console = True

    print("📍 Extracting campus road network...")
    G = ox.graph_from_place(
        COLLEGE_NAME,
        network_type=NETWORK_TYPE,
        simplify=True
    )

    print(f"✅ Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    # ---------------------------------------------------
    # IMPORTANT FIX: disable multiprocessing
    # ---------------------------------------------------
    print("⛰️ Adding elevation (single-process, Windows-safe)...")
    G = ox.elevation.add_node_elevations_raster(
        G,
        filepath=DEM_FILE,
        band=1,
        cpus=1   # 🔴 THIS FIXES YOUR CRASH
    )

    print("📐 Computing slopes...")
    G = ox.elevation.add_edge_grades(G, add_absolute=True)

    print("💾 Saving graph...")
    ox.save_graphml(G, GRAPH_FILE)

    print("📊 Creating edge dataset...")
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges = edges.reset_index()

    # elevation change = slope * length
    edges["elevation_change"] = edges["grade"] * edges["length"]

    edges["elevation_gain"] = edges["elevation_change"].clip(lower=0)
    edges["elevation_loss"] = (-edges["elevation_change"]).clip(lower=0)
    edges["slope_percent"] = edges["grade"] * 100


    edge_features = edges[[
        "u", "v",
        "length",
        "elevation_gain",
        "elevation_loss",
        "slope_percent",
        "grade_abs",
        "highway",
        "geometry"
    ]]

    edge_features.to_csv(EDGE_FILE, index=False)
    print(f"✅ Saved edge features → {EDGE_FILE}")

    # ---------------------------------------------------
    # MAP
    # ---------------------------------------------------
    print("🗺️ Generating map...")
    center = edges.geometry.unary_union.centroid

    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=17,
        tiles="cartodbpositron"
    )

    def color_edge(slope):
        if slope < 2:
            return "green"
        elif slope < 5:
            return "orange"
        else:
            return "red"

    for _, row in edge_features.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color=color_edge(abs(row.slope_percent)),
            weight=4,
            opacity=0.85,
            tooltip=(
                f"Distance: {row.length:.1f} m<br>"
                f"Elevation Gain: {row.elevation_gain:.2f} m<br>"
                f"Slope: {row.slope_percent:.2f}%"
            )
        ).add_to(m)

    m.save(MAP_FILE)
    print(f"✅ Map saved → {MAP_FILE}")
    print("🎯 DONE")

# ---------------------------------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
