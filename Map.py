import os
import time
import osmnx as ox
import geopandas as gpd
import pandas as pd
import folium
from pyrosm import OSM
import networkx as nx
from tqdm import tqdm

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
COLLEGE_NAME = "Indian Institute of Technology Madras, Chennai, India"
OSM_PBF = "data/southern-zone.osm.pbf"
DEM_FILE = "data/srtm.tif"
OUTPUT_DIR = "output"

GRAPH_FILE = f"{OUTPUT_DIR}/campus_graph.graphml"
EDGE_FILE = f"{OUTPUT_DIR}/edges.csv"
MAP_FILE = f"{OUTPUT_DIR}/campus_map.html"

BUFFER_M = 1200

# ---------------------------------------------------
# UTILS
# ---------------------------------------------------
def get_bbox_from_place(place_name, buffer_m):
    gdf = ox.geocode_to_gdf(place_name).to_crs(epsg=32643)
    centroid = gdf.geometry.iloc[0].centroid
    buffered = centroid.buffer(buffer_m)
    buffered = gpd.GeoSeries([buffered], crs=32643).to_crs(epsg=4326)
    west, south, east, north = buffered.total_bounds
    return north, south, east, west

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.log_console = False

    print("📍 Computing bounding box...")
    north, south, east, west = get_bbox_from_place(COLLEGE_NAME, BUFFER_M)

    # ---------------------------------------------------
    # LOAD OSM DATA (OFFLINE, CORRECT)
    # ---------------------------------------------------
    print("📂 Loading road network from local PBF...")
    osm = OSM(OSM_PBF, bounding_box=[west, south, east, north])

    print("📥 Extracting nodes and edges...")
    nodes_gdf, edges_gdf = osm.get_network(
        network_type="driving",
        nodes=True
    )

    print("🔗 Building igraph (Pyrosm)...")
    ig = osm.to_graph(
        nodes_gdf,
        edges_gdf,
        network_type="driving",
        retain_all=False
    )

    print("🔄 Converting igraph → NetworkX...")
    G = nx.MultiDiGraph(ig.to_networkx())

    # REQUIRED for OSMnx compatibility
    G.graph["crs"] = "EPSG:4326"

    # ---------------------------------------------------
    # CORRECTLY ATTACH NODE COORDINATES
    # ---------------------------------------------------
    print("📍 Attaching node coordinates (OSM-safe)...")

    # Pyrosm stores original node id in igraph vertex attribute
    # Usually under "name"
    # ---------------------------------------------------
    # FIND OSM NODE ID ATTRIBUTE SAFELY
    # ---------------------------------------------------
    print("🔍 Detecting OSM node id attribute in igraph...")

    vertex_attrs = ig.vs.attributes()
    print("   Available vertex attributes:", vertex_attrs)

    POSSIBLE_ID_KEYS = ["id", "osmid", "osm_id", "node_id", "name"]

    osm_id_key = None
    for key in POSSIBLE_ID_KEYS:
        if key in vertex_attrs:
            osm_id_key = key
            break

    if osm_id_key is None:
        raise RuntimeError(
            f"Could not find OSM node id in igraph vertices. "
            f"Available attributes: {vertex_attrs}"
        )

    print(f"✅ Using igraph vertex attribute '{osm_id_key}' as OSM node id")

    osm_ids = ig.vs[osm_id_key]

    nodes_gdf = nodes_gdf.set_index("id")

    for nx_node, osm_id in zip(G.nodes(), osm_ids):
        if osm_id in nodes_gdf.index:
            G.nodes[nx_node]["x"] = nodes_gdf.at[osm_id, "lon"]
            G.nodes[nx_node]["y"] = nodes_gdf.at[osm_id, "lat"]


    # ---------------------------------------------------
    # ELEVATION
    # ---------------------------------------------------
    print("⛰️ Adding elevation from DEM...")
    G = ox.elevation.add_node_elevations_raster(
        G,
        filepath=DEM_FILE,
        band=1,
        cpus=1
    )

    print("📐 Computing slopes...")
    G = ox.elevation.add_edge_grades(G, add_absolute=True)

    # ---------------------------------------------------
    # SAVE GRAPH
    # ---------------------------------------------------
    print("💾 Saving graph...")
    ox.save_graphml(G, GRAPH_FILE)

    # ---------------------------------------------------
    # EDGE DATASET (ML READY)
    # ---------------------------------------------------
    print("📊 Creating ML dataset...")
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()

    edges["elevation_change"] = edges["grade"] * edges["length"]
    edges["elevation_gain"] = edges["elevation_change"].clip(lower=0)
    edges["elevation_loss"] = (-edges["elevation_change"]).clip(lower=0)
    edges["slope_percent"] = edges["grade"] * 100

    edge_features = edges[
        [
            "u", "v",
            "length",
            "elevation_gain",
            "elevation_loss",
            "slope_percent",
            "grade_abs",
            "highway",
            "geometry"
        ]
    ]

    edge_features.to_csv(EDGE_FILE, index=False)
    print(f"✅ Saved edge features → {EDGE_FILE}")

    # ---------------------------------------------------
    # MAP
    # ---------------------------------------------------
    print("🗺️ Generating map...")
    center = edges.geometry.unary_union.centroid

    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=16,
        tiles="cartodbpositron"
    )

    for _, row in tqdm(edge_features.iterrows(),
                       total=len(edge_features),
                       desc="Drawing edges"):
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color="red",
            weight=4,
            opacity=0.85
        ).add_to(m)

    m.save(MAP_FILE)

    print(f"✅ Map saved → {MAP_FILE}")
    print(f"🎯 DONE in {time.time() - start_time:.1f} seconds")

# ---------------------------------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
