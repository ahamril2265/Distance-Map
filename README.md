Campus Road Network with Distance & Elevation Analysis

This project builds a campus-level road/walkway network enriched with distance, elevation, and slope information.
The output is designed to support energy estimation and machine learning models (e.g., walking energy, EV usage, robotics path cost).

The implementation uses OpenStreetMap (OSM) data and Digital Elevation Models (DEM) to create a graph-based representation of the campus.

📍 Study Area

Campus: Kumaraguru College of Technology

Location: Coimbatore, Tamil Nadu, India

Network Type: Walkable paths and internal roads

🎯 Objectives

Extract a road/walkway graph of the campus

Compute distance for each edge

Attach elevation data to nodes

Derive slope, elevation gain, and elevation loss for edges

Export:

A graph file for network analysis

A CSV dataset for machine learning

An interactive map for visualization

🧠 Key Concepts Used

Graph-based road modeling

Digital Elevation Models (DEM)

Slope and elevation-derived features

ML-ready feature engineering

Windows-safe multiprocessing handling

🛠️ Tech Stack

Python 3.14

OSMnx – road network extraction & graph processing

GeoPandas / Pandas – spatial and tabular data handling

Rasterio – DEM-based elevation lookup

Folium – interactive map visualization

NetworkX – graph structure

📂 Project Structure
Distance-Map/
│
├── Map.py                 # Main pipeline script
├── README.md              # Project documentation
│
├── data/
│   └── srtm.tif           # DEM (SRTM / ASTER)
│
├── output/
│   ├── campus_graph.graphml
│   ├── edges.csv
│   └── campus_map.html
│
└── cache/                 # OSMnx cache (auto-generated)

⚙️ Pipeline Overview
1. Road Network Extraction

Campus boundary queried from OpenStreetMap

Walkable roads and paths extracted

Graph simplified and clipped to campus polygon

2. Elevation Attachment

Node elevations fetched from DEM raster

Windows-safe single-core execution (cpus=1)

3. Edge Feature Engineering

For each edge, the following are computed:

Feature	Description
length	Edge distance (meters)
grade	Slope (rise/run)
slope_percent	Slope × 100
elevation_gain	Positive elevation change (meters)
elevation_loss	Negative elevation change (meters)
grade_abs	Absolute slope

Elevation change is computed as:

elevation_change = grade × length

📊 Outputs
1️⃣ Graph File

output/campus_graph.graphml

Node attributes:

Latitude, longitude

Elevation (meters)

Edge attributes:

Distance

Slope

Elevation gain/loss

Usable in:

NetworkX

Gephi

QGIS

2️⃣ ML-Ready Dataset

output/edges.csv

Columns include:

u, v, length, elevation_gain, elevation_loss,
slope_percent, grade_abs, highway


This dataset can be directly used for:

Regression models

Energy estimation

Feature analysis

3️⃣ Interactive Map

output/campus_map.html

Color-coded edges:

🟢 Flat

🟠 Moderate incline

🔴 Steep incline

Hover tooltips show:

Distance

Elevation gain

Slope

🖥️ How to Run
Install dependencies
pip install osmnx geopandas rasterio folium networkx pandas

Run the pipeline
python Map.py

⚠️ Known Limitations (Current Stage)

Campus polygon is small
→ Results in 24 nodes, 66 edges

Dataset size is not sufficient for ML training yet

Intended primarily for pipeline validation

🚀 Next Planned Enhancements

Expand spatial extent using a bounding box

Increase graph size to hundreds of edges

Add baseline energy estimation model

Prepare train/test datasets

Optional: Graph Neural Network (GNN) modeling

📌 Academic & Practical Relevance

This project aligns with:

Energy-aware routing

Smart campus research

Transportation analytics

Applied machine learning on graphs

🧑‍💻 Author

ARM
College Student | Engineering & Data Systems