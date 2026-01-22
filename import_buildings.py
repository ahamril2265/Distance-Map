import osmnx as ox
import pandas as pd

PLACE = "Kumaraguru College of Technology, Coimbatore, India"
OUT = "output/locations.csv"

gdf = ox.geometries_from_place(
    PLACE,
    tags={"building": True}
)

gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

rows = []
for i, row in gdf.iterrows():
    if "name" not in row or pd.isna(row["name"]):
        continue

    centroid = row.geometry.centroid
    rows.append({
        "id": len(rows) + 1,
        "name": row["name"],
        "lat": centroid.y,
        "lon": centroid.x,
        "description": "OSM building"
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Imported {len(df)} buildings")
