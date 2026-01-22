import dash
from dash import dcc, html
import plotly.graph_objects as go
import networkx as nx

# ---------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------
print("📂 Loading graph...")
G = nx.read_graphml("output/campus_graph_densified.graphml")

# ---------------------------------------------------
# EXTRACT NODE COORDINATES
# ---------------------------------------------------
lats = []
lons = []
labels = []

for n, data in G.nodes(data=True):
    try:
        lat = float(data["y"])
        lon = float(data["x"])
        lats.append(lat)
        lons.append(lon)
        labels.append(str(n))
    except Exception:
        continue

print(f"✅ Loaded {len(lats)} nodes")

# ---------------------------------------------------
# DASH APP
# ---------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Campus Graph – Node Visualization"),

    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color="blue",
                        opacity=0.7
                    ),
                    text=labels,
                    hoverinfo="text"
                )
            ],
            layout=go.Layout(
                mapbox=dict(
                    style="carto-positron",
                    zoom=16,
                    center=dict(
                        lat=sum(lats) / len(lats),
                        lon=sum(lons) / len(lons)
                    )
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
        )
    )
])

# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
