import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import os

DATA_FILE = "output/locations.csv"

# ---------------------------------------------------
# LOAD / INIT DATA
# ---------------------------------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["id", "name", "lat", "lon", "description"])

    df = pd.read_csv(DATA_FILE)

    # 🔒 Ensure ID column exists
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    return df

# ---------------------------------------------------
# MAP FIGURE
# ---------------------------------------------------
def make_map(df):
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker=dict(size=12, color="red"),
            text=df["name"],
            name="Locations"
        ))
        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()
    else:
        center_lat, center_lon = 11.078349, 76.989480  # campus fallback

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=16,
        mapbox_center=dict(lat=center_lat, lon=center_lon),
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select"
    )
    return fig

# ---------------------------------------------------
# DASH APP
# ---------------------------------------------------
app = dash.Dash(__name__)

df = load_data()

app.layout = html.Div([
    html.H2("Campus Location Manager (CRUD)"),

    html.Div([
        # LEFT PANEL
        html.Div([
            dash_table.DataTable(
                id="table",
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Name", "id": "name"},
                    {"name": "Latitude", "id": "lat"},
                    {"name": "Longitude", "id": "lon"},
                    {"name": "Description", "id": "description"},
                ],
                data=df.to_dict("records"),
                row_selectable="single",
                editable=False,
                style_table={"overflowX": "auto"},
            ),

            html.Hr(),

            html.Div([
                dcc.Input(id="name", placeholder="Block name"),
                dcc.Input(id="lat", placeholder="Latitude", type="number"),
                dcc.Input(id="lon", placeholder="Longitude", type="number"),
                dcc.Input(id="desc", placeholder="Description"),
            ], style={"display": "flex", "gap": "10px"}),

            html.Br(),

            html.Button("➕ Add Location", id="add"),
            html.Button("✏️ Update Selected", id="update", style={"marginLeft": "10px"}),
            html.Button("🗑 Delete Selected", id="delete", style={"marginLeft": "10px"}),

            html.Div(id="status", style={"marginTop": "15px"})
        ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

        # RIGHT PANEL
        html.Div([
            dcc.Graph(id="map", figure=make_map(df)),
            html.Div("Click map to auto-fill latitude & longitude",
                     style={"marginTop": "10px", "color": "gray"})
        ], style={"width": "58%", "display": "inline-block", "marginLeft": "2%"})
    ])
])

# ---------------------------------------------------
# MAP CLICK → AUTO-FILL
# ---------------------------------------------------
@app.callback(
    Output("lat", "value"),
    Output("lon", "value"),
    Input("map", "clickData"),
    prevent_initial_call=True
)
def map_click(click):
    lat = click["points"][0]["lat"]
    lon = click["points"][0]["lon"]
    return round(lat, 6), round(lon, 6)

# ---------------------------------------------------
# CRUD CALLBACK
# ---------------------------------------------------
@app.callback(
    Output("table", "data"),
    Output("status", "children"),
    Output("map", "figure"),
    Input("add", "n_clicks"),
    Input("update", "n_clicks"),
    Input("delete", "n_clicks"),
    State("table", "selected_rows"),
    State("name", "value"),
    State("lat", "value"),
    State("lon", "value"),
    State("desc", "value"),
    prevent_initial_call=True
)
def crud(add_c, update_c, delete_c, selected, name, lat, lon, desc):
    df = load_data()
    action = dash.callback_context.triggered_id

    # ADD
    if action == "add":
        if not name or lat is None or lon is None:
            return df.to_dict("records"), "❌ Name & coordinates required", make_map(df)

        new_id = int(df["id"].max()) + 1 if "id" in df.columns and not df.empty else 1
        df.loc[len(df)] = [new_id, name, lat, lon, desc]
        msg = "✅ Location added"

    # UPDATE
    elif action == "update":
        if not selected:
            return df.to_dict("records"), "❌ Select a row to update", make_map(df)

        idx = selected[0]
        df.loc[idx, ["name", "lat", "lon", "description"]] = [name, lat, lon, desc]
        msg = "✏️ Location updated"

    # DELETE
    elif action == "delete":
        if not selected:
            return df.to_dict("records"), "❌ Select a row to delete", make_map(df)

        df = df.drop(index=selected[0]).reset_index(drop=True)
        msg = "🗑 Location deleted"

    df.to_csv(DATA_FILE, index=False)
    return df.to_dict("records"), msg, make_map(df)

# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
