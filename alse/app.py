import base64
import datetime
import io
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL
import plotly.graph_objects as go

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc

import dash_html_components as html
from dash import dash_table
import plotly.express as px
import torch
import pandas as pd
from alse.alse_workflow import alse
from plotly.subplots import make_subplots

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)


app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Div(id="output-div"),
        html.Div(id="output-datatable"),
        html.Hr(),  # horizontal line
        html.Center(
            [
                dcc.Input(
                    id="batch_size",
                    type="number",
                    placeholder="Number of new points".format("number"),
                ),
                dbc.Button(
                    "Run Bayesian Optimization",
                    id="button_runAL",
                    n_clicks=0,
                    style={"width": "auto"},
                    className="mt-3",
                    color="primary",
                    size="lg",
                ),
                dbc.Button(
                    "Plot Model Predictions",
                    id="button_plot",
                    n_clicks=0,
                    style={"width": "auto"},
                    className="mt-3",
                    color="primary",
                    size="lg",
                ),
            ]
        ),
        html.Div(id="figures"),
        html.Center(
            [
                dcc.Loading(
                    id="run_loading",
                    type="default",
                    children=html.Center(
                        "",
                        id="done_run",
                        style={"width": "100%", "height": "20px", "color": "Red"},
                    ),
                ),
            ],
            className="mt-3",
        ),
        html.P(),
        html.Div(id="suggest_data"),
        dbc.Button(
            "Update Data Table",
            id="button_update_table",
            n_clicks=0,
            style={"width": "auto"},
            className="mt-3",
            color="primary",
            size="lg",
        ),
    ]
)

# TODO: fix the bug where unselected yaxis remain in the section
@app.callback(
    Output("yconstraint", "children"),
    Input("yaxis-name", "value"),
    State("yconstraint", "children"),
)
def set_constraint(y_data, current_cons):
    if y_data is not None:
        current_cons.append(
            html.Div(y_data[-1], id={"type": "y_name_", "index": str(len(y_data))},)
        )
        current_cons.append(
            html.Div(
                [
                    dcc.Dropdown(
                        id={"type": "y_cons_str_", "index": str(len(y_data))},
                        options=[
                            {"label": "Greater than", "value": "gt"},
                            {"label": "Less than", "value": "lt"},
                        ],
                    )
                ],
                style={"width": "20%"},
            )
        )
        current_cons.append(
            dcc.Input(
                id={"type": "y_cons_int_", "index": str(len(y_data))},
                type="number",
                placeholder="constraint",
            )
        )
        return current_cons
    else:
        return dash.no_update


@app.callback(
    Output("all_data", "style_data_conditional"),
    Input("xaxis-name", "value"),
    Input("yaxis-name", "value"),
    State("all_data", "style_data_conditional"),
)
def highlight_columns(x_data, y_data, curr_style):
    if x_data is not None:
        curr_style.append(
            {
                "if": {"column_id": x_data[-1]},
                "backgroundColor": "#99763d",
                "color": "white",
            }
        )
    if y_data is not None:
        curr_style.append(
            {
                "if": {"column_id": y_data[-1]},
                "backgroundColor": "#5c6e3f",
                "color": "white",
            }
        )
    return curr_style


@app.callback(
    Output("suggest_data", "children"),
    Output("model_predictions", "data"),
    Output("model_predictions_overlap", "data"),
    Input("button_runAL", "n_clicks"),
    State("stored-data", "data"),
    State(component_id={"type": "x_min_", "index": ALL}, component_property="value"),
    State(component_id={"type": "x_max_", "index": ALL}, component_property="value"),
    State(
        component_id={"type": "y_cons_str_", "index": ALL}, component_property="value"
    ),
    State(
        component_id={"type": "y_cons_int_", "index": ALL}, component_property="value"
    ),
    State("xaxis-name", "value"),
    State("yaxis-name", "value"),
    State("batch_size", "value"),
    prevent_initial_call=True,
)
def run_alse(
    nbutton, data, x_min, x_max, y_cons_str, y_cons_int, x_names, y_names, batch_size
):
    if nbutton > 0:
        dff = pd.DataFrame(data)
        input_param = []
        for xname in x_names:
            input_param.append(torch.tensor(pd.to_numeric(dff[xname])))
        output_param = []
        for yname in y_names:
            output_param.append(torch.tensor(pd.to_numeric(dff[yname])).unsqueeze(-1))
        X = torch.stack(tuple(input_param), -1)
        constraints = [
            (y_cons_str[i], float(y_cons_int[i])) for i in range(len(y_cons_str))
        ]
        bounds = torch.tensor([[float(i) for i in x_min], [float(i) for i in x_max]])
        algo = alse(X, bounds, output_param, constraints)
        # TODO: store the posterior for all 3 outputs
        algo.initialize_model(
            ["reg"] * len(y_names)
        )  # TODO: add gp type selection (class or reg)

        grid = algo.get_grid(20)  # TODO: allow customized resolution

        new_pts = algo.next_test_points(
            batch_size
        )  # TODO: allow customized number of points
        new_pts_df = pd.DataFrame(new_pts.numpy())
        new_pts_df.columns = [i for i in x_names]
        for i in y_names:
            new_pts_df[i] = "tbd"
        new_pts_df["id"] = f"batch {nbutton}"
        pos, overlap = algo.get_posterior_grid()
        return (
            [
                dash_table.DataTable(
                    id="suggest_table",
                    data=new_pts_df.to_dict("records"),
                    columns=[{"name": i, "id": i, "editable": True} for i in x_names]
                    + [{"name": i, "id": i, "editable": True} for i in y_names],
                    style_data={
                        "whiteSpace": "normal",
                        "height": "auto",
                        "width": "auto",
                    },
                    editable=True,
                )
            ],
            pos,
            overlap,
        )


@app.callback(
    Output("figures", "children"),
    Input("button_plot", "n_clicks"),
    State("stored-data", "data"),
    State(component_id={"type": "x_min_", "index": ALL}, component_property="value"),
    State(component_id={"type": "x_max_", "index": ALL}, component_property="value"),
    State("xaxis-name", "value"),
    State("yaxis-name", "value"),
    State("model_predictions", "data"),
    State("model_predictions_overlap", "data"),
    State("suggest_data", "children"),
    prevent_initial_call=True,
)
def plot_predictions_3d(
    nbutton, data, xmin, xmax, xname, yname, posterior, overlap, suggest_points
):
    # assuming 3D!
    if nbutton > 0:
        dff = pd.DataFrame(data)
        resolution = 20
        [*X] = torch.meshgrid(
            [torch.linspace(xmin[i], xmax[i], resolution) for i in range(len(xmin))],
            indexing="xy",
        )
        fig_output = []
        for i in range(len(posterior)):
            fig = go.Figure(
                data=go.Isosurface(
                    x=[*X][0].flatten(),
                    y=[*X][1].flatten(),
                    z=[*X][2].flatten(),
                    value=posterior[i],
                    surface_count=5,  # number of isosurfaces, 2 by default: only min and max
                    colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
                    caps=dict(x_show=False, y_show=False),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=pd.to_numeric(dff[xname[0]]),
                    y=pd.to_numeric(dff[xname[1]]),
                    z=pd.to_numeric(dff[xname[2]]),
                    mode="markers",
                    marker=dict(size=5),
                )
            )
            fig.update_layout(
                autosize=True,
                title={
                    "text": yname[i],
                    "y": 0.99,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                scene=dict(
                    xaxis_title=xname[0], yaxis_title=xname[1], zaxis_title=xname[2]
                ),
                margin=dict(l=0, r=0, b=0, t=0),
            )
            fig_output.append(dcc.Graph(figure=fig))
            fig_output.append(html.Hr())
        fig = go.Figure(
            go.Isosurface(
                x=[*X][0].flatten(),
                y=[*X][1].flatten(),
                z=[*X][2].flatten(),
                value=overlap,
                isomin=0,
                isomax=1,
                surface_count=4,  # number of isosurfaces, 2 by default: only min and max
                colorbar_nticks=4,  # colorbar ticks correspond to isosurface values
                caps=dict(x_show=False, y_show=False),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=pd.to_numeric(dff[xname[0]]),
                y=pd.to_numeric(dff[xname[1]]),
                z=pd.to_numeric(dff[xname[2]]),
                mode="markers",
                marker=dict(size=5),
                name="Current data",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[i[xname[0]] for i in suggest_points[0]["props"]["data"]],
                y=[i[xname[1]] for i in suggest_points[0]["props"]["data"]],
                z=[i[xname[2]] for i in suggest_points[0]["props"]["data"]],
                mode="markers",
                marker=dict(size=5),
                name="Suggested points",
            )
        )
        fig.update_layout(
            autosize=True,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            title={
                "text": "Printable Boundary",
                "y": 0.99,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            scene=dict(
                xaxis_title=xname[0], yaxis_title=xname[1], zaxis_title=xname[2]
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig_output.append(dcc.Graph(figure=fig))

        return fig_output


@app.callback(
    Output("all_data", "data"),
    Output("stored-data", "data"),
    Input("button_update_table", "n_clicks"),
    State("suggest_table", "data"),
    State("stored-data", "data"),
    prevent_initial_call=True,
)
def update_table(nbutton, new_data, old_data):
    if nbutton > 0:
        new_data = pd.DataFrame(new_data)
        old_df = pd.DataFrame(old_data)
        result = pd.concat([new_data, old_df], ignore_index=True, sort=False)
        return result.to_dict("records"), result.to_dict("records")


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            # html.Button(id="submit-button", children="Create Graph"),
            html.Hr(),
            dash_table.DataTable(
                id="all_data",
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=15,
                export_format="xlsx",
                export_headers="display",
                merge_duplicate_headers=True,
                style_data_conditional=[],
            ),
            dcc.Store(id="stored-data", data=df.to_dict("records")),
            dcc.Store(id="model_predictions"),
            dcc.Store(id="model_predictions_overlap"),
            html.Hr(),  # horizontal line
            html.P("Insert X axis data"),
            dcc.Dropdown(
                id="xaxis-name",
                options=[{"label": x, "value": x} for x in df.columns],
                multi=True,
            ),
            html.Div(id="xrange", children=[html.P("Input Range")],),
            html.P("Inset Y axis data"),
            dcc.Dropdown(
                id="yaxis-name",
                options=[{"label": x, "value": x} for x in df.columns],
                multi=True,
            ),
            html.Div(id="yconstraint", children=[html.P("Output Constraints")],),
        ]
    )


@app.callback(
    Output("output-datatable", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children


@app.callback(
    Output("xrange", "children"),
    Input("xaxis-name", "value"),
    State("xrange", "children"),
)
def set_range(x_data, current_range):
    if x_data is not None:
        current_range.append(
            html.Div(x_data[-1], id={"type": "x_name_", "index": str(len(x_data))},)
        )
        current_range.append(
            dcc.Input(
                id={"type": "x_min_", "index": str(len(x_data))},
                type="number",
                placeholder="Min",
            )
        )
        current_range.append(
            dcc.Input(
                id={"type": "x_max_", "index": str(len(x_data))},
                type="number",
                placeholder="Max",
            )
        )
        return current_range
    else:
        return dash.no_update


# TODO: add user specified punchout_radius, number of points generated in one batch
# TODO: add visualizations
# TODO: generate random initial training points
if __name__ == "__main__":
    app.run_server(debug=True)
