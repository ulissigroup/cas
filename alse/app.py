import base64
import datetime
import io
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL

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
        html.P(),
        html.Div(
            id="xrange",
            # style={'height': '280px', 'overflowX': 'hidden', 'overflowY': 'auto'},
        ),
        dcc.Store(id="xrange-store", storage_type="memory"),
        html.P(),
        html.Div(
            id="yconstraint",
            # style={'height': '280px', 'overflowX': 'hidden', 'overflowY': 'auto'},
        ),
        dcc.Store(id="yconstraint-store", storage_type="memory"),
        html.Hr(),
        html.Center(
            [
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
            ),
            dcc.Store(id="stored-data", data=df.to_dict("records")),
            html.Hr(),  # horizontal line
            html.P("Insert X axis data"),
            dcc.Dropdown(
                id="xaxis-name",
                options=[{"label": x, "value": x} for x in df.columns],
                multi=True,
            ),
            html.P("Inset Y axis data"),
            dcc.Dropdown(
                id="yaxis-name",
                options=[{"label": x, "value": x} for x in df.columns],
                multi=True,
            ),
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


@app.callback(Output("xrange", "children"), Input("xaxis-name", "value"))
def set_range(x_data):
    if x_data is not None:
        children = [html.P("Input Range")]
        for i in range(len(x_data)):
            children.append(
                html.Div(
                    x_data[i],
                    id={"type": "x_name_", "index": str(i)},
                )
            )
            children.append(
                dcc.Input(
                    id={"type": "x_min_", "index": str(i)},
                    type="number",
                    placeholder="Min",
                )
            )
            children.append(
                dcc.Input(
                    id={"type": "x_max_", "index": str(i)},
                    type="number",
                    placeholder="Max",
                )
            )
        return children
    else:
        return dash.no_update


@app.callback(Output("yconstraint", "children"), Input("yaxis-name", "value"))
def set_constraint(y_data):
    if y_data is not None:
        children = [html.P("Output Constraints")]
        for i in range(len(y_data)):
            children.append(
                html.Div(
                    y_data[i],
                    id={"type": "y_name_", "index": str(i)},
                )
            )
            children.append(
                html.Div(
                    [
                        dcc.Dropdown(
                            id={"type": "y_cons_str_", "index": str(i)},
                            options=[
                                {"label": "Greater than", "value": "gt"},
                                {"label": "Less than", "value": "lt"},
                            ],
                        )
                    ],
                    style={"width": "20%"},
                )
            )
            children.append(
                dcc.Input(
                    id={"type": "y_cons_int_", "index": str(i)},
                    type="number",
                    placeholder="constraint",
                )
            )
        return children
    else:
        return dash.no_update


@app.callback(
    Output("suggest_data", "children"),
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
    prevent_initial_call=True,
)
def initialize_alse(
    nbutton, data, x_min, x_max, y_cons_str, y_cons_int, x_names, y_names
):
    if nbutton > 0:
        dff = pd.DataFrame(data)
        input_param = []
        for xname in x_names:
            input_param.append(torch.tensor(dff[xname]))
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
        new_pts = algo.next_test_points(5)
        new_pts_df = pd.DataFrame(new_pts.numpy())
        new_pts_df.columns = [i for i in x_names]
        for i in y_names:
            new_pts_df[i] = "tbd"
        new_pts_df["id"] = f"batch {nbutton}"

        return [
            dash_table.DataTable(
                id="suggest_table",
                data=new_pts_df.to_dict("records"),
                columns=[{"name": i, "id": i, "editable": False} for i in x_names]
                + [{"name": i, "id": i, "editable": True} for i in y_names],
                style_data={
                    "whiteSpace": "normal",
                    "height": "auto",
                    "width": "auto",
                },
                editable=True,
            )
        ]


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


# TODO: add punchout_radius
# TODO: generate suggested test points in a table
# TODO: allow editing of the table (adding outputs for the test points)
# TODO: add a button to update the initial data table
# TODO: add visualizations

if __name__ == "__main__":
    app.run_server(debug=True)
