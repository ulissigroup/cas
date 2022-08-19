import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc

import dash_html_components as html
from dash import dash_table
import plotly.express as px

import pandas as pd


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
        
        html.Div(
            id="xrange",
            # style={'height': '280px', 'overflowX': 'hidden', 'overflowY': 'auto'},
        ),
        html.Div(
            id="yconstraint",
            # style={'height': '280px', 'overflowX': 'hidden', 'overflowY': 'auto'},
        ),
        html.Hr(),
        html.Center(
            dbc.Button(
                "Run Bayesian Optimization",
                id="button_runBO",
                n_clicks=0,
                style={"width": "100%"},
                className="mt-3",
                color="primary",
                size="lg",
            ),
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
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5(
                            "Suggested data table",
                            style={"display": "inline-block"},
                        ),
                    ],
                    width="auto",
                ),
            ],
            justify="between",
        ),
        dash_table.DataTable(
            id={"type": "suggest_table", "index": 3},
            style_table={"height": "200px", "overflowX": "auto", "overflowY": "auto"},
            style_header={"fontWeight": "bold"},
            editable=True,
            css=[{"selector": "table", "rule": "table-layout: fixed"}],
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
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=15,
            ),
            dcc.Store(id="stored-data", data=df.to_dict("records")),
            html.Hr(),  # horizontal line

            html.P("Insert X axis data"),
            dcc.Dropdown(
                id="xaxis-data",
                options=[{"label": x, "value": x} for x in df.columns],
                multi=True,
            ),
            html.P("Inset Y axis data"),
            dcc.Dropdown(
                id="yaxis-data",
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


@app.callback(Output("xrange", "children"), Input("xaxis-data", "value"))
def set_range(x_data):
    if x_data is not None:
        children = [html.P("Input Range")]
        for i in range(len(x_data)):
            children.append(html.Div(
                    x_data[i],
                    id="x_data_" + str(i),
                ))
            children.append(
                dcc.Input(
                    id="x_min_" + str(i),
                    type="number",
                    placeholder="Min",
                ))
            children.append(
                dcc.Input(
                    id="x_max_" + str(i),
                    type="number",
                    placeholder="Max",
                ))
        return children
    else:
        return dash.no_update


@app.callback(Output("yconstraint", "children"), Input("yaxis-data", "value"))
def set_constraint(y_data):
    if y_data is not None:
        children = [html.P("Output Constraints")]
        for i in range(len(y_data)):
            children.append(html.Div(
                    y_data[i],
                    id="y_data_" + str(i),
                ))
            children.append(
                dcc.Input(
                    id="y_cons_str_" + str(i),
                    type="text",
                    placeholder="gt or lt",
                ))
            children.append(
                dcc.Input(
                    id="y_cons_int_" + str(i),
                    type="number",
                    placeholder="constraint",
                ))
        return children
    else:
        return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
