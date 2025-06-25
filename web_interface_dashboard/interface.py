import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.utils import load_excel

#load the insurers names

insureres_name = load_excel("../data/matching_tabelle.xlsx")

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('My Dashboard'),
    html.Label('Select insurer(s):'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': name, 'value': name}
            for name in insureres_name['Name_fm']
        ],
        value=[insureres_name['Name_fm'].iloc[0]] if not insureres_name.empty else [],
        multi=True
    ),
    html.Div(style={'height': '30px'}),
    html.Label('Enter fee increase (%):'),
    dcc.Input(
        id='fee-increase-input',
        type='number',
        value=0,
        min=0,
        step=0.1
    ),
    html.Button('Calculate', id='calculate-btn', n_clicks=0, style={'marginLeft': '10px'}),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    [Input('calculate-btn', 'n_clicks')],
    [dash.dependencies.State('dropdown', 'value'),
     dash.dependencies.State('fee-increase-input', 'value')]
)
def on_calculate_click(n_clicks, selected_insurers, fee_increase):
    if not n_clicks or not selected_insurers:
        return ""
    #TODO calculate predicted churn Example: generate dummy churn values based on fee_increase
    predicted_churn = [min(0.05 + 0.01 * fee_increase, 1.0) for _ in selected_insurers]

    fig = px.bar(
        x=selected_insurers,
        y=predicted_churn,
        labels={'x': 'Insurers', 'y': 'Predicted Churn'},
        title='Predicted Churn by Insurer'
    )
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run(debug=True)
