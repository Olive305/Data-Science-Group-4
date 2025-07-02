import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import sys
import os

from paper.test import starting_point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_extraction.utils import load_excel
from web_interface_dashboard.data_from_models import full_pred

# Load insurer names
insureres_name = starting_point()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('My Dashboard'),
    html.Label('Select insurer(s):'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': n, 'value': n} for n in insureres_name['Krankenkasse']],
        value=[insureres_name['Krankenkasse'].iloc[0]] if not insureres_name.empty else [],
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
    # Loading wrapper around the output div
    dcc.Loading(
        id='loading-output',
        type='circle',
        children=html.Div(id='output')
    )
])

@app.callback(
    Output('output', 'children'),
    Input('calculate-btn', 'n_clicks'),
    State('dropdown', 'value'),
    State('fee-increase-input', 'value')
)
def on_calculate_click(n_clicks, selected_insurers, fee_increase):
    if not n_clicks or not selected_insurers:
        # No output before first click
        return ""
    # get predictions from full_pred
    df_preds = full_pred(insurers=selected_insurers, zb_diff=fee_increase)

    # Import nn_pred here to avoid circular imports if any

    # Get nn_pred values for each insurer
    """
    nn_results = []
    for insurer in selected_insurers:
        try:
            nn_value = pred_nn(insurer, zb_diff=fee_increase)
        except Exception:
            nn_value = 0
        nn_results.append({'Insurer': insurer, 'Model': 'NN Model', 'Predicted Churn': nn_value})
    


    # Prepare full_pred results in long format
    df_long = df_preds.reset_index().melt(
        id_vars='index', value_vars=df_preds.columns,
        var_name='Model', value_name='Predicted Churn'
    ).rename(columns={'index': 'Insurer'})

    # Append nn_pred results
    df_long = pd.concat([df_long, pd.DataFrame(nn_results)], ignore_index=True)
    """

    df_long = df_preds.reset_index().melt(
        id_vars='index', value_vars=df_preds.columns,
        var_name='Model', value_name='Predicted Churn'
    ).rename(columns={'index': 'Insurer'})

    fig = px.bar(
        df_long,
        x='Insurer',
        y='Predicted Churn',
        color='Model',
        barmode='group',
        title=f'Predicted Churn for ΔFee = {fee_increase:.1f}'
    )
    # Return the Graph component
    return dcc.Graph(figure=fig)
def webapp():
    app.run(debug=False)
if __name__ == '__main__':
    app.run(debug=True)
