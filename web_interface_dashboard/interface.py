import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('My Dashboard'),
    html.Label('Select a value:'),
    dcc.Dropdown(
        options=[
            {'label': 'Option 1', 'value': '1'},
            {'label': 'Option 2', 'value': '2'},
            {'label': 'Option 3', 'value': '3'}
        ],
        value='1'
    ),
    html.Div(id='output')
])

# Sample data for the chart
df = pd.DataFrame({
    "Category": ["A", "B", "C"],
    "Value": [10, 20, 30]
})

@app.callback(
    Output('output', 'children'),
    Input('dropdown', 'value')
)
def update_output(selected_value):
    fig = px.bar(df, x="Category", y="Value", title="Sample Bar Chart")
    return dcc.Graph(figure=fig)

# Assign an id to the dropdown for callback
app.layout.children[2].id = 'dropdown'

if __name__ == '__main__':
    app.run(debug=True)
