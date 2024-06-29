import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Input(id='color1', type='text', placeholder='Enter RGB color 1 (e.g., "255,0,0")'),
        dcc.Input(id='color2', type='text', placeholder='Enter RGB color 2 (e.g., "0,0,255")')
    ]),
    html.Div(id='color-blocks', style={'display': 'flex', 'flex-wrap': 'wrap'}),
])

@app.callback(
    Output('color-blocks', 'children'),
    [Input('color1', 'value'),
     Input('color2', 'value')]
)
def update_color_blocks(color1, color2):
    if color1 and color2:
       
        rgb1 = [int(x) for x in color1.split(',')]
        rgb2 = [int(x) for x in color2.split(',')]

        # Calculate interpolated colors
        interpolated_colors = []
        for i in range(100):
            r = int(rgb1[0] + (rgb2[0] - rgb1[0]) * (i / 99))
            g = int(rgb1[1] + (rgb2[1] - rgb1[1]) * (i / 99))
            b = int(rgb1[2] + (rgb2[2] - rgb1[2]) * (i / 99))
            interpolated_colors.append(f'rgb({r}, {g}, {b})')

        color_blocks = [html.Div(style={'background-color': color, 'width': '1%', 'padding-bottom': '1%', 'aspect-ratio': '1'}) for color in interpolated_colors]

        return color_blocks
    else:
        return []

if __name__ == '__main__':
    app.run_server(debug=True)
