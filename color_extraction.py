from dash import Dash, html,  dcc, Input, Output, State, no_update
import numpy as np
from PIL import Image
import io
import plotly.express as px
import base64

app = Dash(__name__, title="Bildfarben", external_scripts = [{"src": "https://cdn.tailwindcss.com"}])

layout = html.Div([
    dcc.Graph(id = 'real_image'),
    dcc.Slider(id = 'threshold', value = 100, min=0, max=250, step=5 ,className = 'w-1/3'),
    html.Div([
        dcc.Graph(id = 'red_image'),
        dcc.Graph(id = 'green_image'),
        dcc.Graph(id = 'blue_image'),
    ], className = 'flex flex-row gap-0 w-full')
], className =  'flex flex-col items-center w-full')

@app.callback(
        Output('real_image', 'figure'),
        Input('real_image', 'style'),
)
def preload_image(s):
    an_image = np.array(Image.open('image.png'))
    return px.imshow(an_image)

@app.callback(
    Output('red_image', 'figure'),
    Output('green_image', 'figure'),
    Output('blue_image', 'figure'),
    Input('threshold', 'value'),
     prevent_initial_call = True
    
)
def update_graphs(v):
    threshold = v

    an_image = np.array(Image.open('image.png'))

    red_channel = an_image[:, :, 0]
    green_channel = an_image[:, :, 1]
    blue_channel = an_image[:, :, 2]
    
    red_image = np.zeros_like(an_image)
    red_image[..., 0] = np.where(red_channel > threshold, red_channel, (threshold-red_channel)*50/threshold + 205)
    red_image[..., 1] = np.where(red_channel > threshold, green_channel, (threshold-red_channel)*50/threshold + 205)
    red_image[..., 2] = np.where(red_channel > threshold, blue_channel, (threshold-red_channel)*50/threshold + 205)

    green_image = np.zeros_like(an_image)
    green_image[..., 0] = np.where(green_channel > threshold, red_channel, (threshold-green_channel)*50/threshold + 205)
    green_image[..., 1] = np.where(green_channel > threshold, green_channel, (threshold-green_channel)*50/threshold + 205)
    green_image[..., 2] = np.where(green_channel > threshold, blue_channel, (threshold-green_channel)*50/threshold + 205)

    blue_image = np.zeros_like(an_image)
    blue_image[..., 0] = np.where(blue_channel > threshold, red_channel, (threshold-blue_channel)*50/threshold + 205)
    blue_image[..., 1] = np.where(blue_channel > threshold, green_channel, (threshold-blue_channel)*50/threshold + 205)
    blue_image[..., 2] = np.where(blue_channel > threshold, blue_channel, (threshold-blue_channel)*50/threshold + 205)

    
    red_image_fig =  px.imshow(red_image)
    green_image_fig = px.imshow(green_image)
    blue_image_fig = px.imshow(blue_image)

    return red_image_fig, green_image_fig, blue_image_fig



app.layout = layout


if __name__ == '__main__':
    app.run_server(debug=True)