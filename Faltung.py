from dash import Dash, html,  dcc, Input, Output, State, no_update
import numpy as np
from PIL import Image
import io
import cv2
import plotly.express as px
import base64

app = Dash(__name__, title="Bildfaltung", external_scripts = [{"src": "https://cdn.tailwindcss.com"}])


def upload(id):
    upload = dcc.Upload(
        id=id,
        children=html.Div('Drag and Drop or Select Image'),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderColor': '#bada55',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    )
    return upload


app.layout = html.Div([
    html.Div("Faltung zweier Bilder", className = 'font-bold text-4xl w-full text-center'),
    html.Div([
            html.Div(upload('upload_first'), id = 'left_side', className = 'w-1/2'),
            html.Div(upload('upload_second'), id = 'right_side', className = 'w-1/2')
          ], className = 'flex flex-row gap-8 w-full justify-center'),
    html.Button("Erstelle Faltung", id= 'combine', className = 'rounded-md px-2 py-2 border-2 border-green-600 bg-green-200 text-zinc-800 font-semibold'),
    dcc.Graph(id = 'combined_images'),
    dcc.Store(id = 'right_array'),
    dcc.Store(id = 'left_array'),
], className = 'flex flex-col items-center gap-8 px-4 py-4')


def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = io.BytesIO(base64.b64decode(content_string))
    image = Image.open(decoded)
    np_array = np.array(image)
    return np_array


@app.callback(
        Output('left_side', 'children'),
        Output('left_array', 'data'),
        Input('upload_first', 'contents'),
)
def update_output(contents):
    if not contents:
        return no_update, no_update
    
    np_array = parse_contents(contents)
    
    fig = px.imshow(np_array)
    return dcc.Graph(figure = fig, style={'width': '600px', 'height': '600px'}), np_array

@app.callback(
        Output('right_side', 'children'),
        Output('right_array', 'data'),
        Input('upload_second', 'contents'),
)
def update_output(contents):
    if not contents:
        return no_update,no_update
    
    np_array = parse_contents(contents)
    
    fig = px.imshow(np_array)
    return dcc.Graph(figure = fig, style={'width': '600px', 'height': '600px'}), np_array


def discrete_convolution_2d(g, h):

    if len(g.shape) == 3:
        g = g[:, :, 0]
    if len(h.shape) == 3:
        h = h[:, :, 0]
        

    rows_g, cols_g = g.shape
    rows_h, cols_h = h.shape
    rows = rows_g + rows_h - 1
    cols = cols_g + cols_h - 1
    
    result = np.zeros((rows, cols))
    
    for x in range(rows):
        print(f"{x} row")
        for y in range(cols):
            for i in range(rows_h):
                for j in range(cols_h):
                    if x - i >= 0 and x - i < rows_g and y - j >= 0 and y - j < cols_g:
                        result[x, y] += g[x - i, y - j] * h[i, j]
    
    return result

@app.callback(
    Output('combined_images', 'figure'),
    Input('combine', 'n_clicks'),
    State('left_array', 'data'),
    State('right_array', 'data'),
)
def combine_images(n, left_array, right_array):
    if not n:
        return no_update
    left_array, right_array = np.array(left_array), np.array(right_array)

    right_array = np.fliplr(right_array)

    
    combined_image = discrete_convolution_2d(left_array, right_array)


    return px.imshow(combined_image)

if __name__ == '__main__':
    app.run_server(debug=True)
