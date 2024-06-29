import time
from dash import Dash, html,  dcc, Input, Output, State, Patch, no_update, set_props
from plotly import graph_objects as go
import numpy as np
import cv2
from PIL import Image
from scipy.signal import convolve
from thinning_masks import masks
#Automatically chooses direct (sum) or Fourier method based on an estimate of which is faster (default). See Notes for more detail.

#PLEASE RUN IN THE CORRESPONDING DIRECTORY

image = np.array(Image.open('assets/image_big.jpeg'))
#image = np.array(Image.open('assets/image.jpeg'))
#image = np.array(Image.open('assets/image_small.jpeg'))

#Aufgabe 1

sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

optimized_sobel_x = np.array([[3, 0, -3],
                                [10, 0, -10],
                                [3, 0, -3]])/32

optimized_sobel_y = np.array([[3, 10, 3],
                            [0, 0, 0],
                            [-3, -10, -3]])/32

laplace_kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

laplace_kernel2 = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])


binomial_mask = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])/16

log_kernel = convolve(laplace_kernel, binomial_mask)


i_matrix = np.zeros((3,3))
i_matrix[1][1] = 1

dog_kernel = convolve(4*(binomial_mask - i_matrix), binomial_mask)

##########################################
##########################################
#Aufgabe 2
def rgb2bin(rgb):
    threshold = 20
    grayscale_image = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    binary_image = np.where(grayscale_image > threshold, 1, 0)
    return np.array(binary_image, np.uint8)

image_log = cv2.filter2D(image, -1, log_kernel)
binary_image = rgb2bin(image_log)

def to_binary_rgb(bin_image):
    rgb_binary_image = np.zeros((bin_image.shape[0], bin_image.shape[1], 3), dtype=np.uint8)
    rgb_binary_image[np.where(bin_image == 1)] = [255, 255, 255]
    rgb_binary_image[np.where(bin_image == 0)] = [0, 0, 0]
    return rgb_binary_image


kernel_erosion = np.array([[0, 1, 0],
                           [1, 1, 0],
                           [0, 0, 0]], np.uint8)

kernel_dilation = np.array([[0, 1, 0],
                            [1, 1, 0],
                            [0, 0, 0]], np.uint8)

img_erosion = cv2.erode(binary_image, kernel_erosion, iterations=1)
img_dilation = cv2.dilate(binary_image, kernel_dilation, iterations=1)

def opening(img):
    eroded = cv2.erode(img, kernel_erosion, iterations=1)
    opened = cv2.dilate(eroded, kernel_dilation, iterations=1)
    return opened

def closing(img):
    dilated = cv2.dilate(img, kernel_dilation, iterations=1)
    closed = cv2.erode(dilated, kernel_erosion, iterations=1)
    return closed

##########################################
##########################################
#Aufgabe 3

image_bin = np.array(Image.open('assets/image_bin.jpeg')) 
image_bin = rgb2bin(image_bin)


def hitmiss(img, kernel1, kernel2):
    eroded1 = cv2.erode(img, kernel1)
    eroded2 = cv2.erode(1 - img, kernel2)
    hitmiss = eroded1 & eroded2
    
    return hitmiss

kernel1 = np.array([[1, 1, 1]], np.uint8)

kernel2 = np.array([[1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1]], np.uint8)

##########################################
##########################################
#Plotting
app = Dash(__name__, title="Blatt 3", external_scripts = [{"src": "https://cdn.tailwindcss.com"}])

layout_image = go.Layout(
    hovermode='closest',
    hoverdistance=1,
    height = 170,
    width = 210,
    xaxis=dict(showgrid=False, fixedrange=True, zeroline = False, showticklabels=False),
    yaxis=dict(showgrid=False, fixedrange=True, zeroline = False, showticklabels=False),
    margin=dict(l=0,r=0,b=0,t=0,pad=0)
    )



def image_graph(z, id):
    return html.Div([
        dcc.Graph(
        figure = go.Figure(
            data = go.Image(z = z), layout=layout_image
        )
        , config={'displayModeBar': False}, id = id),
        html.Div(id, className = 'w-full text-center')
    ], className = 'flex flex-col gap-px' 
    )

app.layout = html.Div([
    html.H1("Bildverarbeitung - Blatt 3", className='font-bold text-4xl'),
    html.Div([
        html.H2("Aufgabe 1 - Kantendetektion", className='font-bold text-xl w-full justify-start'),

        html.Div([
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        options=[
                            {'label': 'Sobel', 'value': 'sobel'},
                            {'label': 'Optimized Sobel', 'value': 'osobel'},
                            {'label': 'Smoothed Sobel', 'value': 'ssobel'},
                            {'label': 'Smoothed Optimized Sobel', 'value': 'sosobel'},
                        ],
                        value='sobel',
                        id = 'switch_sobel',
                        inline = False,
                        labelStyle={'display': 'block', 'marginLeft': '2px', 'fontSize': '0.875rem', 'lineHeight':'1.25rem'}
                    )
                ], className = 'border-2 border-zinc-800 rounded-md p-4'),
                html.Div([
                    dcc.RadioItems(
                        options=[
                            {'label': 'Laplace', 'value': 'laplace'},
                            {'label': 'Laplace2', 'value': 'laplace2'},
                            {'label': 'Smoothed Laplace', 'value': 'slaplace'},
                            {'label': 'Smoothed Laplace2', 'value': 'slaplace2'},
                        ],
                        value='laplace',
                        id = 'switch_laplace',
                        inline = False,
                        labelStyle={'display': 'block', 'marginLeft': '2px', 'fontSize': '0.875rem', 'lineHeight':'1.25rem'}
                    )
                ], className = 'border-2 border-zinc-800 rounded-md p-4'),
            ],className = 'flex flex-col gap-8'),
            html.Div([
                image_graph(image, 'original'),
                image_graph(cv2.filter2D(image, -1, sobel_x), 'sobelx'),
                image_graph(cv2.filter2D(image, -1, sobel_y), 'sobely'),
                image_graph(cv2.filter2D(image, -1, laplace_kernel), 'laplace'),
                image_graph(cv2.filter2D(image, -1, log_kernel), 'log'),
                image_graph(cv2.filter2D(image, -1, dog_kernel), 'dog'),

            ], className = 'grid grid-cols-3 w-1/2 space-x-2 space-y-2'),
        ], className= 'flex flex-row gap-8')
    ], className = 'flex flex-col gap-4 justify-center w-full'),

    html.Div([
        html.H2("Aufgabe 2 - Dilation und Erosion", className='font-bold text-xl w-full justify-start'),
        html.Div([
            image_graph(to_binary_rgb(binary_image), 'binary_image'),
            image_graph(to_binary_rgb(img_erosion), 'erosion'),
            image_graph(to_binary_rgb(img_dilation), 'dilation'),
            image_graph(to_binary_rgb(opening(binary_image)), 'opening'),
            image_graph(to_binary_rgb(closing(binary_image)), 'closing'),
        ], className = 'flex flex-row gap-4')
    ], className = 'flex flex-col gap-2 justify-start w-full'),

    html.Div([
        html.H2("Aufgabe 3 - Ausdünnung", className='font-bold text-xl w-full justify-start'),
        html.Div([
            image_graph(to_binary_rgb(image_bin), 'binary'),
            image_graph(to_binary_rgb(1 - image_bin), 'inverted'),
            html.Div([
                html.Button("Start Thinning", id= 'thinning', className = 'rounded-md px-2 py-2 border-2 border-green-600 bg-green-200 text-zinc-800 font-semibold h-12'),
                dcc.Input(min = 8, value = 8, step=8, type = 'number', id = 'steps', className = 'rounded-md bg-zinc-800 w-1/3 text-white px-2 py-1')
            ], className = 'flex flex-col gap-4 items-center'),
            image_graph(to_binary_rgb(image_bin), 'thinned'),
        ], className = 'flex flex-row gap-4')
    ], className = 'flex flex-col gap-2 justify-start w-full')
        
], className='flex flex-col justify-center items-center gap-8 m-4 text-zinc-800 font-semibold')


#Callbacks
@app.callback(
    Output('sobelx', 'figure'),
    Output('sobely', 'figure'),
    Input('switch_sobel', 'value'),
    prevent_initial_call = True
)
def switch_sobel(v):   
    imagex_patch = Patch()
    imagey_patch = Patch()

    if v=='sobel':
        imagex_patch['data'][0]['z'] = cv2.filter2D(image, -1, sobel_x)
        imagey_patch['data'][0]['z'] = cv2.filter2D(image, -1, sobel_y)
    elif v=='osobel':
        imagex_patch['data'][0]['z'] = cv2.filter2D(image, -1, optimized_sobel_x)
        imagey_patch['data'][0]['z'] = cv2.filter2D(image, -1, optimized_sobel_y)
    elif v=='ssobel':
        imagex_patch['data'][0]['z'] = cv2.filter2D(cv2.filter2D(image, -1, binomial_mask), -1, sobel_x)
        imagey_patch['data'][0]['z'] = cv2.filter2D(cv2.filter2D(image, -1, binomial_mask), -1, sobel_y)
    else:
        imagex_patch['data'][0]['z'] = cv2.filter2D(cv2.filter2D(image, -1, binomial_mask), -1, optimized_sobel_x)
        imagey_patch['data'][0]['z'] = cv2.filter2D(cv2.filter2D(image, -1, binomial_mask), -1, optimized_sobel_y)
    
    return imagex_patch, imagey_patch

@app.callback(
    Output('laplace', 'figure'),
    Input('switch_laplace', 'value'),
    prevent_initial_call = True
)
def switch_laplace(v):
    image_patch = Patch()

    if v=='laplace':
        image_patch['data'][0]['z'] = cv2.filter2D(image, -1, laplace_kernel)
    elif v=='laplace2':
        image_patch['data'][0]['z'] = cv2.filter2D(image, -1, laplace_kernel2)
    elif v=='slaplace':
        image_patch['data'][0]['z'] = cv2.filter2D(cv2.filter2D(image, -1, binomial_mask), -1, laplace_kernel) 
    else:
        image_patch['data'][0]['z'] = cv2.filter2D(cv2.filter2D(image, -1, binomial_mask), -1, laplace_kernel2) 
    return image_patch

@app.callback(
    Output('thinned', 'figure'),
    Input('thinning', 'n_clicks'),
    State('steps', 'value'),
    prevent_initial_call = True
)
def thinning(n, steps):
    if not n:
        return no_update
    
    patch = Patch()
    thinned_image_bin = image_bin
    for i in range(steps):
        mask1, mask2 = masks[i%8]
        edge = hitmiss(thinned_image_bin, mask1, mask2)
        thinned_image_bin = thinned_image_bin - edge
    
    z = to_binary_rgb(thinned_image_bin)
    patch['data'][0]['z'] = z
    return patch




if __name__ == '__main__':
    app.run_server(debug=True)


