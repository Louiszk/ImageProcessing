import math
import cmath
from dash import Dash, html,  dcc, Input, Output, State, Patch, no_update
from plotly import graph_objects as go
import numpy as np
import cv2
from PIL import Image

#PLEASE RUN IN THE CORRESPONDING DIRECTORY
#Aufgabe 1
N = 100
abtast_range = range(N)

x_abtast = [i * 4 * math.pi / N for i in abtast_range]

y_abtast1 = [math.sin(x) for x in x_abtast]
y_abtast2 = [math.sin(x) + (3 * math.sin(2 * x + 1) - 1) for x in x_abtast]

y_fourier1 = []
y_fourier2 = []

for k in abtast_range:
        
        gk1 = 0
        gk2 = 0
        for n in abtast_range:
                gk1 += cmath.exp(-2j * cmath.pi * n * k / N) * y_abtast1[n]     #DFT
                gk2 += cmath.exp(-2j * cmath.pi * n * k / N) * y_abtast2[n]

        y_fourier1.append(gk1/N)
        y_fourier2.append(gk2/N)

frequencies = [x/(4*math.pi) for x in abtast_range[:math.ceil(N/2)]]

y_fourier1_real = [y.real for y in y_fourier1]
y_fourier2_real = [y.real for y in y_fourier2]

y_fourier1_im = [y.imag for y in y_fourier1]
y_fourier2_im = [y.imag for y in y_fourier2]



#Plotting
app = Dash(__name__, title="Blatt 2", external_scripts = [{"src": "https://cdn.tailwindcss.com"}])

layout_hm = go.Layout(
    hovermode='closest',
    hoverdistance=1,
    coloraxis_colorbar_x=-0.15,
    height = 240,
    width = 320,
    xaxis=dict(showgrid=False, fixedrange=True, zeroline = False, showticklabels=False),
    yaxis=dict(showgrid=False, fixedrange=True, zeroline = False, showticklabels=False),
    margin=dict(l=0,r=0,b=0,t=0,pad=0)
    )
layout_image = go.Layout(
    hovermode='closest',
    hoverdistance=1,
    height = 340,
    width = 420,
    xaxis=dict(showgrid=False, fixedrange=True, zeroline = False, showticklabels=False),
    yaxis=dict(showgrid=False, fixedrange=True, zeroline = False, showticklabels=False),
    margin=dict(l=0,r=0,b=0,t=0,pad=0)
    )

marker = dict(symbol='square',color = 'rgba(0,0,0,0)', size=40, line=dict(color='black', width=2))
scatter = go.Scatter(mode='markers', marker=marker, showlegend=False)

app.layout = html.Div([
    html.H1("Bildverarbeitung - Blatt 2", className='font-bold text-4xl'),
    html.Div([
        html.H2("Aufgabe 1 - DFT", className='font-bold text-xl w-full justify-start'),
        dcc.Store(id = 'image_data', data = np.array(Image.open('assets/image.jpg')).tolist()),
        html.Div([
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=x_abtast, y=y_abtast1, name = 'sin(x)'), go.Scatter(x=frequencies, y=y_fourier1_real, name = 'Fourier Re', mode= 'markers'), go.Scatter(x=frequencies, y=y_fourier1_im, name = 'Fourier Im', mode= 'markers')],
                        layout=dict(title = '(1)')
                    ),
                    style={'height': '300px'}
                ),
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=x_abtast, y=y_abtast2, name = 'sin(x) + (3sin(2x + 1) - 1)'), go.Scatter(x=frequencies, y=y_fourier2_real, name = 'Fourier Re', mode= 'markers'), go.Scatter(x=frequencies, y=y_fourier2_im, name = 'Fourier Im', mode= 'markers')],
                        layout=dict(title = '(2)')
                    ),
                    style={'height': '300px'}
                )], className = 'flex flex-row gap-2 items-center justify-center'),

        html.P("(1) Der Imaginärteil zeigt k=2 woraus die Frequenz f=1/2π folgt. (2) Es gibt einen Ausschlag bei k=0 der sich aufgrund der Konstante ergibt. Zudem einen bei k=2 und k=4, was den Frequenzen f=1/2π und f=1/π entspricht und aufgrund der sin(x) und sin(2x) entsteht.", className = 'w-7/12'),
    ], className = 'flex flex-col gap-0 justify-start w-full'),

    html.Div([
        html.H2("Aufgabe 2 & 3 - Box & Gauß Filter", className='font-bold text-xl w-full justify-start'),
        html.Div([
                html.Div([
                    dcc.Graph(figure = go.Figure(data=go.Heatmap(z=np.ones((5, 5))/25, xgap= 2, ygap= 2, colorscale='Blugrn'), layout=layout_hm), config={'displayModeBar': False}, id = 'heatmap'),
                    html.Div([
                          html.Button("Gauß-Filter", id= 'gauss', className = 'rounded-md px-2 py-2 border-2 border-green-600 bg-green-200 text-zinc-800 font-semibold h-12'),
                          html.Button("Binomial-Filter", id= 'binomial', className = 'rounded-md px-2 py-2 border-2 border-green-600 bg-green-200 text-zinc-800 font-semibold h-12')
                    ], className = 'flex flex-row gap-1')
                ], className = 'flex flex-col gap-1'),
                dcc.Graph(figure = go.Figure(data = [go.Image(z = Image.open('assets/image.jpg')), scatter], layout=layout_image), config={'displayModeBar': False}, id = 'image'),
                html.Div([
                     html.Button("Animation", id= 'animation', className = 'rounded-md px-2 py-2 border-2 border-green-600 bg-green-200 text-zinc-800 font-semibold h-12'),
                     html.Img(src = 'assets/arrow.png', style={'height': '40px'}),
                     html.Button("CV2 Filter", id= 'cv2', className = 'rounded-md px-2 py-2 border-2 border-green-600 bg-green-200 text-zinc-800 font-semibold h-12'),
                ]),
                dcc.Graph(figure = go.Figure(data=[go.Image(z = np.full((42, 42, 3), 255, dtype=np.uint8))], layout = layout_image), config={'displayModeBar': False}, id = 'image_output'),
                dcc.Store(id = 'previous_data', data = np.ones((5,5)).tolist()),
                dcc.Interval(id = 'animation_interval', interval = 50, max_intervals=38*38-1, disabled = True)
        ], className = 'flex flex-row gap-2 items-center justify-center'),

        html.P("2b) Erhöht man das Gewicht des mittleren Eintrags (Klicken der Heatmap) wird der Blur minimiert. Das ist nur logisch, da der Hauptpixel nun den größten Einfluss hat.", className = 'w-7/12'),
        html.P("3b) Erhöht man die Größe des Gauß/Binomial Filters, so wird der Blur verstärkt, da mehr Pixel mit einfließen und die mittleren weniger stark gewichtet werden.", className = 'w-7/12'),
    ], className = 'flex flex-col gap-0 justify-start w-full')
        
], className='flex flex-col justify-center items-center gap-8 m-4 text-zinc-800 font-semibold')


#Callbacks

def normalize_matrix(data):
    total_sum = sum(sum(row) for row in data)
    normalized_data = [[value / total_sum for value in row] for row in data]
    return normalized_data

@app.callback(
    Output('heatmap', 'figure'),
    Output('previous_data', 'data'),
    Input('heatmap', 'clickData'),
    State('previous_data', 'data'),
    prevent_initial_call = True
)
def update_heatmap(clickData, previous_data):
    if not clickData:
        return no_update, no_update
    
    clicked_point = clickData['points'][0]
    row = clicked_point['y']
    col = clicked_point['x']

    new_data = previous_data
    new_data[row][col] += 1

    normalized_data = normalize_matrix(new_data)

    heatmap_patch = Patch()
    heatmap_patch['data'][0]['z'] = normalized_data

    return heatmap_patch, new_data

#Aufgabe 2
@app.callback(
    Output('image_output', 'figure'),
    Input('cv2', 'n_clicks'),
    State('heatmap', 'figure'),
    State('image_data', 'data'),
    prevent_initial_call = True
)
def cv2_filter(n, data, image_data):
    if not n:
        return no_update
    
    kernel = np.array(data['data'][0]['z'])

    image = np.array(image_data, np.uint8)
    
    filtered_image = cv2.filter2D(image, -1, kernel)

    image_patch = Patch()
    image_patch['data'][0] = go.Image(z = filtered_image)
    
    return image_patch


@app.callback(
    Output('animation_interval', 'disabled'),
    Input('animation', 'n_clicks'),
    prevent_initial_call = True
)
def start_animation(n):
    if not n:
        return no_update
    return False

#Clientside Callback for faster Animation | Currently only works for kernel 5x5
app.clientside_callback(
'''
function (n, hmData, imageFigure, outputFigure){
    
    let positionKernel = [n % 38 + 2, Math.floor(n / 38) + 2];
    let imageData = imageFigure['data'][0]['z'];
    let kernel = hmData['data'][0]['z'];
    let newPixel = [0, 0, 0];
    for (let i = -2; i < 3; i++) {
        for (let j = -2; j < 3; j++) {
            let x = positionKernel[0] + i;
            let y = positionKernel[1] + j;

            let kernelValue = kernel[i + 2][j + 2];
            let pixelValue = imageData[y][x];
            newPixel[0] += pixelValue[0] * kernelValue;
            newPixel[1] += pixelValue[1] * kernelValue;
            newPixel[2] += pixelValue[2] * kernelValue;
        }
    }

    outputFigure['data'][0]['z'][positionKernel[1]][positionKernel[0]] = newPixel;

    imageFigure['data'][1]['x'] = [positionKernel[0]];
    imageFigure['data'][1]['y'] = [positionKernel[1]];

    return [JSON.parse(JSON.stringify(imageFigure)), JSON.parse(JSON.stringify(outputFigure))];
}
''',
    Output('image', 'figure', allow_duplicate=True),
    Output('image_output', 'figure', allow_duplicate=True),
    Input('animation_interval', 'n_intervals'),
    State('heatmap', 'figure'),
    State('image', 'figure'),
    State('image_output', 'figure'),
    prevent_initial_call = True
)

    
#Aufgabe 3

def gauss_filter(size, std):
    kernel = np.zeros((size, size))
    center = size//2

    for x in range(size):
        for y in range(size):
            kernel[x][y]= math.e**(-((x - center) ** 2 + (y - center) ** 2) / (2 * std ** 2))#/(2*math.pi * std ** 2)

    kernel = kernel / np.sum(kernel)
    return kernel

def binomial_filter(p):
    kernel = np.array([1, 1])
    
    for _ in range(p-1):
        kernel = np.convolve(kernel, [1, 1])
    
    kernel = kernel / 2**p
    kernel2d = np.outer(kernel, kernel)
    return kernel2d

@app.callback(
     Output('heatmap', 'figure', allow_duplicate=True),
     Input('gauss', 'n_clicks'),
     prevent_initial_call = True
)
def calc_gauss(n):
    if not n:
        return no_update
    heatmap_patch = Patch()
    size = 15 if n%3 == 0 else 3 if n%3==1 else 7
    heatmap_patch['data'][0]['z'] = gauss_filter(size, 2)
    return heatmap_patch

@app.callback(
     Output('heatmap', 'figure', allow_duplicate=True),
     Input('binomial', 'n_clicks'),
     prevent_initial_call = True
)
def calc_binomial(n):
    if not n:
        return no_update
    heatmap_patch = Patch()
    heatmap_patch['data'][0]['z'] = binomial_filter(4 + ((n-1)%3)*2)
    return heatmap_patch

if __name__ == '__main__':
    app.run_server(debug=True)


