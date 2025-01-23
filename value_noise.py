import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
import plotly.graph_objects as go
import plotly.offline as pyo

def get_noise_value(x, y, seed):
    random.seed(int(str(x)+str(y))*seed)
    return random.random()

def interp(x, x0, x1):
    '''
    Assuming x is a point value between two points for interpolation, 
    the input to this function should be x - x0 / x1 - x0, or in the context 
    of what I have used it would be v_x / res_size
    '''
    return x0*(2*(x**3)-3*(x**2)+1) + x1*(-2*(x**3)+3*(x**2)) 
    # + d0*(x**3-2*(x**2)+x) + d1*(x**3-x**2)

def fade(val):
    return ((6*val - 15)*val + 10)*val*val*val

def value_noise_height(x, y, res_size, seed):
    v_x = x % res_size
    v_y = y % res_size

    x1 = x-v_x
    x2 = x1+res_size
    y1 = y-v_y
    y2 = y1+res_size

    # x2-x = (x-v_x+res_size-x) = (res_size-v_x)

    v1 = get_noise_value(x1, y1, seed)
    v2 = get_noise_value(x1, y2, seed)
    v3 = get_noise_value(x2, y1, seed)
    v4 = get_noise_value(x2, y2, seed)

    # x_frac = v_x / res_size
    # y_frac = v_y / res_size

    # i13 = interp(x_frac, v1, v3)
    # i24 = interp(x_frac, v2, v4)

    # y_interp = interp(y_frac, i13, i24)

    # return y_interp

    # My attempt at implementing bicubic interpolation (doesn't work)

    # px00 = px10 = (v3 - v1) * v_x / res_size
    # px01 = px11 = (v4 - v2) * v_x / res_size
    # py00 = py01 = (v2 - v1) * v_y / res_size
    # py10 = py11 = (v4 - v3) * v_y / res_size

    # pxy = ((v4 - v3) - (v2 - v1)) * v_x * v_y / (res_size * res_size)

    # c_array = np.array([[1, 0, 0, 0],
    #                          [0, 0, 1, 0],
    #                          [-3, 3, -2, -1],
    #                          [2, -2, 1, 1]])
    # v_array = np.array([[v1, v2, py00, py01],
    #                          [v3, v4, py10, py11],
    #                          [px00, px01, pxy, pxy],
    #                          [px10, px11, pxy, pxy]])

    # coefficients = c_array @ v_array @ c_array.T

    # return np.array([1, x, x**2, x**3]) @ coefficients @ np.array([[1],[y],[y**2],[y**3]])
    
    

    # Linear Interpolation

    x_lerp1 = ((x2-x) / res_size * v1) + ((x-x1) / res_size * v3)
    x_lerp2 = ((x2-x) / res_size * v2) + ((x-x1) / res_size * v4)

    return ((y2-y) / res_size * x_lerp1) + ((y-y1) / res_size * x_lerp2)

def value_noise(width, resolution, seed=None, x_offset=0, y_offset=0):
    if seed is None:
        seed = random.randint(0, sys.maxsize)

    if seed < 1:
        return -1

    hmap = np.zeros((width, width))

    x_points = np.linspace(x_offset, x_offset+(resolution * (width // resolution)), resolution+1, dtype=int)
    y_points = np.linspace(y_offset, y_offset+(resolution * (width // resolution)), resolution+1, dtype=int)

    z = np.zeros((resolution+1, resolution+1))

    for x in range(len(z)):
        for y in range(len(z)):
            z[x][y] = get_noise_value(x_points[x], y_points[y], seed)

    for x in range(x_offset, x_offset+width):
        for y in range(y_offset, y_offset+width):
            hmap[x][y] = interpolate.interpn((x_points, y_points), z, [x,y], method='cubic')

    return hmap

    # for x in range(x_offset, x_offset+width):
    #     for y in range(y_offset, y_offset+width):
    #         hmap[x][y] = value_noise_height(x, y, width//resolution, seed)

    # return hmap

def norm_zero(h):
    ''' 
    return: values in an array normalized in the range [0, 1]
    '''
    return (h - np.min(h)) / (np.max(h) - np.min(h))

hmap = value_noise(100, 10, seed=1)

# for i in range(4, 21, 4):
#     hmap += value_noise(100, i)

hmap = norm_zero(hmap)

plt.imshow(hmap, cmap='grey')
plt.yticks([])
plt.xticks([])
plt.savefig('value_noise_bicubic', bbox_inches='tight', pad_inches=0)
plt.show()

# Call function to view heightmap in 3D


# import plotly.io as pio

# pio.renderers.default = 'browser'

# fig = go.Figure(data=[go.Surface(z=hmap)])

# fig.update_layout(scene=dict(zaxis=dict(range=[-0.5,5])))

# fig.update_traces(colorscale=["blue", "teal", "green", "green", "grey", "grey", "white", "white"])

# fig.show()