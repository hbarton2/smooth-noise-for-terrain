import numpy as np
import random
import math
import sys
from IPython.display import clear_output

def vectors(x, y, res_size):
    v1 = np.array([x, y])
    v2 = np.array([x, y-res_size])
    v3 = np.array([x-res_size, y])
    v4 = np.array([x-res_size, y-res_size])
    return (v1, v2, v3, v4)

def interpolate(t, x, y):
    return x + t*(y-x)

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def fade(val):
    # return val * val * (3 - 2 * val)
    return ((6*val - 15)*val + 10)*val*val*val

def fade_d(val):
    return 30*val*val*(val*(val-2)+1)

# def make_unit_vectors(width, res, seed, offset=0):
#     size = res + 1 + offset//res
#     unit_vectors = [[(0, 0) for _ in range(size)] for _ in range(size)]
#     for i in range(size):
#         for j in range(size):
#             # random.seed(int(str((i*(width//res)))+str(j*(width//res)))*seed)
#             # x = random.random() * 2 * math.pi
#             # unit_vectors[i][j] = np.array([math.cos(x), math.sin(x)])
#             unit_vectors[i][j] = get_unit_vector(i, j, seed)

#     return np.array(unit_vectors)

def get_unit_vector(x, y, seed):
    random.seed(int(str(x)+str(y))*seed)
    x = random.random() * 2 * math.pi
    return np.array([math.cos(x), math.sin(x)])

def make_unit_vectors(x, y, res_size, seed):
    xoff = x % res_size
    yoff = y % res_size
    x1 = x - xoff
    x2 = x + res_size - xoff
    y1 = y - yoff
    y2 = y + res_size - yoff
    return (get_unit_vector(x1, y1, seed), get_unit_vector(x1, y2, seed), get_unit_vector(x2, y1, seed), get_unit_vector(x2, y2, seed))
    

def norm(h, a, b):
    '''
    return: h values normalized in the range [a,b]
    '''
    if a==b:
        return np.full(h.shape, a)

    return ((h - np.min(h)) / (np.max(h) - np.min(h)) * abs(b-a)) - a

def norm_zero(h):
    ''' 
    return: values in an array normalized in the range [0, 1]
    '''
    return (h - np.min(h)) / (np.max(h) - np.min(h))

def perlin(x, y, res_size, seed):
    ''' 
    x: x coordinate to generate noise value for
    y: y coordinate to generate noise value for 
    res_size: the size of each box bounded by 4 gradients (total size of noise // resolution of noise)
    seed: random seed being used for generation

    return: a value of noise at a certain (x,y) coordinate
    '''
    v_x = x % res_size
    v_y = y % res_size

    vs = vectors(v_x, v_y, res_size)
    uv = make_unit_vectors(x, y, res_size, seed)

    dotTopLeft = dot(vs[0], uv[0])
    dotTopRight = dot(vs[1], uv[1])
    dotBottomLeft = dot(vs[2], uv[2])
    dotBottomRight = dot(vs[3], uv[3])

    x_t =  v_x / res_size
    y_t =  v_y / res_size

    x_t = fade(x_t)
    y_t = fade(y_t)

    z = interpolate(x_t, interpolate(y_t, dotTopLeft, dotTopRight), interpolate(y_t, dotBottomLeft, dotBottomRight))

    return z
    #return ((z - np.min(z)) / (np.max(z) - np.min(z)) * 2) - 1

def make_perlin_heightmap(width, resolution, x_offset=0, y_offset=0, seed=None):
    ''' 
    width: size on each axis for the heightmap
    resolution: resolution of the noise (how many gradients are generated on each axis)
    x_offset: offset of origin x coordinate from the true origin
    y_offset: offset of origin y coordinate from the true origin
    seed: random seed being used for repeatability

    return: perlin noise np array of size (width, width)
    '''

    if seed is None:
        seed = random.randint(0, sys.maxsize)

    hmap = np.zeros((width, width))

    for x in range(width):
        for y in range(width):
            hmap[x][y] = perlin(x+y_offset, y+x_offset, width//resolution, seed)
    return hmap

def make_perlin_gradient_heightmap(width, resolution, x_offset=0, y_offset=0, seed=None, sum_gradient=None):

    if seed is None:
        seed = random.randint(0, sys.maxsize)

    hmap = np.zeros((width, width))

    for x in range(width):
        for y in range(width):
            # Offsets are flipped because of the way arrays are
            hmap[x][y] = perlin(x+y_offset, y+x_offset, width//resolution, seed)

    # hmap = norm_zero(hmap)

    print(np.max(hmap))
    print(np.min(hmap))

    hmap_g = np.linalg.norm(np.array(np.gradient(hmap)), axis=0)

    hmap_g = 1 / (1+hmap_g)

    if sum_gradient is not None:
        hmap_g = hmap_g + sum_gradient

    hmap = hmap * hmap_g

    return hmap, hmap_g

def fractal_noise(width, _res, functions=None, comb_func=None, persistence=0.5, seed=None, x_offset=0, y_offset=0, use_perlin_gradient=False):
    h = np.zeros((width, width))

    if use_perlin_gradient:
        g_z = np.zeros((width, width))

    scale = 1

    if seed is not None:
        random.seed(seed)

    for r in range(len(_res)):
        res = _res[r]
        # print("Current scale", scale_current)
        print("Resolution", res, "starting.")
        
        if use_perlin_gradient:
            z, g_z = make_perlin_gradient_heightmap(width, res, x_offset, y_offset, seed=seed, sum_gradient=g_z)
        else:
            z = make_perlin_heightmap(width, res, x_offset, y_offset, seed=seed, )

        if functions is not None:
            cur_func = functions[r]
            if cur_func: z = cur_func(z)

        if r==0:
            h += z
        else:
            h = comb_func(h, z, scale)

        clear_output(wait=True)
        print("Resolution", res, "finished.")
        scale = scale*persistence

    return norm_zero(h)