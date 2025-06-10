""" 
Program to compute Voronoi diagram using JFA.

@author yisiox
@version September 2022
"""

import cupy as cp
from random import sample

# global variables
x_dim = 512
y_dim = 512
noSeeds = 1024

# diagram is represented as a 2d array where each element is
# x coord of source * y_dim + y coord of source
ping = cp.full((x_dim, y_dim), -1, dtype = int)
pong = None



import torch
import time

def process_tensors(tensor1, tensor2):
    # start_time = time.time()

    tensor2_unique = torch.unique(tensor2)
    mask = torch.isin(tensor1, tensor2_unique, assume_unique=True)
    tensor1[~mask] = -1

    end_time = time.time()
    # print(f"Computation time: {end_time - start_time} seconds")

    return tensor1

def test_performance():
    computation_times = []

    for _ in range(10):
        tensor1 = torch.randint(0, 40001, (1024, 1024)).cuda()
        tensor2 = torch.randint(5000, 15000, (512, 512)).cuda()

        process_tensors(tensor1, tensor2)



def voronoi_solve(texture, mask, device='cuda'):
    '''
        This is a warpper of the original cupy voronoi implementation
        The texture color where mask value is 1 will propagate to its
        neighbors.
        args:
            texture - A multi-channel tensor, (H, W, C)
            mask - A single-channel tensor, (H, W)
        return:
            texture - Propagated tensor
    '''
    h, w, c = texture.shape
    # hwc_texture = texture.permute(1,2,0)
    valid_pix_coord = torch.where(mask>0)

    indices = torch.arange(0, h*w).cuda().reshape(h, w)
    idx_map = -1 * torch.ones((h,w), dtype=torch.int64).cuda()
    idx_map[valid_pix_coord] = indices[valid_pix_coord]

    ping = cp.asarray(idx_map)
    pong = cp.copy(ping)
    ping = JFAVoronoiDiagram(ping, pong)

    voronoi_map = torch.as_tensor(ping, device=device)
    nc_voronoi_texture = torch.index_select(texture.reshape(h*w, c), 0, voronoi_map.reshape(h*w))
    voronoi_texture = nc_voronoi_texture.reshape(h, w, c)

    return voronoi_texture

def generateRandomSeeds(n):
    """
    Function to generate n random seeds.

    @param n The number of seeds to generate.
    """
    global ping, pong

    if n > x_dim * y_dim:
        print("Error: Number of seeds greater than number of pixels.")
        return

    # take sample of cartesian product
    coords = [(x, y) for x in range(x_dim) for y in range(y_dim)]
    seeds = sample(coords, n)
    for i in range(n):
        x, y = seeds[i]
        ping[x, y] = x * y_dim + y
    pong = cp.copy(ping)

displayKernel = cp.ElementwiseKernel(
        "int64 x",
        "int64 y",
        f"y = (x < 0) ? x : x % 103",
        "displayTransform")



voronoiKernel = cp.RawKernel(r"""
    extern "C" __global__
    void voronoiPass(const long long step, const long long xDim, const long long yDim, const long long *ping, long long *pong) {
        long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        long long stp = blockDim.x * gridDim.x;

        for (long long k = idx; k < xDim * yDim; k += stp) {
            long long dydx[] = {-1, 0, 1};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    long long dx = (step * dydx[i]) * yDim;
                    long long dy = step * dydx[j];
                    long long src = k + dx + dy;
                    if (src < 0 || src >= xDim * yDim) 
                        continue;
                    if (ping[src] == -1)
                        continue;
                    if (pong[k] == -1) {
                        pong[k] = ping[src];
                        continue;
                    }
                    long long x1 = k / yDim;
                    long long y1 = k % yDim;
                    long long x2 = pong[k] / yDim;
                    long long y2 = pong[k] % yDim;
                    long long x3 = ping[src] / yDim;
                    long long y3 = ping[src] % yDim;
                    long long curr_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                    long long jump_dist = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);
                    if (jump_dist < curr_dist)
                        pong[k] = ping[src];
                }
            }
        }
    }
    """, "voronoiPass")


'''

    y and x is actually w and h? (according to experiment result)

'''
def JFAVoronoiDiagram(ping, pong):
    # global ping, pong
    # compute initial step size
    x_dim, y_dim = ping.shape
    step = max(x_dim, y_dim) // 2
    # initalise frame number and display original state
    frame = 0
    # iterate while step size is greater than 0
    while step:
        voronoiKernel((min(x_dim, 1024),), (min(y_dim, 1024),), (step, x_dim, y_dim, ping, pong))
        ping, pong = pong, ping
        frame += 1
        step //= 2
        # displayDiagram(frame, ping)
    return ping