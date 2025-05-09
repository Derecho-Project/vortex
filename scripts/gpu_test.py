from numba import jit, cuda 
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer       

@cuda.jit                      
def func(a):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < 10000000: 
        a[pos] += 1

if __name__=="__main__": 
    n = 10000000                            
    a = np.ones(n, dtype = np.float64) 

    threadsperblock = 10
    blockspergrid = (n + (threadsperblock - 1))
    for i in range(10000000):
        func[blockspergrid, threadsperblock](a) 
