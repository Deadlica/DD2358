import numpy as np
import time
from timeit import default_timer as timer

def checktick():
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 = time.time_ns()
        t2 = time.time_ns()
        while(t2 - t1) < 1e-16:
            t2 = time.time_ns()
        t1 = t2
        timesfound[i] = t1
    minDelta = 1000000
    Delta = np.diff(timesfound)
    minDelta = Delta.min()
    return minDelta

if __name__ == "__main__":
    print(checktick())
