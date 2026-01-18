# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10001)
import time

FLAT = 0
UPPER2 = 1
LOWER2 = 2

L = int(input())

m = {}

def nWays(d, s):
    if (d, s) in m:
        return m[(d,s)]

    if d == L:
        if s == FLAT:
            return 1
        else:
            return 0
    else:
        counter = 0
        if s == FLAT:
            counter += nWays(d+1, UPPER2)
            counter += nWays(d+1, LOWER2)
            if d+2 <= L:
                counter += nWays(d+2, FLAT)
        else:  # s is either UPPER2 or LOWER2
            counter += nWays(d+1, FLAT)
            if d+2 <= L:
                counter += nWays(d+2, s)
        m[(d, s)] = counter
        return counter

start = time.time()
print(nWays(0, FLAT))
end = time.time()
print("running time", end - start)
# print(m)