# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10000)

N,M = map(int, input().split())
w = list(map(int, input().split()))
v = list(map(int, input().split()))

x = [0]*N

memo = {}
calls = 0

def maxVal(i,C):
    global calls
    calls += 1
    
    if (i, C) in memo:
        return memo[(i, C)]
    
    if i == N: 
        result = 0
    else:
        skip = maxVal(i+1, C)
        if w[i] <= C:
            take = v[i] + maxVal(i+1, C-w[i])
        else:
            take = -1
        result = max(skip, take)
    
    memo[(i, C)] = result
    return result

import time
start_time = time.time() 
max_value = maxVal(0, M)
end_time = time.time()
print(max_value)
print("running time:", end_time - start_time)
print("function calls:", calls)