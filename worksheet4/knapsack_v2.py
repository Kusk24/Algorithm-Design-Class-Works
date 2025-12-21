# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

N,M = map(int, input().split())
w = list(map(int, input().split()))
v = list(map(int, input().split()))
calls = 0

def maxVal(i,C):
    global calls
    calls += 1
    if i == N: 
        return 0
    else:
        skip = maxVal(i+1, C)
        if w[i] <= C:
            take = v[i] + maxVal(i+1, C-w[i])
        else:
            take = -1
        return max(skip, take)
    

import time
start_time = time.time() 
print(maxVal(0,M))
end_time = time.time()
print("running time" , end_time - start_time)
print("function calls:", calls)