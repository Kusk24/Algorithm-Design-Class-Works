# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10000)

N,M = map(int, input().split())
w = list(map(int, input().split()))
v = list(map(int, input().split()))

x = [0]*N
calls = 0

def comb(i):
    global calls
    calls += 1
    if i == N:
        sw = sv = 0
        for j in range(N):
            if x[j] == 1:
                sw += w[j]
                sv += v[j]
        if sw > M:
            return -1
        else: 
            return sv
    else: 
        x[i] = 0
        a = comb(i+1)
        x[i] = 1
        b = comb(i+1)
        return max(a,b)

import time
start_time = time.time() 
print(comb(0))
end_time = time.time()
print("running time" , end_time - start_time)
print("function calls:", calls)