# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10000)
import time

N = int(input("Enter N: "))
v = []
d = []

calls = 0

for i in range(N):
    vi, di = map(int, input("Enter vividness and dullness: ").split())
    v.append(vi)
    d.append(di)

memo = {}
def comb(i, product_v, sum_d, count):
    global calls
    if (i, product_v, sum_d, count) in memo:
        return memo[(i, product_v, sum_d, count)]
    
    calls += 1
    if i == N:
        if count == 0:
            result = float('inf')
        else:
            result = abs(product_v - sum_d)
    else:
        skip = comb(i + 1, product_v, sum_d, count)
        take = comb(i + 1, product_v * v[i], sum_d + d[i], count + 1)
        result = min(skip, take)
    
    memo[(i, product_v, sum_d, count)] = result
    return result

start = time.time()
ans = comb(0, 1, 0, 0)  
print(ans)
end = time.time()
print("functions calls = ", calls)
print("running time = ", end - start)