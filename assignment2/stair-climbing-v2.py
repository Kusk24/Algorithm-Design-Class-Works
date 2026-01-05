#Name - Win Yu Maung
#ID - 6612054
#Sec - 541

import sys
import time
sys.setrecursionlimit(10000)

start = time.time()

N = int(input("Enter N: "))
c = list(map(int, input("Enter costs: ").split()))

memo = {}

def minCost(i):
    if i < 0:
        return 0
    if i in memo:
        return memo[i]
    
    if i == 0 or i == 1:
        memo[i] = c[i]
    else:
        memo[i] = c[i] + min(minCost(i-1), minCost(i-2))
    
    return memo[i]

if N == 1:
    result = c[0]
else:
    result = min(minCost(N-1), minCost(N-2))

print("Minimum cost:", result)

finish = time.time()
print("running time =", finish - start)