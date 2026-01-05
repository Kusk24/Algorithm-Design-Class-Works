#Name - Win Yu Maung
#ID - 6612054
#Sec - 541

import sys
import time
sys.setrecursionlimit(10000)

start = time.time()

N = int(input("Enter N: "))
c = list(map(int, input("Enter costs: ").split()))

if N == 1:
    result = c[0]
else:
    dp = [0] * N
    dp[0] = c[0]
    dp[1] = c[1]
    
    for i in range(2, N):
        dp[i] = c[i] + min(dp[i-1], dp[i-2])
    
    result = min(dp[N-1], dp[N-2])

print("Minimum cost:", result)

finish = time.time()
print("running time =", finish - start)