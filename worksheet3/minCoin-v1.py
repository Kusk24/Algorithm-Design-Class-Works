# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
import time
sys.setrecursionlimit(10000)

C = list(map(int, input("Enter coin list: ").split()))
n = int(input("Enter n: "))
calls = 0

def minChange(n, C):
    global calls
    calls += 1
    if n == 0:
        return 0
    
    v = float('inf')

    for c in C:
        if c <= n:
            v = min( minChange(n-c,C) + 1, v)
    
    return v

start_time = time.time()
print("Answer: ", minChange(n,C))
end_time = time.time()
print("recursive calls = " + str(calls))
print("time = ", end_time - start_time)