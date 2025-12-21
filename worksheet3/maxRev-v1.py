# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
import time
sys.setrecursionlimit(10000)

pList = list(map(int, input("Enter rev prices: ").split()))
l = int(input("Enter length L: "))
calls = 0

def maxRev(l):
    global calls
    calls += 1
    
    if l == 0:
        return 0
    
    v = float('-inf')

    for i in range(1, len(pList)+1):
        if i <= l:
            v = max( maxRev(l-i) + pList[i-1] , v)
    
    return v

start_time = time.time()
print("Answer: ",maxRev(l))
end_time = time.time()
print("recursive calls = ", str(calls))
print("time = ", end_time - start_time)