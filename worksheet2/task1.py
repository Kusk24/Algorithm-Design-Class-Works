# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10000) 

n = int(input("Enter n: "))
x = [0] * n

def comb(i):
    if i == n:
        print(*x)
        return
    
    x[i] = 0
    comb(i + 1)
    
    x[i] = 1
    comb(i + 1)

comb(0)