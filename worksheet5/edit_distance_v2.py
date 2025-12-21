# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10000)

import time
start = time.time()

A = input("Enter the letter A: ")
B = input("Enter the letter B: ")
calls = 0

memo = {}

def edit(i, j):
    global calls
    calls += 1

    if (i, j) in memo:
        return memo[(i, j)]

    if i == 0:
        memo[(i, j)] = j
        return j
    if j == 0:
        memo[(i, j)] = i
        return i

    if A[i-1] == B[j-1]:
        memo[(i, j)] = edit(i-1, j-1)
        return memo[(i, j)]

    memo[(i, j)] = 1 + min(
        edit(i-1, j-1),  # Replace
        edit(i-1, j),    # Delete
        edit(i, j-1)     # Insert
    )

    return memo[(i, j)]

result = edit(len(A), len(B))
print(result)

end = time.time()
print("running time = ", end - start)
print("function count = ", calls)
