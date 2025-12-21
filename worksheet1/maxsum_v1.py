# Win Yu Maung
# 6612054
# 541

import time

int_list = list(map(int, input().split()))

start = time.process_time()

def Sum(x, i, j):
    s = 0
    for k in range(i, j+1):
        s += x[k]
    return s

max_sum = int_list[0]
for i in range(len(int_list)):
    for j in range(len(int_list)):
        sum = Sum(int_list, i, j)
        if sum >= max_sum:
            max_sum = sum


finish = time.process_time()
print(max_sum)
print("running time =", finish - start)