# Win Yu Maung
# 6612054
# 541

import time

int_list = list(map(int, input().split()))

start = time.process_time()

current_sum = 0
acc_list = []
for x in int_list:
    current_sum += x
    acc_list.append(current_sum)

max_sum = int_list[0]
n = len(int_list)

def Sum(i, j):
    if i == 0:
        return acc_list[j]
    return acc_list[j] - acc_list[i - 1]

for i in range(n):
    for j in range(i, n):   
        current = Sum(i, j)
        if current > max_sum:
            max_sum = current

finish = time.process_time()

print(max_sum)
print("running time =", finish - start)