# Win Yu Maung
# 6612054
# 541

import time

int_list = list(map(int, input().split()))

start = time.process_time()

current_sum = int_list[0]
max_sum = int_list[0]

for x in int_list[1:]:
    current_sum = max(x, current_sum + x)
    max_sum = max(max_sum, current_sum)

finish = time.process_time()

print(max_sum)
print("running time =", finish - start)
