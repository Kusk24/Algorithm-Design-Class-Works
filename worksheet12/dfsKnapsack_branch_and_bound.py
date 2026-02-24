# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
import time
sys.setrecursionlimit(10000)

class obj:
    def __init__(self, w, v):
        self.w = w
        self.v = v
        self.r = v / w

x = input().split()
N = int(x[0])
M = int(x[1])

w = input().split()
v = input().split()

item = []
for i in range(N):
    item.append(obj(int(w[i]), int(v[i])))

item.sort(key=lambda x: x.r, reverse=True)

maxV = 0
call_count = 0

def Bound(i, C):
    global item, N
    
    total_weight = 0
    total_value = 0
    j = i
    fraction = 1.0
    
    while j < N and fraction == 1.0:
        weight_to_take = min(C - total_weight, item[j].w)
        fraction = float(weight_to_take) / item[j].w
        total_weight += fraction * item[j].w
        total_value += fraction * item[j].v
        j += 1
    
    return total_value

def knapsack_dfs(index, current_weight, current_value):
    global maxV, item, N, M, call_count
    
    if current_weight > M:
        return
    
    call_count += 1
    
    if index == N:
        if current_value > maxV:
            maxV = current_value
        return
    
    remaining_capacity = M - current_weight
    if current_value + Bound(index, remaining_capacity) <= maxV:
        return
    
    bound_if_take = -1
    remaining_after_take = M - (current_weight + item[index].w)
    
    if current_weight + item[index].w <= M:
        value_if_take = current_value + item[index].v
        bound_if_take = value_if_take + Bound(index + 1, remaining_after_take)
    
    value_if_skip = current_value
    remaining_after_skip = M - current_weight
    bound_if_skip = value_if_skip + Bound(index + 1, remaining_after_skip)
    
    if bound_if_take > bound_if_skip:
        knapsack_dfs(index + 1, current_weight + item[index].w, current_value + item[index].v)
        knapsack_dfs(index + 1, current_weight, current_value)
    else:
        knapsack_dfs(index + 1, current_weight, current_value)
        knapsack_dfs(index + 1, current_weight + item[index].w, current_value + item[index].v)

start_time = time.process_time()
knapsack_dfs(0, 0, 0)
end_time = time.process_time()

print("Max Value:", maxV)
print("Number of calls in full branch and bound DFS:", call_count)
print("Running time:", end_time - start_time)
