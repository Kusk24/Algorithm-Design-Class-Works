# Name : Win Yu Maung
# ID   : 6612054
# Sec  : 541

import sys
sys.setrecursionlimit(100000)

d = 3 
start_p = []
for _ in range(d):
    start_p += list(map(int, input().split()))

goal_p = [1, 2, 3, 4, 5, 6, 7, 8, 0]

def manhattan(p):
    h = 0
    for i in range(len(p)):
        val = p[i]
        if val != 0:
            # Current position
            curr_r, curr_c = i // d, i % d
            # Goal position for value 'val'
            # In [1,2,3,4,5,6,7,8,0], '1' is at index 0, '2' at 1...
            goal_idx = val - 1
            goal_r, goal_c = goal_idx // d, goal_idx % d
            h += abs(curr_r - goal_r) + abs(curr_c - goal_c)
    return h

def get_successors(p):
    succ = []
    hole = p.index(0)
    r, c = hole // d, hole % d
    
    for dr, dc in [(0,-1), (0,1), (1,0), (-1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < d and 0 <= nc < d:
            target = nr * d + nc
            new_p = p[:]
            new_p[hole], new_p[target] = new_p[target], new_p[hole]
            succ.append(new_p)
    return succ

def search(path, g, threshold):
    curr = path[-1]
    f = g + manhattan(curr)
    
    if f > threshold:
        return f
    if curr == goal_p:
        return "FOUND"
    
    min_val = float('inf')
    for s in get_successors(curr):
        if s not in path:
            path.append(s)
            t = search(path, g + 1, threshold)
            if t == "FOUND":
                return "FOUND"
            if t < min_val:
                min_val = t
            path.pop()
    return min_val

def ida_star(root):
    threshold = manhattan(root)
    path = [root]
    while True:
        t = search(path, 0, threshold)
        if t == "FOUND":
            return len(path) - 1
        if t == float('inf'):
            return -1
        threshold = t

print(ida_star(start_p))

# Question

# You are given a 3 × 3 sliding puzzle (8-puzzle).
# The puzzle contains numbers 1–8 and an empty space represented by 0.
# You may move the empty space:

# up
# down
# left
# right

# Your task is to determine the minimum number of moves required to reach the goal configuration:
# 1 2 3
# 4 5 6
# 7 8 0

# The solution must use IDA* search with Manhattan distance heuristic.

# Input
# Three lines representing the puzzle state.
# Example:
# 1 2 3
# 4 5 6
# 7 0 8

# Output
# Print the minimum number of moves required.