# Name : Win Yu Maung
# ID   : 6612054
# Sec  : 541

import sys
sys.setrecursionlimit(20001)

n = 9
d = 3 
p = []

for i in range(d):
    p += list(map(int, input().split()))

import copy

def valid(i, j):
    if i >= 0 and i < d and j >= 0 and j < d:
        return True
    else:
        return False


class state:
    def __init__(self, p):
        self.p = copy.deepcopy(p)
        self.g = 0
        self.h = 1000000000


def manhatton(p):
    h = 0
    for i in range(len(p)):
        if p[i] != 0:
            tr = i // d
            tc = i % d
            r = p[i] // d
            c = p[i] % d
            h += abs(tr - r) + abs(tc - c)
    return h


adj = [(0,-1),(0,1),(1,0),(-1,0)]


def successor(s):
    succ = []
    for i in range(len(s.p)):
        if s.p[i] == 0:
            hole = i
            break

    r = hole // d
    c = hole % d

    for a in adj:
        i = r + a[0]
        j = c + a[1]

        if valid(i, j):
            target = i * d + j
            m = s.p[:]
            m[target], m[hole] = m[hole], m[target]

            u = state(m)
            u.g = s.g + 1
            u.h = manhatton(m)
            succ.append(u)

    return succ

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