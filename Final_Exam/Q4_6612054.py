# Name - Win Yu Maung
# ID - 6612054  
# Sec - 541

N, M = map(int,input().split())
S, T = map(int,input().split())

from collections import deque
import heapq
maze = []

for _ in range(M):
    u, v, e = map(int, input().split())
    maze.append([u, v, e])

def findMinimumEnergy(n, maze, src, dst):
    graph = {i: [] for i in range(n)}
    for u, v, price in maze:
        graph[u].append((v, price))
    pq = [(0, src, 0)]
    visited = {}
    while pq:
        cost, node, stops = heapq.heappop(pq)
        if node == dst:
            return cost
        if (node, stops) in visited and visited[(node, stops)] <= cost:
            continue
        visited[(node, stops)] = cost
        for neighbor, price in graph[node]:
            new_cost = cost + price
            heapq.heappush(pq, (new_cost, neighbor, stops + 1))
    return -1

energy = findMinimumEnergy(N, maze, S, T)
if energy == -1:
    print("Impossible to reach the relic")
else:
    print(energy)

#I used uniform cost search because it is weighted graph energy and it needs minimun not the shortest
