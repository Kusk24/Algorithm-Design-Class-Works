# Name - Win Yu Maung
# ID - 6612054  
# Sec - 541

import heapq

N = int(input())
coor = []
for _ in range(N):
    x,y = map(int, input().split())
    coor.append([x,y])

def euclidean(p1, p2):
    x = (p2[0] - p1[0])**2
    y = (p2[1] - p1[1])**2
    return (x + y)**0.5

def minCostConnectPointsPrim(points):
    n = len(points)  
    visited = [False] * n
    min_cost = 0
    pq = [(0,0)]  
    edges_used = 0
    
    while pq and edges_used < n:
        cost, point = heapq.heappop(pq)
        if visited[point]:
            continue
        visited[point] = True
        min_cost += cost
        edges_used += 1
        for next_point in range(n):
            if not visited[next_point]:
                dist = euclidean(points[point], points[next_point])
                heapq.heappush(pq, (dist, next_point))
    return min_cost

print(minCostConnectPointsPrim(coor))

#I used prim algorithm becauses it is MST problem with implicit edges
