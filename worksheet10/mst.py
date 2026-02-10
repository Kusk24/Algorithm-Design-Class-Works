# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


def union(parent, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        parent[yroot] = xroot

V, E = map(int, input().split())
edges = []

for _ in range(E):
    u, v, w = map(int, input().split())
    edges.append((w, u, v))   
edges.sort()

parent = [i for i in range(V)]

mst_weight = 0
edge_used = 0

for w, u, v in edges:
    if find(parent, u) != find(parent, v):
        union(parent, u, v)
        mst_weight += w
        edge_used += 1

        if edge_used == V - 1:
            break

print(mst_weight)
