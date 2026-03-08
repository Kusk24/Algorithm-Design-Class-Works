# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

from collections import deque

H, T = map(int, input(().split()))

grid = []
for _ in range(T):
    grid.append(list(map(int, input().split())))

def bfs(start):
    q = deque()
    q.append((0, start))

    visited = set()
    visited.add((0, start))

    while q:
        t, h = q.popleft()

        if t == T:
            return True
        
        if 0 <= h < H:
            if grid[t][h] == 0 and (t+1, h) not in visited:
                visited.add((t+1, h))
                q.append((t+1, h))
        
        if 0 <= (h + 1) < H:
            if grid[t][h + 1] == 0 and (t+1, h+1) not in visited:
                visited.add((t+1, h+1))
                q.append((t+1, h+1))
            
        if 0 <= (h - 1) < H:
            if grid[t][h - 1] == 0 and (t+1, h-1) not in visited:
                visited.add((t+1, h-1))
                q.append((t+1, h-1))
        
    return False

for h in range(H):
    if bfs(h):
        print(h+1)
        break





# push (0, start_height)

# while queue not empty
#     pop state

#     if interval == T
#         success

#     try moves:
#         height
#         height + 1
#         height - 1

#         if next cell is 0
#             push (interval+1, new_height)