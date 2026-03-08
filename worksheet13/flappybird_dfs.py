# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

H, T = map(int, input().split())

grid = []
for _ in range(T):
    grid.append(list(map(int, input().split())))

visited = set()

def dfs(t, h):

    if t == T:
        return True
    
    if (t, h) in visited:
        return False
    
    visited.add((t, h))
    
    # same height
    if 0 <= h < H and grid[t][h] == 0:
        if dfs(t+1, h):
            return True

    # up
    if 0 <= h+1 < H and grid[t][h+1] == 0:
        if dfs(t+1, h+1):
            return True

    # down
    if 0 <= h-1 < H and grid[t][h-1] == 0:
        if dfs(t+1, h-1):
            return True

    return False


for h in range(H):
    visited.clear()
    if dfs(0, h):
        print(h+1)
        break