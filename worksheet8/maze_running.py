# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

from collections import deque

M, N = map(int, input().split())

start_r, start_c = map(int, input().split())
end_r, end_c = map(int, input().split())

maze = []
for _ in range(M):
    maze.append(list(map(int, input().split())))

visited = [[False] * N for _ in range(M)]

directions = [(1,0), (-1,0), (0,1), (0,-1)]

def valid(r, c):
    return 0 <= r < M and 0 <= c < N and maze[r][c] == 0 and not visited[r][c]

def bfs():
    queue = deque()
    queue.append((start_r, start_c, 0))
    visited[start_r][start_c] = True

    while queue:
        r,c, step = queue.pop.left()

        if r == end_r and c == end_c:
            return step
        
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            if valid(nr, nc):
                visited[nr][nc] = True
                queue.append((nr, nc, step+1))

    return -1

print(bfs())