M, N = map(int,input().split())

matrix = []

for _ in range(M):
    matrix.append((list(map(int, input().split()))))

visited = [[False] * N for _ in range(M)]

def dfs(row, col):
    stack = [(row, col)]
    visited[row][col] = True
    size = 1

    while stack:
        x, y = stack.pop() 

        axis = [(1,0), (-1,0), (0,1), (0,-1)]
        for dx, dy in axis:
            newx, newy = x+dx, y+dy

            if 0 <= newx < M and 0 <= newy < N:
                if not visited[newx][newy] and matrix[newx][newy] == 1:
                    visited[newx][newy] = True
                    stack.append((newx, newy))
                    size +=1
    
    return size

max_cloud = 0

for i in range(M):
    for j in range(N):
        if matrix[i][j] == 1 and not visited[i][j]:
            max_cloud = max(max_cloud, dfs(i,j))
        
print(max_cloud)


### Recursive dfs version
# def dfs(row, col):
#     visited[row][col] = True
#     size = 1

#     for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
#         newx, newy = row + dx, col + dy

#         if 0 <= newx < M and 0 <= newy < N:
#             if not visited[newx][newy] and matrix[newx][newy] == 1:
#                 size += dfs(newx, newy)

#     return size