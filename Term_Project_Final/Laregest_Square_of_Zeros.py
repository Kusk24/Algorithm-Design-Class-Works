import sys
input = sys.stdin.readline

H, W = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(H)]

dp = [[0] * W for _ in range(H)]

max_side = 0

for i in range(H):
    for j in range(W):
        if matrix[i][j] == 0:
            if i == 0 or j == 0:
                dp[i][j] = 1
            else:
                dp[i][j] = min(
                    dp[i-1][j],      # top
                    dp[i][j-1],      # left
                    dp[i-1][j-1]     # diagonal
                ) + 1
            max_side = max(max_side, dp[i][j])

print(max_side * max_side)