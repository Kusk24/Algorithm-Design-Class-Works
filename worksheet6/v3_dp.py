# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

N, W = map(int, input().split())
v = []
w = []
for _ in range(N):
    vi, wi = map(int, input().split())
    v.append(vi)
    w.append(wi)

dp = [[0]*(W+1) for i in range(N+1)]

for i in range(1, N+1):
    for C in range(0, W+1):
        skip = dp[i-1][C]
        
        take = 0
        if w[i-1] <= C:
            take = v[i-1] + dp[i-1][C - w[i-1]]
        
        dp[i][C] = max(skip, take)

print(dp[N][W])