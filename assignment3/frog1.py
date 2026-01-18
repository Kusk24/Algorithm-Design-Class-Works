# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

N = int(input(""))
h = list(map(int, input("").split()))

dp = [0] * N

dp[0] = 0
dp[1] = abs(h[1] - h[0])

for i in range(2, N):
    cost_from_prev = dp[i - 1] + abs(h[i] - h[i - 1])
    cost_from_prev2 = dp[i - 2] + abs(h[i] - h[i - 2])
    dp[i] = min(cost_from_prev, cost_from_prev2)

print(dp[N - 1])