# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

n = int(input())
prices = list(map(float, input().split()))

dp = [float('inf')] * (n + 1)
dp[0] = 0.0

for i in range(1, n + 1):
    dp[i] = dp[i - 1] + prices[i - 1]
    
    if i >= 2:
        cost = sum(prices[i - 2:i]) - 0.5 * min(prices[i - 2:i])
        dp[i] = min(dp[i], dp[i - 2] + cost)
    
    if i >= 3:
        cost = sum(prices[i - 3:i]) - min(prices[i - 3:i])
        dp[i] = min(dp[i], dp[i - 3] + cost)

print("%.1f" % dp[n])