# Name : Win Yu Maung
# ID   : 6612054
# Sec  : 541

n, d = map(int, input().split())

investments = []
for _ in range(d):
    f, e = map(int, input().split()) 
    investments.append((f, e)) 

dp = [0] * (n + 1)

for profit, cost in investments:
    for i in range(cost, n + 1):
        if dp[i - cost] + profit > dp[i]:
            dp[i] = dp[i - cost] + profit

print(dp[n])


# Question

# A venture capitalist has n units of money to invest.
# There are d investment opportunities.
# Each investment has:

# e = cost per share
# f = profit per share

# The investor may buy multiple shares of each investment.
# Your goal is to maximize total profit.

# Input
# n d
# f1 e1
# f2 e2
# ...
# fd ed

# n = total money
# d = number of investment options
# f = profit
# e = cost

# Output
# Print the maximum profit possible.