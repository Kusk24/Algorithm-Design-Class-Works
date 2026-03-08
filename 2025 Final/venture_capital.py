# Name : Win Yu Maung
# ID   : 6612054
# Sec  : 541

n, d = map(int, input().split())

funds = []

for _ in range(d):
    f, e = map(int, input().split())
    funds.append((f / e, e, f))

funds.sort(reverse=True)

total_profit = 0
remaining_money = n

for ratio, e, f in funds:
    shares = remaining_money // e
    total_profit += shares * f
    remaining_money -= shares * e

print(total_profit)


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