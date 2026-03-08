# Name : Zwe Nyan Win
# ID   : 6540179
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
    remaining_money -= shares * f

print(total_profit)
