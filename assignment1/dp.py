states = [(1, 0)]  # (product_v, sum_d)
ans = float('inf')

for i in range(N):
    new_states = []
    for p, s in states:
        new_states.append((p, s))                     # skip
        new_states.append((p * v[i], s + d[i]))       # take
    states = new_states

for p, s in states:
    if p != 1 or s != 0:   # exclude empty set
        ans = min(ans, abs(p - s))

print(ans)
