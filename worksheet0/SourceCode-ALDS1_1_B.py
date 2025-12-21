# 6612054
# Win Yu Maung
# 541

a, b = map(int, input().split())

while b != 0:
    a, b = b, a % b

print(a)