# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import time

C = list(map(int, input("Enter coin list: ").split()))
n = int(input("Enter n: "))

start_time = time.time()

minCoin = [float('inf')] * (n + 1)
minCoin[0] = 0

for amount in range(1, n + 1):
    for c in C:
        if c <= amount:
            minCoin[amount] = min(minCoin[amount], 1 + minCoin[amount - c])

end_time = time.time()

print("Answer:", minCoin[n])
print("time =", end_time - start_time)
