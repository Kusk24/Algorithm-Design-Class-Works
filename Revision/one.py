
N = list(map(int,input().split()))

import heapq

heapq.heapify(N)

cost = 0

while (len(N)) > 1:
    first = heapq.heappop(N)
    second = heapq.heappop(N)

    tempcost = first + second
    heapq.heappush(N, tempcost)

    cost += tempcost

print(cost)
