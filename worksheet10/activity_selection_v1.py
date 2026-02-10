# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

n = int(input())
activities = []

for i in range(n):
    s, f = input().split()
    activities.append((int(s), int(f)))

activities.sort(key=lambda x: x[0])

count = 1
last_finish = activities[0][1]

for i in range(1, n):
    if activities[i][0] > last_finish:
        count += 1
        last_finish = activities[i][1]

print(count)