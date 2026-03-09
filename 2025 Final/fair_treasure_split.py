# Name : Win Yu Maung
# ID   : 6612054
# Sec  : 541

n = int(input())
values = list(map(int, input().split()))

total = sum(values)

if total % 2 != 0:
    print("Can't split fairly")
else:
    target = total // 2
    parent = [-1] * (target + 1)
    parent[0] = 0 

    for i in range(n):
        v = values[i]
        for s in range(target, v - 1, -1):
            if parent[s] == -1 and parent[s - v] != -1:
                parent[s] = i 

    if parent[target] != -1:
        group1 = []
        temp_target = target
        used_indices = set()
        
        while temp_target > 0:
            idx = parent[temp_target]
            group1.append(values[idx])
            used_indices.add(idx)
            temp_target -= values[idx]
        
        group2 = [values[i] for i in range(n) if i not in used_indices]
        
        print(*(group1))
    else:
        print("Can't split fairly")
   
# Question

# A pirate discovered n pieces of treasure, each having a certain value.
# The pirate wants to divide the treasure into two groups with equal total value so that the split is fair.

# Write a program that determines whether the treasure can be split fairly.
        
# Input
# n
# v1 v2 v3 ... vn

# n = number of treasure pieces
# vi = value of each treasure piece

# Output
# If the treasures can be split equally, print the values.
# Otherwise print:
# Can't split fairly