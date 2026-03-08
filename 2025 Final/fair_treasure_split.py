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

    dp = [False] * (target + 1)
    dp[0] = True

    for v in values:
        for s in range(target, v-1, -1):
            dp[s] = dp[s] or dp[s - v]

    if dp[target]:
        print("Can split fairly")
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