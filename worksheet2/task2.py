# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import sys
sys.setrecursionlimit(10000) 

n = int(input("Enter n: "))
k = int(input("Enter k: "))
x = [0] * n

# No.3
# def comb(i):
#     if i == n:
#         print(*x)
#         return 1
#     x[i] = 0
#     c1 = comb(i + 1)

#     x[i] = 1
#     c2 = comb(i + 1)

#     return c1 + c2  

# total = comb(0)
# print("Total combinations = ", total)



# No.4
# def comb(i, count):
#     if i == n:
#         if count == k:
#             print(*x)
#             return 1
#         return 0
#     else:    
#         x[i] = 0
#         c1 = comb(i + 1, count)

#         x[i] = 1
#         c2 = comb(i + 1, count + 1)

#         return c1 + c2  

# total = comb(0,0)
# print("Total combinations = ", total)


# No.5
def comb(i):
    if i == n:
        print(*x)
        return 1
    
    x[i] = 0
    c1 = comb(i + 1)

    x[i] = 1
    c2 = comb(i + 1)

    x[i] = 2
    c3 = comb(i + 1)
    return c1 + c2 + c3 

total = comb(0)
print("Total combinations = ", total)