# Name - Win Yu Maung
# ID - 6612054
# Sec - 541


import sys
sys.setrecursionlimit(10000)

values = list(map(int, input("Enter values: ").split()))
n = len(values)
x = [0] * n

def comb(i):
    total1 = 0
    total2 = 0
    if i == n:
        for j in range(n):
            if x[j] == 0:
                total1 += values[j]
            else:
                total2 += values[j]
        return abs(total1 - total2)
    else:
        x[i] = 0
        diff0 = comb(i+1)

        x[i] = 1
        diff1 = comb(i+1)

        return min(diff1, diff0)

ans = comb(0)

print("Minimal Difference:", ans)
    


#additional
# total = sum(values)
# def comb(i, sum1):
#     if i == n:
#         sum2 = total - sum1
#         return abs(sum1 - sum2)

#     diff1 = comb(i + 1, sum1 + values[i])

#     diff0 = comb(i + 1, sum1)

#     return min(diff1, diff0)

# ans = comb(0, 0)
# print("Minimal Difference:", ans)

