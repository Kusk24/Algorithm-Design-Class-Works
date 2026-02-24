# Name - Win Yu Maung
# ID - 6612054  
# Sec - 541

A = list(map(int, input().split()))

#divide
def maxSubArraySum(A, p, r):
    if p == r:
        return A[p]
    
    q = (p + r) // 2
    
    left = maxSubArraySum(A, p, q)        
    right = maxSubArraySum(A, q + 1, r)
    middle = maxCrossingSum(A, p, q, r)
    
    return max(left, right, middle)

#conquer
def maxCrossingSum(A, p, q, r):
    sum_left = 0
    best_left = float('-inf')
    for i in range(q, p - 1, -1):  
        sum_left += A[i]
        best_left = max(best_left, sum_left)
    
    sum_right = 0
    best_right = float('-inf')
    for i in range(q + 1, r + 1):  
        sum_right += A[i]
        best_right = max(best_right, sum_right)
    
    return best_left + best_right


print(maxSubArraySum(A, 0, len(A) - 1))
