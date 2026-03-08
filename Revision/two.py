
MOD = 2147483647

def multiply(A, B):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % MOD,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % MOD],
         [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % MOD,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % MOD]
        ]

def power(M, n):
    if n == 1:
        return M

    half = power(M, n // 2)
    result = multiply(half, half)

    if n%2 == 1:
        result = multiply(result, half)
    
    return result

def fibonacci(n):
    if n == 0:
        return 0
    
    M = [[1,1],[1,0]]
    result = power(M, n-1)

    return result[0][0] % MOD

n = int(input())
print(fibonacci(n))