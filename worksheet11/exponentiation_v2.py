# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

a = int(input("Enter a: "))
x = int(input("Enter x: "))

def expo(a, x):
    if x == 0:
        return 1
    
    if x%2 == 0:
        answer = (expo(a, x // 2) * expo(a, x // 2))
    else:
        answer = a * (expo(a, x // 2) * expo(a, x // 2))
        
    return answer % 2147483647
    
print(expo(a, x))