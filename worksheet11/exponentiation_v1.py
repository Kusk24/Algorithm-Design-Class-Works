# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

a = int(input("Enter a: "))
x = int(input("Enter x: "))

def expo(a, x):
    result = 1
    for i in range(x):
        result *= a
    return result % 2147483647
    
print(expo(a, x))