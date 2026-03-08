# Name : Win Yu Maung
# ID   : 6612054
# Sec  : 541

n = int(input("Enter number of treasure pieces : "))
values = list(map(int, input().split()))

sum = sum(values)    
rerult = []
print("Total :" , sum)
values.sort(reverse = True)

if sum % 2 == 0:
    print(values[0])
    for i in range(1 , len(values)):
        print(values[i])
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