# Name : Zwe Nyan Win
# ID   : 6540179
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
        

   

        





    


