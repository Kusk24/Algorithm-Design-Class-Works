# INVCNT - Inversion Count (Brute Force Approach)
# Time Complexity: O(n²) - SLOW for large inputs
# Use this only for understanding, not for submission!

def count_inversions_bruteforce(arr):
    """
    Simple approach: Check all pairs (i, j) where i < j
    Count how many have arr[i] > arr[j]
    
    This is easy to understand but SLOW for large arrays!
    Time Complexity: O(n²)
    """
    count = 0
    n = len(arr)
    
    # Check every possible pair
    for i in range(n):
        for j in range(i + 1, n):
            # If element at position i is greater than element at position j
            # and i comes before j, we have an inversion
            if arr[i] > arr[j]:
                count += 1
    
    return count


# Main program
t = int(input())  # Number of test cases

for _ in range(t):
    input()  # Read blank line
    n = int(input())  # Size of array
    
    # Read the array
    arr = []
    for i in range(n):
        arr.append(int(input()))
    
    # Count and print inversions
    result = count_inversions_bruteforce(arr)
    print(result)


"""
Why is this approach slow?
===========================
For n = 200,000 (maximum in problem):
- Brute Force: ~20,000,000,000 operations (20 billion!) - Will timeout!
- Merge Sort: ~3,600,000 operations (3.6 million) - Fast!

That's why we use the modified merge sort approach!
"""
