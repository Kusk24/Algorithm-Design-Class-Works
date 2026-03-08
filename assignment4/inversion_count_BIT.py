# INVCNT - Inversion Count using Binary Indexed Tree (Fenwick Tree)
# Time Complexity: O(n log n)
# Space Complexity: O(n)

# What is a Binary Indexed Tree (BIT)?
# ====================================
# A BIT (also called Fenwick Tree) is a data structure that:
# - Efficiently updates elements: O(log n)
# - Efficiently computes prefix sums: O(log n)
# 
# We'll use it to count how many elements we've seen that are greater
# than the current element!

class BinaryIndexedTree:
    """
    Binary Indexed Tree (Fenwick Tree) for efficient range queries
    
    This data structure helps us count:
    - How many numbers less than X have we seen so far?
    - How many numbers greater than X have we seen so far?
    """
    
    def __init__(self, size):
        """Initialize BIT with given size"""
        self.size = size
        self.tree = [0] * (size + 1)  # 1-indexed array
    
    def update(self, index, value):
        """
        Add 'value' to position 'index'
        O(log n) time complexity
        
        The magic of BIT: Each position is responsible for a range
        We use bit manipulation to find which positions to update
        """
        while index <= self.size:
            self.tree[index] += value
            index += index & (-index)  # Add last set bit
    
    def query(self, index):
        """
        Get sum from position 1 to 'index' (prefix sum)
        O(log n) time complexity
        
        This tells us: How many elements from 1 to index have we seen?
        """
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)  # Remove last set bit
        return result
    
    def range_query(self, left, right):
        """
        Get sum from position 'left' to 'right'
        Uses: sum(left, right) = sum(1, right) - sum(1, left-1)
        """
        if left > 1:
            return self.query(right) - self.query(left - 1)
        return self.query(right)


def count_inversions_BIT(arr):
    """
    Count inversions using Binary Indexed Tree
    
    Algorithm Steps:
    1. Compress the array values to range [1, n] (coordinate compression)
    2. Process array from left to right
    3. For each element:
       - Count how many LARGER elements we've seen before it (inversions!)
       - Mark this element as "seen" in the BIT
    
    Args:
        arr: input array
    
    Returns:
        Number of inversions
    """
    n = len(arr)
    
    # Step 1: Coordinate Compression
    # Convert array values to ranks (1 to n)
    # Example: [100, 20, 50] becomes [3, 1, 2]
    # This allows us to use a BIT of size n instead of max(arr)
    sorted_arr = sorted(enumerate(arr), key=lambda x: x[1])
    rank = [0] * n
    for i, (original_index, _) in enumerate(sorted_arr):
        rank[original_index] = i + 1  # Ranks are 1-indexed
    
    # Step 2: Create BIT of size n
    bit = BinaryIndexedTree(n)
    
    # Step 3: Count inversions
    inversions = 0
    
    for i in range(n):
        # How many elements GREATER than current have we seen?
        # Total elements seen so far: i
        # Elements <= current: query(rank[i])
        # Elements > current: i - query(rank[i])
        elements_greater = i - bit.query(rank[i])
        inversions += elements_greater
        
        # Mark current element as seen
        bit.update(rank[i], 1)
    
    return inversions


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
    result = count_inversions_BIT(arr)
    print(result)


"""
Example Walkthrough with BIT:
==============================
Input: [3, 1, 2]

Step 1: Coordinate Compression
- Original: [3, 1, 2]
- Sorted by value: [(1, at index 1), (2, at index 2), (3, at index 0)]
- Ranks: [3, 1, 2] (3 is rank 3, 1 is rank 1, 2 is rank 2)

Step 2: Process each element
i=0, element=3 (rank=3):
  - Elements seen so far: 0
  - Elements <= 3 seen: query(3) = 0
  - Elements > 3 seen: 0 - 0 = 0 inversions
  - Mark rank 3 as seen in BIT
  - Total inversions: 0

i=1, element=1 (rank=1):
  - Elements seen so far: 1
  - Elements <= 1 seen: query(1) = 0
  - Elements > 1 seen: 1 - 0 = 1 inversion (the 3!)
  - Mark rank 1 as seen in BIT
  - Total inversions: 1

i=2, element=2 (rank=2):
  - Elements seen so far: 2
  - Elements <= 2 seen: query(2) = 1 (we've seen 1)
  - Elements > 2 seen: 2 - 1 = 1 inversion (the 3!)
  - Mark rank 2 as seen in BIT
  - Total inversions: 2

Final Answer: 2 inversions

Why BIT is useful here:
- Efficiently track which values we've seen: O(log n) per update
- Efficiently count how many values <= X we've seen: O(log n) per query
- Total: O(n log n) for processing all elements
"""
