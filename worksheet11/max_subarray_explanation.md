# Maximum Subarray Problem - Complete Explanation

## Problem
Find the contiguous subarray (within an array of integers) which has the largest sum.

Example: `[-2, 1, -3, 4, -1, 2, 1, -5, 4]`  
Answer: `6` (from subarray `[4, -1, 2, 1]`)

---

## Four Different Approaches

### 1. Brute Force (maxsum_v1.py) - O(n³)
**Strategy:** Try every possible subarray, calculate its sum

```python
# Check all pairs (i, j) where i <= j
for i in range(n):
    for j in range(i, n):
        sum = Sum(array, i, j)  # O(n) to calculate sum
        max_sum = max(max_sum, sum)
```

**Time Complexity:** O(n³)
- Outer loop: n iterations
- Inner loop: n iterations  
- Sum calculation: n operations
- Total: n × n × n = O(n³)

**Pros:** Simple to understand  
**Cons:** Extremely slow for large inputs

---

### 2. Prefix Sum Optimization (maxsum_v2.py) - O(n²)
**Strategy:** Pre-compute cumulative sums to avoid recalculating

```python
# Build prefix sum array first
acc_list[i] = sum of elements from index 0 to i

# Now sum(i, j) can be calculated in O(1):
sum(i, j) = acc_list[j] - acc_list[i-1]

# Then try all pairs
for i in range(n):
    for j in range(i, n):
        current = Sum(i, j)  # Now O(1) instead of O(n)
        max_sum = max(max_sum, current)
```

**Time Complexity:** O(n²)
- Build prefix array: O(n)
- Two nested loops: n × n = O(n²)
- Total: O(n²)

**Pros:** Faster than brute force  
**Cons:** Still too slow for n > 10,000

---

### 3. Kadane's Algorithm (maxsum_v3.py) - O(n) ⭐ FASTEST
**Strategy:** Use dynamic programming - at each position, decide whether to extend current subarray or start fresh

```python
current_sum = array[0]  # Max sum ending at current position
max_sum = array[0]      # Overall maximum

for x in array[1:]:
    # Key decision: extend current subarray OR start new one
    current_sum = max(x, current_sum + x)
    max_sum = max(max_sum, current_sum)
```

**Why it works:**
- If `current_sum + x` is negative, starting fresh from `x` is better
- We only need to remember the best sum ending at current position

**Time Complexity:** O(n) - single pass through array  
**Space Complexity:** O(1) - only two variables

**Pros:** Fastest possible, elegant  
**Cons:** Requires insight to understand the logic

---

### 4. Divide and Conquer (maxSubSum.py) - O(n log n)
**Strategy:** Split array in half, the answer is either:
1. Entirely in the left half
2. Entirely in the right half  
3. **Crosses the midpoint** (spans both halves)

```python
def maxSubSum(i, k):
    # Base case: single element
    if i == k:
        return n[i]
    
    # 1. Split at midpoint
    mid = (i + k) // 2
    
    # 2. Recursively solve both halves
    left_max = maxSubSum(i, mid)
    right_max = maxSubSum(mid + 1, k)
    
    # 3. Find max subarray that CROSSES the midpoint
    # Extend left from mid
    left_sum = -infinity
    current = 0
    for j from mid down to i:
        current += n[j]
        left_sum = max(left_sum, current)
    
    # Extend right from mid+1
    right_sum = -infinity
    current = 0
    for j from mid+1 to k:
        current += n[j]
        right_sum = max(right_sum, current)
    
    crossing_max = left_sum + right_sum
    
    # 4. Return the best of the three
    return max(left_max, right_max, crossing_max)
```

**Why Step 3 is crucial:**
- Without checking crossing subarrays, you'd miss cases like:
  - Array: `[4, -1, 2]` split at index 1
  - Left max: 4, Right max: 2
  - But crossing subarray `[4, -1, 2]` = 5 is the answer!

**Time Complexity Analysis:**
- Split into 2 subproblems of size n/2: **2T(n/2)**
- Finding crossing sum: scan both sides = **O(n)**
- Recurrence: `T(n) = 2T(n/2) + O(n)`
- By Master Theorem: **T(n) = O(n log n)**

**Pros:** Demonstrates divide-and-conquer paradigm well  
**Cons:** Slower than Kadane's O(n), more complex code

---

## Performance Comparison

| Algorithm | Time Complexity | Space | Best For |
|-----------|----------------|-------|----------|
| Brute Force (v1) | O(n³) | O(1) | n < 100 |
| Prefix Sum (v2) | O(n²) | O(n) | n < 1,000 |
| Kadane's (v3) | **O(n)** | O(1) | **Production** |
| Divide & Conquer | O(n log n) | O(log n) | **Learning** |

---

## When to Use Each

- **Production code:** Use Kadane's algorithm (v3) - fastest and simplest
- **Learning recursion:** Use Divide & Conquer - great teaching example
- **Understanding the problem:** Start with v1, then optimize to v2, v3
- **Interviews:** Know all approaches, implement Kadane's

---

## Key Takeaway

The maximum subarray problem demonstrates how algorithm design affects performance:
- **O(n³) → O(n)** is a 1000× speedup for n=1000!
- Different paradigms (DP vs D&C) can solve the same problem
- The simplest solution (Kadane's) is often the best
