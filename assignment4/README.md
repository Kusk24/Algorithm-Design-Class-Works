# INVCNT - Inversion Count Problem

## Problem Summary
Count the number of inversions in an array. An inversion is a pair of indices (i, j) where i < j but A[i] > A[j].

**Example:** Array [3, 1, 2] has 2 inversions:
- (0, 1): 3 > 1
- (0, 2): 3 > 2

## Solutions Provided

### 1. `inversion_count.py` - **RECOMMENDED FOR SUBMISSION**
- **Algorithm:** Modified Merge Sort (Divide and Conquer)
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(n)
- **Best for:** All test cases, especially large inputs (up to 200,000 elements)

#### How it works:
1. **Divide:** Split array into two halves
2. **Conquer:** Recursively count inversions in each half
3. **Combine:** Count inversions during merge
   - **Key Insight:** When merging, if an element from the right half is smaller than an element from the left half, it forms inversions with ALL remaining elements in the left half

### 2. `inversion_count_bruteforce.py` - **FOR LEARNING ONLY**
- **Algorithm:** Nested loops to check all pairs
- **Time Complexity:** O(n²)
- **Best for:** Understanding the problem with small inputs only
- **Warning:** Will timeout on large inputs!

## Running the Solutions

### Using the efficient solution:
```bash
python3 inversion_count.py < test_input.txt
```

### Using the brute force solution (small inputs only):
```bash
python3 inversion_count_bruteforce.py < test_input.txt
```

## Performance Comparison

For n = 200,000 (maximum input size):
- **Brute Force:** ~20 billion operations → **TIMEOUT**
- **Merge Sort:** ~3.6 million operations → **PASS**

**Merge Sort is about 5,500x faster!**

## Test Cases Included

`test_input.txt` contains the example from the problem:
- Test 1: [3, 1, 2] → Expected: 2
- Test 2: [2, 3, 8, 6, 1] → Expected: 5

## Algorithm Connection to Course Material

This solution uses **Divide and Conquer**, similar to:
- **worksheet11/maxSubSum.py** - Maximum subarray problem
- **Merge Sort** - Classic divide and conquer sorting algorithm

The pattern is the same:
1. Divide problem into smaller subproblems
2. Solve subproblems recursively
3. Combine solutions (this is where we count inversions!)

## Tips for Understanding

1. Start by understanding what an inversion is
2. Try the brute force approach on small examples by hand
3. Understand why merge sort can count inversions efficiently
4. The magic happens during the merge step!

## Submission
Use **`inversion_count.py`** for your submission!
