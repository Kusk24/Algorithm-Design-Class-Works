"""
================================================================================
ASSIGNMENT 4 - QUESTIONS 3 & 4
================================================================================
"""

# ============================================================================
# QUESTION 3: Educational DP Contest AtCoder A – Frog 1
# ============================================================================
"""
PROBLEM STATEMENT:

There are N stones, numbered 1, 2, ..., N.

For each i (1 ≤ i ≤ N), the height of Stone i is h_i.

There is a frog who is initially on Stone 1.
He will repeat the following action some number of times to reach Stone N:

    • If the frog is currently on Stone i, jump to Stone i + 1 or Stone i + 2.

Here, a cost of |h_i − h_j| is incurred, where j is the stone to land on.

Find the minimum possible total cost incurred before the frog reaches Stone N.

CONSTRAINTS:
- All values in input are integers.
- 2 ≤ N ≤ 10^5
- 1 ≤ h_i ≤ 10^4

INPUT SPECIFICATION:
- The first line of input will contain an integer N.
- The second line of input will contain N spaced integers, h_i, the height of stone i.

OUTPUT SPECIFICATION:
Output a single integer, the minimum possible total cost incurred.

EXAMPLE:
Input:
4
10 30 40 20

Output:
30

Explanation:
The frog can jump: 1 → 2 → 4
- Cost from stone 1 to 2: |10 - 30| = 20
- Cost from stone 2 to 4: |30 - 20| = 10
- Total cost: 30
"""


# ============================================================================
# QUESTION 4: INVCNT – Inversion Count
# ============================================================================
"""
PROBLEM STATEMENT:

Tags: #number-theory #sorting

Let A[0…n−1] be an array of n distinct positive integers.
If i < j and A[i] > A[j] then the pair (i,j) is called an inversion of A.

Given n and an array A, your task is to find the number of inversions of A.

INPUT:
- The first line contains t, the number of testcases followed by a blank space.
- Each of the t tests starts with a number n (n ≤ 200000).
- Then n + 1 lines follow.
- In the i-th line a number A[i−1] is given (A[i−1] ≤ 10^7).
- The (n + 1)-th line is a blank space.

OUTPUT:
For every test output one line giving the number of inversions of A.

EXAMPLE:
Input:
2

3
3
1
2

5
2
3
8
6
1

Output:
2
5

EXPLANATION:
Test 1: Array = [3, 1, 2]
  Inversions: (0,1) because 3>1, (0,2) because 3>2
  Total: 2 inversions

Test 2: Array = [2, 3, 8, 6, 1]
  Inversions: 
    - (0,4): 2 > 1
    - (1,4): 3 > 1
    - (2,3): 8 > 6
    - (2,4): 8 > 1
    - (3,4): 6 > 1
  Total: 5 inversions
"""
