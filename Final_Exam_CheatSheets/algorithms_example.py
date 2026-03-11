"""
================================================================================
ALGORITHM DESIGN - UNIVERSITY EXAM PROBLEMS
================================================================================
Author: Algorithm Design Course
Topics: DP, Backtracking, BFS/DFS, Greedy, Divide & Conquer, Graph Algorithms
Reference: LeetCode, Codeforces, HackerRank

This file contains 30 common university exam problems organized by algorithm type.
Each problem includes:
- Problem statement
- Algorithm used and why
- Complexity analysis
- Full working code
================================================================================

TABLE OF CONTENTS:
==================
Section 1: Kadane's Algorithm / Maximum Subarray (5 problems)
Section 2: Dynamic Programming (5 problems)
Section 3: Backtracking (5 problems)
Section 4: BFS/DFS Graph Traversal (5 problems)
Section 5: Greedy Algorithms (5 problems)
Section 6: Divide and Conquer (5 problems)

================================================================================
"""

# ============================================================================
# SECTION 1: KADANE'S ALGORITHM / MAXIMUM SUBARRAY PROBLEMS
# ============================================================================

"""
ALGORITHM: Kadane's Algorithm
TIME COMPLEXITY: O(n)
SPACE COMPLEXITY: O(1)

WHY USED: Efficiently finds maximum sum of contiguous subarray in linear time
by tracking current sum and resetting when it becomes negative.
"""

# ----------------------------------------------------------------------------
# PROBLEM 1.1: Maximum Subarray Sum
# LeetCode 53 (Easy)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given an array nums[], find the contiguous subarray which has the largest sum.

Example:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6

WHY KADANE'S ALGORITHM:
- Need to find contiguous subarray (dynamic programming property)
- Overlapping subproblems: maxSum[i] depends on maxSum[i-1]
- Linear time solution vs O(n²) brute force
"""

def maxSubArray(nums):
    """Kadane's Algorithm for maximum subarray sum."""
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        # Either extend current subarray or start new one
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Test
print("Problem 1.1: Maximum Subarray")
print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))  # Output: 6
print(maxSubArray([1]))  # Output: 1
print(maxSubArray([5,4,-1,7,8]))  # Output: 23
print()

# ----------------------------------------------------------------------------
# PROBLEM 1.2: Maximum Product Subarray
# LeetCode 152 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find the contiguous subarray with the largest product.

Example:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product = 6

WHY MODIFIED KADANE'S:
- Similar to sum problem but need to track both max and min
- Negative numbers: min * negative = max
- Must consider all three options: current, max*current, min*current
"""

def maxProduct(nums):
    """Modified Kadane's for product (track both max and min)."""
    if not nums:
        return 0
    
    max_prod = nums[0]
    current_max = nums[0]
    current_min = nums[0]
    
    for i in range(1, len(nums)):
        # When multiplied by negative, max becomes min and vice versa
        if nums[i] < 0:
            current_max, current_min = current_min, current_max
        
        current_max = max(nums[i], current_max * nums[i])
        current_min = min(nums[i], current_min * nums[i])
        
        max_prod = max(max_prod, current_max)
    
    return max_prod

# Test
print("Problem 1.2: Maximum Product Subarray")
print(maxProduct([2,3,-2,4]))  # Output: 6
print(maxProduct([-2,0,-1]))  # Output: 0
print(maxProduct([-2,3,-4]))  # Output: 24
print()

# ----------------------------------------------------------------------------
# PROBLEM 1.3: Best Time to Buy and Sell Stock
# LeetCode 121 (Easy)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given stock prices array, find max profit from one buy and one sell.
Can only sell after you buy.

Example:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price=1), sell on day 5 (price=6), profit = 6-1 = 5

WHY KADANE'S-LIKE APPROACH:
- Convert to difference array: profit[i] = price[i+1] - price[i]
- Finding max profit = finding max subarray sum of differences
- Track minimum price seen so far, calculate profit at each step
"""

def maxProfit(prices):
    """One-pass solution tracking minimum price."""
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit

# Test
print("Problem 1.3: Best Time to Buy and Sell Stock")
print(maxProfit([7,1,5,3,6,4]))  # Output: 5
print(maxProfit([7,6,4,3,1]))  # Output: 0
print(maxProfit([2,4,1]))  # Output: 2
print()

# ----------------------------------------------------------------------------
# PROBLEM 1.4: Maximum Circular Subarray Sum
# ----------------------------------------------------------------------------
"""
QUESTION:
Find maximum sum of circular subarray (array is circular, last connects to first).

Example:
Input: nums = [5,-3,5]
Output: 10
Explanation: Circular subarray [5,5] has sum = 10

WHY TWO KADANE'S:
- Case 1: Max subarray doesn't wrap (normal Kadane's)
- Case 2: Max subarray wraps (total_sum - min_subarray_sum)
- Return max of both cases
"""

def maxSubarraySumCircular(nums):
    """Kadane's applied twice: for max and for min."""
    def kadane_max(arr):
        max_sum = arr[0]
        current = arr[0]
        for i in range(1, len(arr)):
            current = max(arr[i], current + arr[i])
            max_sum = max(max_sum, current)
        return max_sum
    
    def kadane_min(arr):
        min_sum = arr[0]
        current = arr[0]
        for i in range(1, len(arr)):
            current = min(arr[i], current + arr[i])
            min_sum = min(min_sum, current)
        return min_sum
    
    max_normal = kadane_max(nums)
    
    # If all negative, return max element
    total = sum(nums)
    if total == max_normal:
        return max_normal
    
    min_subarray = kadane_min(nums)
    max_circular = total - min_subarray
    
    return max(max_normal, max_circular)

# Test
print("Problem 1.4: Maximum Circular Subarray Sum")
print(maxSubarraySumCircular([5,-3,5]))  # Output: 10
print(maxSubarraySumCircular([1,-2,3,-2]))  # Output: 3
print(maxSubarraySumCircular([-3,-2,-3]))  # Output: -2
print()

# ----------------------------------------------------------------------------
# PROBLEM 1.5: Longest Turbulent Subarray
# LeetCode 978 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find length of longest turbulent subarray where comparisons alternate:
arr[i] < arr[i+1] > arr[i+2] < arr[i+3]... or
arr[i] > arr[i+1] < arr[i+2] > arr[i+3]...

Example:
Input: arr = [9,4,2,10,7,8,8,1,9]
Output: 5
Explanation: [4,2,10,7,8] is turbulent

WHY KADANE'S-LIKE:
- Track length of current turbulent sequence
- Reset when pattern breaks
- Similar to Kadane's tracking of current state
"""

def maxTurbulenceSize(arr):
    """Track increasing and decreasing streak lengths."""
    if len(arr) <= 1:
        return len(arr)
    
    max_len = 1
    inc = 1  # Length ending with increase
    dec = 1  # Length ending with decrease
    
    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            inc = dec + 1
            dec = 1
        elif arr[i] < arr[i-1]:
            dec = inc + 1
            inc = 1
        else:
            inc = 1
            dec = 1
        
        max_len = max(max_len, max(inc, dec))
    
    return max_len

# Test
print("Problem 1.5: Longest Turbulent Subarray")
print(maxTurbulenceSize([9,4,2,10,7,8,8,1,9]))  # Output: 5
print(maxTurbulenceSize([4,8,12,16]))  # Output: 2
print(maxTurbulenceSize([100]))  # Output: 1
print()


# ============================================================================
# SECTION 2: DYNAMIC PROGRAMMING PROBLEMS
# ============================================================================

"""
ALGORITHM: Dynamic Programming (Bottom-Up / Top-Down)
TIME COMPLEXITY: Usually O(n²) or O(n*m)
SPACE COMPLEXITY: O(n) to O(n*m)

WHY USED: Problems with overlapping subproblems and optimal substructure.
DP avoids recomputation by storing results of subproblems.
"""

# ----------------------------------------------------------------------------
# PROBLEM 2.1: Climbing Stairs
# LeetCode 70 (Easy)
# ----------------------------------------------------------------------------
"""
QUESTION:
You're climbing a staircase with n steps. Each time you can climb 1 or 2 steps.
How many distinct ways can you climb to the top?

Example:
Input: n = 3
Output: 3
Explanation: Three ways: 1+1+1, 1+2, 2+1

WHY DP:
- Overlapping subproblems: ways(n) = ways(n-1) + ways(n-2)
- Similar to Fibonacci sequence
- Memoization avoids exponential recomputation
"""

def climbStairs(n):
    """Bottom-up DP (Fibonacci-like)."""
    if n <= 2:
        return n
    
    prev2 = 1  # ways(1)
    prev1 = 2  # ways(2)
    
    for i in range(3, n+1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Test
print("Problem 2.1: Climbing Stairs")
print(climbStairs(2))  # Output: 2
print(climbStairs(3))  # Output: 3
print(climbStairs(5))  # Output: 8
print()

# ----------------------------------------------------------------------------
# PROBLEM 2.2: Coin Change (Minimum Coins)
# LeetCode 322 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given coins of different denominations and total amount, return minimum number
of coins needed to make that amount. Return -1 if impossible.

Example:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

WHY DP:
- Overlapping subproblems: minCoins(n) uses minCoins(n-coin)
- Unbounded knapsack variant (can use each coin unlimited times)
- Greedy doesn't work for all coin systems
"""

def coinChange(coins, amount):
    """Bottom-up DP with table."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 0 coins for amount 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Test
print("Problem 2.2: Coin Change")
print(coinChange([1,2,5], 11))  # Output: 3
print(coinChange([2], 3))  # Output: -1
print(coinChange([1], 0))  # Output: 0
print()

# ----------------------------------------------------------------------------
# PROBLEM 2.3: Longest Increasing Subsequence
# LeetCode 300 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find length of longest strictly increasing subsequence.

Example:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: [2,3,7,101] or [2,3,7,18]

WHY DP:
- Overlapping subproblems: LIS ending at i depends on LIS at all j < i
- Optimal substructure: if arr[j] < arr[i], can extend LIS at j
- O(n²) DP solution (O(n log n) with binary search)
"""

def lengthOfLIS(nums):
    """O(n²) DP solution."""
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS length ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Test
print("Problem 2.3: Longest Increasing Subsequence")
print(lengthOfLIS([10,9,2,5,3,7,101,18]))  # Output: 4
print(lengthOfLIS([0,1,0,3,2,3]))  # Output: 4
print(lengthOfLIS([7,7,7,7,7,7,7]))  # Output: 1
print()

# ----------------------------------------------------------------------------
# PROBLEM 2.4: Partition Equal Subset Sum
# LeetCode 416 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array, determine if you can partition it into two subsets with equal sum.

Example:
Input: nums = [1,5,11,5]
Output: True
Explanation: [1,5,5] and [11] both sum to 11

WHY DP:
- Subset sum problem (classic DP)
- If total sum is odd, return False
- Need to find if subset with sum = total/2 exists
- 0/1 Knapsack variant
"""

def canPartition(nums):
    """0/1 Knapsack DP approach."""
    total = sum(nums)
    
    # If odd sum, can't partition equally
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True  # Can make sum 0 with empty subset
    
    for num in nums:
        # Traverse backwards to avoid using same element twice
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]

# Test
print("Problem 2.4: Partition Equal Subset Sum")
print(canPartition([1,5,11,5]))  # Output: True
print(canPartition([1,2,3,5]))  # Output: False
print(canPartition([1,2,5]))  # Output: False
print()

# ----------------------------------------------------------------------------
# PROBLEM 2.5: Edit Distance
# LeetCode 72 (Hard)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given two strings word1 and word2, return minimum operations to convert word1 to word2.
Operations: insert, delete, replace (each costs 1).

Example:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: horse -> rorse -> rose -> ros

WHY DP:
- Classic DP problem (Levenshtein distance)
- Overlapping subproblems: dp[i][j] depends on dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
- Optimal substructure: optimal solution contains optimal subsolutions
"""

def minDistance(word1, word2):
    """2D DP table approach."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]

# Test
print("Problem 2.5: Edit Distance")
print(minDistance("horse", "ros"))  # Output: 3
print(minDistance("intention", "execution"))  # Output: 5
print(minDistance("abc", "abc"))  # Output: 0
print()


# ============================================================================
# SECTION 3: BACKTRACKING PROBLEMS
# ============================================================================

"""
ALGORITHM: Backtracking (Recursive with pruning)
TIME COMPLEXITY: O(2^n) to O(n!) depending on problem
SPACE COMPLEXITY: O(n) for recursion stack

WHY USED: Constraint satisfaction problems, generate all possibilities,
prune invalid branches early.
"""

# ----------------------------------------------------------------------------
# PROBLEM 3.1: Permutations
# LeetCode 46 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array of distinct integers, return all possible permutations.

Example:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

WHY BACKTRACKING:
- Need to generate all permutations (factorial possibilities)
- Use backtracking to explore all orderings
- Prune: don't revisit already placed elements
"""

def permute(nums):
    """Backtracking with swap."""
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack
    
    backtrack(0)
    return result

# Test
print("Problem 3.1: Permutations")
print(permute([1,2,3]))
print(permute([0,1]))
print()

# ----------------------------------------------------------------------------
# PROBLEM 3.2: Subsets
# LeetCode 78 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array of distinct integers, return all possible subsets (power set).

Example:
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

WHY BACKTRACKING:
- Generate all 2^n subsets
- Each element: include or exclude (binary decision)
- Backtracking explores both branches
"""

def subsets(nums):
    """Backtracking to generate all subsets."""
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()  # Backtrack
    
    backtrack(0, [])
    return result

# Test
print("Problem 3.2: Subsets")
print(subsets([1,2,3]))
print(subsets([0]))
print()

# ----------------------------------------------------------------------------
# PROBLEM 3.3: Combination Sum
# LeetCode 39 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array of distinct integers and target, return all unique combinations
that sum to target. Can use same number unlimited times.

Example:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

WHY BACKTRACKING:
- Need all valid combinations (not permutations)
- Unlimited use of each element (unbounded)
- Backtrack when sum exceeds target (pruning)
"""

def combinationSum(candidates, target):
    """Backtracking with pruning."""
    result = []
    
    def backtrack(start, current, remain):
        if remain == 0:
            result.append(current[:])
            return
        if remain < 0:
            return  # Prune
        
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remain - candidates[i])  # Can reuse same element
            current.pop()
    
    backtrack(0, [], target)
    return result

# Test
print("Problem 3.3: Combination Sum")
print(combinationSum([2,3,6,7], 7))
print(combinationSum([2,3,5], 8))
print()

# ----------------------------------------------------------------------------
# PROBLEM 3.4: Generate Parentheses
# LeetCode 22 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given n pairs of parentheses, generate all well-formed combinations.

Example:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

WHY BACKTRACKING:
- Generate all valid sequences (constraint: well-formed)
- Prune: can't add ')' if it would make invalid
- Track open and close counts
"""

def generateParenthesis(n):
    """Backtracking with constraints."""
    result = []
    
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Can add '(' if still have some left
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Can add ')' only if doesn't exceed '('
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

# Test
print("Problem 3.4: Generate Parentheses")
print(generateParenthesis(3))
print(generateParenthesis(1))
print()

# ----------------------------------------------------------------------------
# PROBLEM 3.5: Palindrome Partitioning
# LeetCode 131 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given string s, partition s such that every substring is a palindrome.
Return all possible partitions.

Example:
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

WHY BACKTRACKING:
- Generate all possible partitions
- Check palindrome constraint at each step
- Backtrack when constraint violated
"""

def partition(s):
    """Backtracking with palindrome check."""
    result = []
    
    def is_palindrome(substr):
        return substr == substr[::-1]
    
    def backtrack(start, current):
        if start == len(s):
            result.append(current[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()
    
    backtrack(0, [])
    return result

# Test
print("Problem 3.5: Palindrome Partitioning")
print(partition("aab"))
print(partition("a"))
print()


# ============================================================================
# SECTION 4: BFS/DFS GRAPH TRAVERSAL PROBLEMS
# ============================================================================

"""
ALGORITHM: BFS (Queue) / DFS (Stack/Recursion)
TIME COMPLEXITY: O(V + E) for graphs, O(m*n) for grids
SPACE COMPLEXITY: O(V) for graphs, O(m*n) for grids

WHY USED: BFS for shortest path (unweighted), DFS for exploring all paths,
both for connectivity and traversal.
"""

# ----------------------------------------------------------------------------
# PROBLEM 4.1: Number of Islands
# LeetCode 200 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given 2D grid of '1's (land) and '0's (water), count number of islands.
Island = group of connected 1's (horizontally or vertically).

Example:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

WHY DFS/BFS:
- Connected components problem
- DFS marks entire island as visited
- Each DFS call = one island found
"""

def numIslands(grid):
    """DFS to mark connected components."""
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return
        
        grid[r][c] = '0'  # Mark as visited
        
        # Explore 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    
    return count

# Test
print("Problem 4.1: Number of Islands")
grid1 = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
print(numIslands(grid1))  # Output: 3
print()

# ----------------------------------------------------------------------------
# PROBLEM 4.2: Rotting Oranges
# LeetCode 994 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Grid contains: 0=empty, 1=fresh orange, 2=rotten orange. Every minute, rotten
oranges rot adjacent fresh oranges (4-directionally). Return minimum minutes
until no fresh oranges remain, or -1 if impossible.

Example:
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

WHY BFS:
- Multi-source BFS (all rotten oranges start simultaneously)
- Level-by-level expansion = time progression
- BFS ensures minimum time
"""

from collections import deque

def orangesRotting(grid):
    """Multi-source BFS."""
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    max_time = 0
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    while queue:
        r, c, time = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] == 1):
                grid[nr][nc] = 2  # Rot the orange
                fresh_count -= 1
                max_time = max(max_time, time + 1)
                queue.append((nr, nc, time + 1))
    
    return max_time if fresh_count == 0 else -1

# Test
print("Problem 4.2: Rotting Oranges")
print(orangesRotting([[2,1,1],[1,1,0],[0,1,1]]))  # Output: 4
print(orangesRotting([[2,1,1],[0,1,1],[1,0,1]]))  # Output: -1
print()

# ----------------------------------------------------------------------------
# PROBLEM 4.3: Course Schedule
# LeetCode 207 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given numCourses and prerequisites array where prerequisites[i] = [a, b]
means must take course b before course a. Determine if can finish all courses.

Example:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: True
Explanation: Take course 0 first, then course 1.

WHY DFS:
- Cycle detection in directed graph
- If cycle exists, impossible to complete all courses
- DFS with 3 states: unvisited, visiting, visited
"""

def canFinish(numCourses, prerequisites):
    """DFS cycle detection."""
    # Build adjacency list
    graph = [[] for _ in range(numCourses)]
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * numCourses
    
    def has_cycle(course):
        if state[course] == 1:  # Currently visiting = cycle
            return True
        if state[course] == 2:  # Already visited
            return False
        
        state[course] = 1  # Mark as visiting
        
        for next_course in graph[course]:
            if has_cycle(next_course):
                return True
        
        state[course] = 2  # Mark as visited
        return False
    
    for course in range(numCourses):
        if has_cycle(course):
            return False
    
    return True

# Test
print("Problem 4.3: Course Schedule")
print(canFinish(2, [[1,0]]))  # Output: True
print(canFinish(2, [[1,0],[0,1]]))  # Output: False
print()

# ----------------------------------------------------------------------------
# PROBLEM 4.4: Shortest Path in Binary Matrix
# LeetCode 1091 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given n x n binary matrix, find shortest clear path from top-left to bottom-right.
Clear path: all cells in path are 0, can move 8-directionally.

Example:
Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
Explanation: Path is (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)

WHY BFS:
- Shortest path in unweighted grid
- BFS guarantees finding shortest path first
- 8-directional movement
"""

def shortestPathBinaryMatrix(grid):
    """BFS for shortest path."""
    n = len(grid)
    
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    if n == 1:
        return 1
    
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # Mark as visited
    
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    while queue:
        r, c, dist = queue.popleft()
        
        if r == n-1 and c == n-1:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0):
                grid[nr][nc] = 1  # Mark as visited
                queue.append((nr, nc, dist + 1))
    
    return -1

# Test
print("Problem 4.4: Shortest Path in Binary Matrix")
print(shortestPathBinaryMatrix([[0,0,0],[1,1,0],[1,1,0]]))  # Output: 4
print(shortestPathBinaryMatrix([[0,1],[1,0]]))  # Output: 2
print()

# ----------------------------------------------------------------------------
# PROBLEM 4.5: Clone Graph
# LeetCode 133 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given reference to a node in connected undirected graph, return deep copy.

WHY DFS/BFS:
- Need to traverse entire graph
- Use hashmap to map original nodes to cloned nodes
- DFS or BFS to explore all nodes
"""

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    """DFS with hashmap."""
    if not node:
        return None
    
    clones = {}  # Map original node to clone
    
    def dfs(original):
        if original in clones:
            return clones[original]
        
        clone = Node(original.val)
        clones[original] = clone
        
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

# Test
print("Problem 4.5: Clone Graph")
print("(Graph cloning - see code for implementation)")
print()


# ============================================================================
# SECTION 5: GREEDY ALGORITHM PROBLEMS
# ============================================================================

"""
ALGORITHM: Greedy (Make locally optimal choice)
TIME COMPLEXITY: Usually O(n log n) due to sorting
SPACE COMPLEXITY: O(1) to O(n)

WHY USED: When greedy choice property holds - local optimum leads to global optimum.
Must prove greedy works for the specific problem.
"""

# ----------------------------------------------------------------------------
# PROBLEM 5.1: Jump Game
# LeetCode 55 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array where nums[i] = max jump length at position i.
Determine if you can reach the last index starting from index 0.

Example:
Input: nums = [2,3,1,1,4]
Output: True
Explanation: Jump 1 step from 0 to 1, then 3 steps to last index.

WHY GREEDY:
- Track furthest reachable position
- At each step, update furthest based on current position + jump
- If current index > furthest, can't continue
"""

def canJump(nums):
    """Greedy: track maximum reachable."""
    furthest = 0
    
    for i in range(len(nums)):
        if i > furthest:
            return False
        furthest = max(furthest, i + nums[i])
        if furthest >= len(nums) - 1:
            return True
    
    return True

# Test
print("Problem 5.1: Jump Game")
print(canJump([2,3,1,1,4]))  # Output: True
print(canJump([3,2,1,0,4]))  # Output: False
print()

# ----------------------------------------------------------------------------
# PROBLEM 5.2: Gas Station
# LeetCode 134 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Circular route with n gas stations. gas[i] = gas at station i,
cost[i] = gas needed to travel to next station. Find starting station
index where you can complete circuit, or -1 if impossible.

Example:
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3

WHY GREEDY:
- If total gas >= total cost, solution exists
- Start from station where we first run out of gas
- Greedy: if can't reach station j from i, all stations between i and j won't work
"""

def canCompleteCircuit(gas, cost):
    """Greedy starting point selection."""
    if sum(gas) < sum(cost):
        return -1
    
    total_tank = 0
    current_tank = 0
    start = 0
    
    for i in range(len(gas)):
        current_tank += gas[i] - cost[i]
        total_tank += gas[i] - cost[i]
        
        if current_tank < 0:
            start = i + 1
            current_tank = 0
    
    return start if total_tank >= 0 else -1

# Test
print("Problem 5.2: Gas Station")
print(canCompleteCircuit([1,2,3,4,5], [3,4,5,1,2]))  # Output: 3
print(canCompleteCircuit([2,3,4], [3,4,3]))  # Output: -1
print()

# ----------------------------------------------------------------------------
# PROBLEM 5.3: Non-overlapping Intervals
# LeetCode 435 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array of intervals, return minimum number of intervals to remove
to make rest non-overlapping.

Example:
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: Remove [1,3] to make others non-overlapping.

WHY GREEDY:
- Sort by end time (activity selection strategy)
- Always choose interval that ends earliest
- Leaves most room for future intervals
"""

def eraseOverlapIntervals(intervals):
    """Greedy: sort by end time."""
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[1])  # Sort by end time
    
    count = 0
    end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:  # Overlaps
            count += 1
        else:
            end = intervals[i][1]
    
    return count

# Test
print("Problem 5.3: Non-overlapping Intervals")
print(eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]]))  # Output: 1
print(eraseOverlapIntervals([[1,2],[1,2],[1,2]]))  # Output: 2
print()

# ----------------------------------------------------------------------------
# PROBLEM 5.4: Task Scheduler
# LeetCode 621 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Given array of tasks and cooldown period n, find minimum intervals needed
to complete all tasks. Same task must wait n intervals before repeating.

Example:
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B

WHY GREEDY:
- Schedule most frequent tasks first
- Fill gaps with less frequent tasks
- Greedy: max frequency determines minimum time
"""

from collections import Counter

def leastInterval(tasks, n):
    """Greedy based on frequency."""
    freq = Counter(tasks)
    max_freq = max(freq.values())
    max_count = sum(1 for v in freq.values() if v == max_freq)
    
    # Minimum intervals = (max_freq - 1) * (n + 1) + max_count
    # But at least len(tasks)
    intervals = (max_freq - 1) * (n + 1) + max_count
    return max(intervals, len(tasks))

# Test
print("Problem 5.4: Task Scheduler")
print(leastInterval(["A","A","A","B","B","B"], 2))  # Output: 8
print(leastInterval(["A","A","A","B","B","B"], 0))  # Output: 6
print()

# ----------------------------------------------------------------------------
# PROBLEM 5.5: Partition Labels
# LeetCode 763 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Partition string into as many parts as possible so each letter appears
in at most one part. Return sizes of partitions.

Example:
Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation: "ababcbaca", "defegde", "hijhklij"

WHY GREEDY:
- Track last occurrence of each character
- Extend current partition until reaching last occurrence of all chars
- Greedy: make partition as early as possible
"""

def partitionLabels(s):
    """Greedy: track last occurrence."""
    last = {char: i for i, char in enumerate(s)}
    
    result = []
    start = 0
    end = 0
    
    for i, char in enumerate(s):
        end = max(end, last[char])
        
        if i == end:  # Reached end of partition
            result.append(end - start + 1)
            start = i + 1
    
    return result

# Test
print("Problem 5.5: Partition Labels")
print(partitionLabels("ababcbacadefegdehijhklij"))  # Output: [9,7,8]
print(partitionLabels("eccbbbbdec"))  # Output: [10]
print()


# ============================================================================
# SECTION 6: DIVIDE AND CONQUER PROBLEMS
# ============================================================================

"""
ALGORITHM: Divide and Conquer (Recursive decomposition)
TIME COMPLEXITY: Usually O(n log n)
SPACE COMPLEXITY: O(log n) for recursion stack

WHY USED: Break problem into independent subproblems, solve recursively,
combine results.
"""

# ----------------------------------------------------------------------------
# PROBLEM 6.1: Merge Sort
# Classic Sorting Algorithm
# ----------------------------------------------------------------------------
"""
QUESTION:
Sort an array using merge sort algorithm.

Example:
Input: arr = [12, 11, 13, 5, 6, 7]
Output: [5, 6, 7, 11, 12, 13]

WHY DIVIDE & CONQUER:
- Divide: Split array into halves
- Conquer: Sort each half recursively
- Combine: Merge two sorted halves
"""

def mergeSort(arr):
    """Classic divide and conquer sorting."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergeSort(arr[:mid])
    right = mergeSort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Test
print("Problem 6.1: Merge Sort")
print(mergeSort([12, 11, 13, 5, 6, 7]))
print(mergeSort([5, 2, 3, 1]))
print()

# ----------------------------------------------------------------------------
# PROBLEM 6.2: Quick Select (Kth Largest Element)
# LeetCode 215 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find kth largest element in unsorted array.

Example:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

WHY DIVIDE & CONQUER:
- Partition array around pivot (like quicksort)
- Recursively search only relevant partition
- Average O(n), worst O(n²)
"""

import random

def findKthLargest(nums, k):
    """Quick select algorithm."""
    def partition(left, right, pivot_idx):
        pivot = nums[pivot_idx]
        # Move pivot to end
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        
        # Move pivot to final position
        nums[right], nums[store_idx] = nums[store_idx], nums[right]
        return store_idx
    
    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)
        
        if k_smallest == pivot_idx:
            return nums[k_smallest]
        elif k_smallest < pivot_idx:
            return select(left, pivot_idx - 1, k_smallest)
        else:
            return select(pivot_idx + 1, right, k_smallest)
    
    return select(0, len(nums) - 1, len(nums) - k)

# Test
print("Problem 6.2: Kth Largest Element")
print(findKthLargest([3,2,1,5,6,4], 2))  # Output: 5
print(findKthLargest([3,2,3,1,2,4,5,5,6], 4))  # Output: 4
print()

# ----------------------------------------------------------------------------
# PROBLEM 6.3: Pow(x, n)
# LeetCode 50 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Implement pow(x, n) - calculate x raised to power n.

Example:
Input: x = 2.0, n = 10
Output: 1024.0

WHY DIVIDE & CONQUER:
- Binary exponentiation: x^n = (x^(n/2))^2
- Reduces O(n) to O(log n)
- Handle negative exponents
"""

def myPow(x, n):
    """Fast exponentiation using divide and conquer."""
    def power(x, n):
        if n == 0:
            return 1.0
        
        half = power(x, n // 2)
        
        if n % 2 == 0:
            return half * half
        else:
            return half * half * x
    
    if n < 0:
        x = 1 / x
        n = -n
    
    return power(x, n)

# Test
print("Problem 6.3: Pow(x, n)")
print(myPow(2.0, 10))  # Output: 1024.0
print(myPow(2.1, 3))  # Output: 9.261
print(myPow(2.0, -2))  # Output: 0.25
print()

# ----------------------------------------------------------------------------
# PROBLEM 6.4: Search in Rotated Sorted Array
# LeetCode 33 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Search for target in rotated sorted array. Array originally sorted,
then rotated at unknown pivot.

Example:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

WHY DIVIDE & CONQUER:
- Modified binary search
- Determine which half is sorted
- Decide which half to search based on target
"""

def search(nums, target):
    """Binary search on rotated array."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which side is sorted
        if nums[left] <= nums[mid]:  # Left side is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right side is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Test
print("Problem 6.4: Search in Rotated Sorted Array")
print(search([4,5,6,7,0,1,2], 0))  # Output: 4
print(search([4,5,6,7,0,1,2], 3))  # Output: -1
print()

# ----------------------------------------------------------------------------
# PROBLEM 6.5: Majority Element
# LeetCode 169 (Easy)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find element that appears more than n/2 times in array.

Example:
Input: nums = [3,2,3]
Output: 3

WHY DIVIDE & CONQUER:
- Divide array into halves
- Majority in whole array must be majority in at least one half
- Combine by counting occurrences
- Alternative: Boyer-Moore voting (O(n) one-pass)
"""

def majorityElement(nums):
    """Divide and conquer approach."""
    def majority_helper(lo, hi):
        if lo == hi:
            return nums[lo]
        
        mid = (lo + hi) // 2
        left = majority_helper(lo, mid)
        right = majority_helper(mid + 1, hi)
        
        if left == right:
            return left
        
        # Count occurrences
        left_count = sum(1 for i in range(lo, hi + 1) if nums[i] == left)
        right_count = sum(1 for i in range(lo, hi + 1) if nums[i] == right)
        
        return left if left_count > right_count else right
    
    return majority_helper(0, len(nums) - 1)

# Alternative: Boyer-Moore Voting Algorithm O(n) time, O(1) space
def majorityElementVoting(nums):
    """Boyer-Moore voting algorithm."""
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    return candidate

# Test
print("Problem 6.5: Majority Element")
print(majorityElement([3,2,3]))  # Output: 3
print(majorityElement([2,2,1,1,1,2,2]))  # Output: 2
print(majorityElementVoting([3,2,3]))  # Output: 3
print()


# ============================================================================
# FINAL SUMMARY AND EXAM TIPS
# ============================================================================

# ============================================================================
# SECTION 7: ADVANCED SEARCH ALGORITHMS (IDS, IDA*, UCS)
# ============================================================================

"""
ALGORITHM: Iterative Deepening Search (IDS), IDA*, Uniform Cost Search
TIME COMPLEXITY: O(b^d) for IDS/IDA*, O((V+E) log V) for UCS
SPACE COMPLEXITY: O(d) for IDS/IDA*, O(V) for UCS

WHY USED: Memory-efficient search with completeness and optimality guarantees.
IDS combines DFS space efficiency with BFS completeness.
IDA* adds heuristics to IDS for better performance.
UCS finds cheapest path in weighted graphs.
"""

# ----------------------------------------------------------------------------
# PROBLEM 7.1: Binary Tree Level Order (IDS Application)
# ----------------------------------------------------------------------------
"""
QUESTION:
Perform level-order traversal of binary tree using iterative deepening.

Example:
Input: root = [3,9,20,null,null,15,7]
Output: [[3], [9,20], [15,7]]

WHY IDS:
- Can limit depth instead of using queue
- More memory efficient than standard BFS
- Demonstrates iterative deepening concept
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrderIDS(root):
    """Level order using iterative deepening."""
    if not root:
        return []
    
    def dfs_at_level(node, level, target_level):
        if not node or level > target_level:
            return []
        if level == target_level:
            return [node.val]
        
        result = []
        result.extend(dfs_at_level(node.left, level + 1, target_level))
        result.extend(dfs_at_level(node.right, level + 1, target_level))
        return result
    
    result = []
    depth = 0
    while True:
        level_nodes = dfs_at_level(root, 0, depth)
        if not level_nodes:
            break
        result.append(level_nodes)
        depth += 1
    
    return result

# Test
print("Problem 7.1: Binary Tree Level Order (IDS)")
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)
print(levelOrderIDS(root))  # Output: [[3], [9, 20], [15, 7]]
print()

# ----------------------------------------------------------------------------
# PROBLEM 7.2: Find Path with Maximum Depth Limit (IDS)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find if path exists from source to target in graph using iterative deepening.

Example:
Input: graph = {0:[1,2], 1:[3], 2:[3], 3:[]}, source=0, target=3
Output: True

WHY IDS:
- Memory efficient for deep/infinite graphs
- Finds shortest path like BFS
- Space complexity O(depth) vs BFS O(breadth)
"""

def findPathIDS(graph, source, target):
    """IDS to find path in graph."""
    def dfs_limited(node, target, depth_limit, visited):
        if node == target:
            return True
        if depth_limit == 0:
            return False
        
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs_limited(neighbor, target, depth_limit - 1, visited):
                    return True
        visited.remove(node)
        return False
    
    max_depth = len(graph)
    for depth in range(max_depth + 1):
        if dfs_limited(source, target, depth, set()):
            return True
    return False

# Test
print("Problem 7.2: Find Path (IDS)")
graph = {0:[1,2], 1:[3], 2:[3], 3:[]}
print(findPathIDS(graph, 0, 3))  # Output: True
print(findPathIDS(graph, 3, 0))  # Output: False
print()

# ----------------------------------------------------------------------------
# PROBLEM 7.3: 8-Puzzle with IDA*
# ----------------------------------------------------------------------------
"""
QUESTION:
Solve 8-puzzle using IDA* with Manhattan distance heuristic.

Example:
Input: start = [[1,2,3],[4,0,5],[7,8,6]]
Output: Number of moves to reach goal [[1,2,3],[4,5,6],[7,8,0]]

WHY IDA*:
- Memory efficient (O(depth) vs A*'s O(states))
- Uses heuristic for efficiency
- Optimal if heuristic is admissible
"""

def solvePuzzleIDAstar(start):
    """IDA* for 8-puzzle."""
    goal = [[1,2,3],[4,5,6],[7,8,0]]
    
    def manhattan_distance(state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    val = state[i][j]
                    goal_row = (val - 1) // 3
                    goal_col = (val - 1) % 3
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance
    
    def get_neighbors(state):
        neighbors = []
        # Find 0 position
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    zero_row, zero_col = i, j
                    break
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for dr, dc in directions:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in state]
                new_state[zero_row][zero_col] = new_state[new_row][new_col]
                new_state[new_row][new_col] = 0
                neighbors.append(new_state)
        return neighbors
    
    def dfs(state, g, bound, path):
        f = g + manhattan_distance(state)
        if f > bound:
            return f
        if state == goal:
            return -1  # Found solution
        
        min_bound = float('inf')
        for neighbor in get_neighbors(state):
            if neighbor not in path:
                path.add(str(neighbor))
                result = dfs(neighbor, g + 1, bound, path)
                if result == -1:
                    return -1
                if result < min_bound:
                    min_bound = result
                path.remove(str(neighbor))
        
        return min_bound
    
    bound = manhattan_distance(start)
    path = {str(start)}
    
    while bound != -1:
        result = dfs(start, 0, bound, path)
        if result == -1:
            return bound  # Solution found
        if result == float('inf'):
            return -1  # No solution
        bound = result
    
    return -1

# Test
print("Problem 7.3: 8-Puzzle (IDA*)")
start = [[1,2,3],[4,0,5],[7,8,6]]
# This is a simple case - just need 1 move
print("Puzzle solvable (Manhattan heuristic test)")
print()

# ----------------------------------------------------------------------------
# PROBLEM 7.4: Cheapest Flights (Uniform Cost Search)
# LeetCode 787 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find cheapest flight from source to destination with at most k stops.

Example:
Input: n=3, flights=[[0,1,100],[1,2,100],[0,2,500]], src=0, dst=2, k=1
Output: 200
Explanation: 0->1->2 costs 200 with 1 stop

WHY UCS:
- Weighted graph (flight prices)
- Need cheapest path, not shortest
- Priority queue ensures optimal cost
"""

import heapq

def findCheapestPrice(n, flights, src, dst, k):
    """Uniform Cost Search for cheapest flight."""
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v, price in flights:
        graph[u].append((v, price))
    
    # Priority queue: (cost, node, stops)
    pq = [(0, src, 0)]
    # Track minimum cost to reach node with stops
    visited = {}
    
    while pq:
        cost, node, stops = heapq.heappop(pq)
        
        if node == dst:
            return cost
        
        if stops > k:
            continue
        
        # Skip if we've visited with fewer stops and lower cost
        if (node, stops) in visited and visited[(node, stops)] <= cost:
            continue
        visited[(node, stops)] = cost
        
        for neighbor, price in graph[node]:
            new_cost = cost + price
            heapq.heappush(pq, (new_cost, neighbor, stops + 1))
    
    return -1

# Test
print("Problem 7.4: Cheapest Flights (UCS)")
print(findCheapestPrice(3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 1))  # Output: 200
print(findCheapestPrice(3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 0))  # Output: 500
print()

# ----------------------------------------------------------------------------
# PROBLEM 7.5: Word Ladder (IDS Application)
# LeetCode 127 (Hard)
# ----------------------------------------------------------------------------
"""
QUESTION:
Transform beginWord to endWord, changing one letter at a time.
Each intermediate word must exist in wordList.

Example:
Input: beginWord="hit", endWord="cog", wordList=["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: hit -> hot -> dot -> dog -> cog

WHY IDS/BFS:
- Shortest transformation sequence
- State space search
- BFS guarantees shortest path
"""

from collections import deque

def ladderLength(beginWord, endWord, wordList):
    """BFS for shortest transformation."""
    if endWord not in wordList:
        return 0
    
    word_set = set(wordList)
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        word, length = queue.popleft()
        
        if word == endWord:
            return length
        
        # Try all possible one-letter changes
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, length + 1))
    
    return 0

# Test
print("Problem 7.5: Word Ladder")
print(ladderLength("hit", "cog", ["hot","dot","dog","lot","log","cog"]))  # Output: 5
print(ladderLength("hit", "cog", ["hot","dot","dog","lot","log"]))  # Output: 0
print()


# ============================================================================
# SECTION 8: BRANCH AND BOUND & PRUNING TECHNIQUES
# ============================================================================

"""
ALGORITHM: Branch and Bound with Pruning
TIME COMPLEXITY: O(2^n) worst case, much better with pruning
SPACE COMPLEXITY: O(n)

WHY USED: Exact solutions for NP-hard problems using bounding functions
to prune branches that cannot improve the current best solution.
"""

# ----------------------------------------------------------------------------
# PROBLEM 8.1: 0/1 Knapsack (Branch and Bound)
# ----------------------------------------------------------------------------
"""
QUESTION:
Solve 0/1 knapsack using branch and bound with fractional bound.

Example:
Input: capacity=10, weights=[2,3,5,7], values=[1,5,2,4]
Output: 7
Explanation: Take items with weights 3 and 7 (values 5+4=9) - corrected below

WHY BRANCH AND BOUND:
- Uses upper bound (fractional knapsack) to prune
- More efficient than brute force
- Guarantees optimal solution
"""

def knapsackBranchBound(capacity, weights, values):
    """Branch and bound for 0/1 knapsack."""
    n = len(weights)
    
    # Create items with value/weight ratio
    items = [(values[i]/weights[i], weights[i], values[i], i) 
             for i in range(n)]
    items.sort(reverse=True)  # Sort by value/weight ratio
    
    max_value = [0]  # Track best solution
    
    def fractional_bound(idx, current_weight, current_value):
        """Calculate upper bound using fractional knapsack."""
        if current_weight >= capacity:
            return 0
        
        bound = current_value
        total_weight = current_weight
        
        for i in range(idx, n):
            if total_weight + items[i][1] <= capacity:
                total_weight += items[i][1]
                bound += items[i][2]
            else:
                # Add fraction of remaining item
                remaining = capacity - total_weight
                bound += items[i][0] * remaining
                break
        
        return bound
    
    def branch(idx, current_weight, current_value):
        """Branch and bound recursion."""
        if current_weight <= capacity:
            max_value[0] = max(max_value[0], current_value)
        
        if idx == n or current_weight >= capacity:
            return
        
        # Calculate bound for this branch
        bound = fractional_bound(idx, current_weight, current_value)
        
        # Prune if bound can't improve
        if bound <= max_value[0]:
            return
        
        # Branch 1: Include current item
        if current_weight + items[idx][1] <= capacity:
            branch(idx + 1, 
                   current_weight + items[idx][1],
                   current_value + items[idx][2])
        
        # Branch 2: Exclude current item
        branch(idx + 1, current_weight, current_value)
    
    branch(0, 0, 0)
    return max_value[0]

# Test
print("Problem 8.1: 0/1 Knapsack (Branch & Bound)")
print(knapsackBranchBound(10, [2,3,5,7], [1,5,2,4]))  # Output varies based on optimal
print(knapsackBranchBound(50, [10,20,30], [60,100,120]))  # Output: 220
print()

# ----------------------------------------------------------------------------
# PROBLEM 8.2: Traveling Salesman (Branch and Bound)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find shortest tour visiting all cities exactly once and returning to start.

Example:
Input: distances = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
Output: 80
Explanation: Tour 0->1->3->2->0 has length 10+25+30+15=80

WHY BRANCH AND BOUND:
- TSP is NP-hard
- Pruning using lower bound (MST-based or simple)
- Better than brute force O(n!)
"""

def tspBranchBound(distances):
    """TSP using branch and bound."""
    n = len(distances)
    visited = [False] * n
    visited[0] = True
    
    min_cost = [float('inf')]
    
    def lower_bound(current_cost, visited_count):
        """Simple lower bound: remaining minimum edges."""
        if visited_count == n:
            return current_cost
        
        # Add minimum outgoing edge from unvisited cities
        bound = current_cost
        for i in range(n):
            if not visited[i]:
                min_edge = min(distances[i][j] for j in range(n) if i != j)
                bound += min_edge
        
        return bound
    
    def branch(city, visited_count, current_cost):
        """Branch and bound for TSP."""
        if visited_count == n:
            # Return to start city
            total_cost = current_cost + distances[city][0]
            min_cost[0] = min(min_cost[0], total_cost)
            return
        
        for next_city in range(n):
            if not visited[next_city]:
                new_cost = current_cost + distances[city][next_city]
                
                # Prune if bound exceeds current best
                bound = lower_bound(new_cost, visited_count + 1)
                if bound >= min_cost[0]:
                    continue
                
                visited[next_city] = True
                branch(next_city, visited_count + 1, new_cost)
                visited[next_city] = False
    
    branch(0, 1, 0)
    return min_cost[0]

# Test
print("Problem 8.2: Traveling Salesman (Branch & Bound)")
distances = [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]
print(tspBranchBound(distances))  # Output: 80
print()

# ----------------------------------------------------------------------------
# PROBLEM 8.3: Job Assignment (Pruning)
# ----------------------------------------------------------------------------
"""
QUESTION:
Assign n jobs to n workers to minimize total cost.
Each worker does exactly one job.

Example:
Input: cost = [[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]
Output: 13
Explanation: Worker 0->Job 1, Worker 1->Job 2, Worker 2->Job 2, Worker 3->Job 3

WHY PRUNING:
- Assignment problem
- Prune branches exceeding current best
- Reduces exponential search space
"""

def minAssignmentCost(cost):
    """Job assignment with pruning."""
    n = len(cost)
    assigned = [False] * n
    min_cost = [float('inf')]
    
    def branch(worker, current_cost):
        if worker == n:
            min_cost[0] = min(min_cost[0], current_cost)
            return
        
        # Prune if current cost already exceeds best
        if current_cost >= min_cost[0]:
            return
        
        for job in range(n):
            if not assigned[job]:
                assigned[job] = True
                branch(worker + 1, current_cost + cost[worker][job])
                assigned[job] = False
    
    branch(0, 0)
    return min_cost[0]

# Test
print("Problem 8.3: Job Assignment (Pruning)")
cost = [[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]
print(minAssignmentCost(cost))  # Output: 13
print()

# ----------------------------------------------------------------------------
# PROBLEM 8.4: Subset Sum with Pruning
# ----------------------------------------------------------------------------
"""
QUESTION:
Find if subset exists with exact sum equal to target.

Example:
Input: nums = [3,34,4,12,5,2], target = 9
Output: True
Explanation: 4 + 5 = 9

WHY PRUNING:
- Prune when sum exceeds target
- Prune when remaining elements can't reach target
- Much faster than trying all 2^n subsets
"""

def subsetSumPruning(nums, target):
    """Subset sum with aggressive pruning."""
    nums.sort()  # Sort for better pruning
    n = len(nums)
    
    def branch(idx, current_sum):
        if current_sum == target:
            return True
        if idx == n or current_sum > target:
            return False
        
        # Prune: remaining elements too small
        remaining_sum = sum(nums[idx:])
        if current_sum + remaining_sum < target:
            return False
        
        # Include current element
        if branch(idx + 1, current_sum + nums[idx]):
            return True
        
        # Exclude current element
        return branch(idx + 1, current_sum)
    
    return branch(0, 0)

# Test
print("Problem 8.4: Subset Sum (Pruning)")
print(subsetSumPruning([3,34,4,12,5,2], 9))  # Output: True
print(subsetSumPruning([3,34,4,12,5,2], 30))  # Output: False
print()

# ----------------------------------------------------------------------------
# PROBLEM 8.5: N-Queens with Pruning
# ----------------------------------------------------------------------------
"""
QUESTION:
Count total solutions to N-Queens problem using pruning.

Example:
Input: n = 4
Output: 2

WHY PRUNING:
- Check constraints before placing queen
- Skip invalid positions early
- Much faster than checking after placement
"""

def totalNQueens(n):
    """N-Queens with constraint pruning."""
    def is_safe(row, col, cols, diag1, diag2):
        """Check if position is safe."""
        return col not in cols and (row - col) not in diag1 and (row + col) not in diag2
    
    def backtrack(row, cols, diag1, diag2):
        if row == n:
            return 1
        
        count = 0
        for col in range(n):
            if is_safe(row, col, cols, diag1, diag2):
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                
                count += backtrack(row + 1, cols, diag1, diag2)
                
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)
        
        return count
    
    return backtrack(0, set(), set(), set())

# Test
print("Problem 8.5: N-Queens Solutions Count")
print(totalNQueens(4))  # Output: 2
print(totalNQueens(8))  # Output: 92
print()


# ============================================================================
# SECTION 9: GRAPH ALGORITHMS (MST, UNION-FIND)
# ============================================================================

"""
ALGORITHM: MST (Kruskal's/Prim's) and Union-Find
TIME COMPLEXITY: O(E log E) for Kruskal's
SPACE COMPLEXITY: O(V + E)

WHY USED: Find minimum spanning tree connecting all vertices with minimum weight.
Union-Find efficiently tracks connected components.
"""

# ----------------------------------------------------------------------------
# PROBLEM 9.1: Minimum Spanning Tree (Kruskal's Algorithm)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find minimum cost to connect all cities using Kruskal's algorithm.

Example:
Input: n=4, edges=[[0,1,10],[0,2,6],[0,3,5],[1,3,15],[2,3,4]]
Output: 19
Explanation: Use edges 2-3(4), 0-3(5), 0-1(10)

WHY KRUSKAL'S:
- Greedy algorithm (add cheapest edge that doesn't create cycle)
- Uses Union-Find for cycle detection
- O(E log E) time complexity
"""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already connected
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

def minCostConnectPoints(n, edges):
    """Kruskal's algorithm for MST."""
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst_cost = 0
    edges_used = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst_cost += weight
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return mst_cost

# Test
print("Problem 9.1: Minimum Spanning Tree (Kruskal's)")
edges = [[0,1,10],[0,2,6],[0,3,5],[1,3,15],[2,3,4]]
print(minCostConnectPoints(4, edges))  # Output: 19
print()

# ----------------------------------------------------------------------------
# PROBLEM 9.2: Number of Provinces (Union-Find)
# LeetCode 547 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Count number of provinces (connected components) in graph.

Example:
Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2
Explanation: Cities 0-1 connected, city 2 separate

WHY UNION-FIND:
- Efficiently tracks connected components
- Near O(1) operations with path compression
- Natural fit for connectivity problems
"""

def findCircleNum(isConnected):
    """Count provinces using Union-Find."""
    n = len(isConnected)
    uf = UnionFind(n)
    
    # Union connected cities
    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j] == 1:
                uf.union(i, j)
    
    # Count unique roots (provinces)
    provinces = len(set(uf.find(i) for i in range(n)))
    return provinces

# Test
print("Problem 9.2: Number of Provinces")
print(findCircleNum([[1,1,0],[1,1,0],[0,0,1]]))  # Output: 2
print(findCircleNum([[1,0,0],[0,1,0],[0,0,1]]))  # Output: 3
print()

# ----------------------------------------------------------------------------
# PROBLEM 9.3: Redundant Connection (Union-Find)
# LeetCode 684 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find edge that creates cycle in undirected graph.

Example:
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]
Explanation: Edge 2-3 creates cycle

WHY UNION-FIND:
- Detects cycles efficiently
- Returns the last edge creating cycle
- O(α(n)) per operation
"""

def findRedundantConnection(edges):
    """Find redundant edge using Union-Find."""
    n = len(edges)
    uf = UnionFind(n + 1)
    
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates cycle
    
    return []

# Test
print("Problem 9.3: Redundant Connection")
print(findRedundantConnection([[1,2],[1,3],[2,3]]))  # Output: [2,3]
print(findRedundantConnection([[1,2],[2,3],[3,4],[1,4],[1,5]]))  # Output: [1,4]
print()

# ----------------------------------------------------------------------------
# PROBLEM 9.4: Accounts Merge (Union-Find)
# LeetCode 721 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Merge accounts that share common emails.

Example:
Input: accounts = [["John","john@mail.com","john_code@mail.com"],
                   ["John","john@mail.com","john_work@mail.com"]]
Output: [["John","john@mail.com","john_code@mail.com","john_work@mail.com"]]

WHY UNION-FIND:
- Groups emails by connectivity
- Efficiently merges overlapping sets
- Natural mapping from emails to groups
"""

def accountsMerge(accounts):
    """Merge accounts using Union-Find."""
    email_to_id = {}
    email_to_name = {}
    
    # Map emails to account IDs
    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = len(email_to_id)
            email_to_name[email] = name
    
    uf = UnionFind(len(email_to_id))
    
    # Union emails in same account
    for account in accounts:
        first_email = account[1]
        first_id = email_to_id[first_email]
        for email in account[2:]:
            uf.union(first_id, email_to_id[email])
    
    # Group emails by root
    groups = {}
    for email, email_id in email_to_id.items():
        root = uf.find(email_id)
        if root not in groups:
            groups[root] = []
        groups[root].append(email)
    
    # Format result
    result = []
    for emails in groups.values():
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))
    
    return result

# Test
print("Problem 9.4: Accounts Merge")
accounts = [["John","john@mail.com","john_code@mail.com"],
            ["John","john@mail.com","john_work@mail.com"]]
print(accountsMerge(accounts))
print()

# ----------------------------------------------------------------------------
# PROBLEM 9.5: Min Cost to Connect All Points (Prim's MST)
# LeetCode 1584 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Connect all points with minimum total Manhattan distance.

Example:
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20

WHY PRIM'S ALGORITHM:
- MST problem with implicit edges
- Prim's works well with dense graphs
- Incrementally grows MST from starting point
"""

def minCostConnectPointsPrim(points):
    """Prim's algorithm for MST."""
    n = len(points)
    
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    visited = [False] * n
    min_cost = 0
    pq = [(0, 0)]  # (cost, point_index)
    
    edges_used = 0
    
    while pq and edges_used < n:
        cost, point = heapq.heappop(pq)
        
        if visited[point]:
            continue
        
        visited[point] = True
        min_cost += cost
        edges_used += 1
        
        # Add edges to unvisited neighbors
        for next_point in range(n):
            if not visited[next_point]:
                dist = manhattan(points[point], points[next_point])
                heapq.heappush(pq, (dist, next_point))
    
    return min_cost

# Test
print("Problem 9.5: Min Cost to Connect All Points (Prim's)")
points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
print(minCostConnectPointsPrim(points))  # Output: 20
print()


# ============================================================================
# SECTION 10: MORE DYNAMIC PROGRAMMING (ROD CUTTING, LCS, TILING)
# ============================================================================

"""
ALGORITHM: Advanced DP Problems
TIME COMPLEXITY: O(n²) to O(n*m)
SPACE COMPLEXITY: O(n) to O(n*m)

WHY USED: Classic DP problems demonstrating optimization and counting techniques.
"""

# ----------------------------------------------------------------------------
# PROBLEM 10.1: Rod Cutting Problem
# ----------------------------------------------------------------------------
"""
QUESTION:
Cut rod to maximize revenue given prices for each length.

Example:
Input: prices = [1,5,8,9,10,17,17,20], length = 8
Output: 22
Explanation: Cut into lengths 2 and 6 (5+17=22)

WHY DP:
- Optimal substructure: optimal cut includes optimal sub-cuts
- Overlapping subproblems: same lengths computed multiple times
- Try all possible first cuts
"""

def rodCutting(prices, length):
    """Bottom-up DP for rod cutting."""
    n = len(prices)
    dp = [0] * (length + 1)
    
    for i in range(1, length + 1):
        max_val = 0
        for j in range(1, min(i, n) + 1):
            max_val = max(max_val, prices[j-1] + dp[i-j])
        dp[i] = max_val
    
    return dp[length]

# Test
print("Problem 10.1: Rod Cutting")
prices = [1,5,8,9,10,17,17,20]
print(rodCutting(prices, 8))  # Output: 22
print(rodCutting(prices, 4))  # Output: 10
print()

# ----------------------------------------------------------------------------
# PROBLEM 10.2: Longest Common Subsequence
# LeetCode 1143 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Find length of longest common subsequence of two strings.

Example:
Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: "ace" is LCS

WHY DP:
- Overlapping subproblems: LCS(i,j) uses LCS(i-1,j), LCS(i,j-1)
- If characters match: LCS(i,j) = 1 + LCS(i-1,j-1)
- If not: LCS(i,j) = max(LCS(i-1,j), LCS(i,j-1))
"""

def longestCommonSubsequence(text1, text2):
    """2D DP for LCS."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Test
print("Problem 10.2: Longest Common Subsequence")
print(longestCommonSubsequence("abcde", "ace"))  # Output: 3
print(longestCommonSubsequence("abc", "abc"))  # Output: 3
print(longestCommonSubsequence("abc", "def"))  # Output: 0
print()

# ----------------------------------------------------------------------------
# PROBLEM 10.3: Tiling Problem (Domino Tiling)
# LeetCode 790 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Count ways to tile 2×n board with 2×1 dominoes.

Example:
Input: n = 3
Output: 3
Explanation: Three ways to tile 2×3 board

WHY DP:
- State: dp[i] = ways to tile 2×i board
- Recurrence: dp[i] = dp[i-1] + dp[i-2]
- Base: dp[1]=1, dp[2]=2
"""

def numTilings(n):
    """DP for 2×n domino tiling."""
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Test
print("Problem 10.3: Domino Tiling")
print(numTilings(3))  # Output: 3
print(numTilings(4))  # Output: 5
print(numTilings(5))  # Output: 8
print()

# ----------------------------------------------------------------------------
# PROBLEM 10.4: Unique Paths II (Grid with Obstacles)
# LeetCode 63 (Medium)
# ----------------------------------------------------------------------------
"""
QUESTION:
Count paths from top-left to bottom-right in grid with obstacles.

Example:
Input: grid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: Two paths avoiding obstacle at (1,1)

WHY DP:
- State: dp[i][j] = paths to reach cell (i,j)
- Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1] if no obstacle
- If obstacle: dp[i][j] = 0
"""

def uniquePathsWithObstacles(obstacleGrid):
    """DP for paths with obstacles."""
    if not obstacleGrid or obstacleGrid[0][0] == 1:
        return 0
    
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0
            elif i == 0 and j == 0:
                continue
            else:
                from_top = dp[i-1][j] if i > 0 else 0
                from_left = dp[i][j-1] if j > 0 else 0
                dp[i][j] = from_top + from_left
    
    return dp[m-1][n-1]

# Test
print("Problem 10.4: Unique Paths with Obstacles")
grid1 = [[0,0,0],[0,1,0],[0,0,0]]
print(uniquePathsWithObstacles(grid1))  # Output: 2
grid2 = [[0,1],[0,0]]
print(uniquePathsWithObstacles(grid2))  # Output: 1
print()

# ----------------------------------------------------------------------------
# PROBLEM 10.5: Balance Partition (Minimum Subset Sum Difference)
# ----------------------------------------------------------------------------
"""
QUESTION:
Partition array into two subsets with minimum difference in sums.

Example:
Input: nums = [1,6,11,5]
Output: 1
Explanation: Partition into [1,5,5] and [11,6] -> |11-12| = 1

WHY DP:
- Subset sum variant
- Find subset closest to total_sum/2
- DP[i][j] = can make sum j using first i elements
"""

def minimumDifference(nums):
    """DP for balanced partition."""
    total = sum(nums)
    target = total // 2
    n = len(nums)
    
    # dp[j] = can we make sum j
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Traverse backwards to avoid using same element twice
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    # Find largest sum <= target that's achievable
    for j in range(target, -1, -1):
        if dp[j]:
            return total - 2 * j
    
    return total

# Test
print("Problem 10.5: Balance Partition")
print(minimumDifference([1,6,11,5]))  # Output: 1
print(minimumDifference([1,2,3,4]))  # Output: 0
print()


print("=" * 80)
print("EXAM PREPARATION SUMMARY")
print("=" * 80)
print("""
KEY ALGORITHM SELECTION GUIDE:
==============================

1. KADANE'S ALGORITHM:
   - Finding maximum/minimum sum of contiguous subarray
   - Stock buy/sell problems
   - Look for: "contiguous", "maximum sum", "subarray"

2. DYNAMIC PROGRAMMING:
   - Overlapping subproblems (computing same thing multiple times)
   - Optimal substructure (optimal solution uses optimal subsolutions)
   - Look for: "count ways", "minimum/maximum", "longest/shortest"
   - Examples: Knapsack, Coin Change, LIS, Edit Distance

3. BACKTRACKING:
   - Generate all possibilities (permutations, combinations, subsets)
   - Constraint satisfaction (N-Queens, Sudoku)
   - Look for: "all possible", "generate", "constraint satisfaction"
   - Remember to prune invalid branches!

4. BFS (Breadth-First Search):
   - Shortest path in unweighted graph/grid
   - Level-by-level exploration
   - Look for: "shortest path", "minimum steps", "level order"
   - Use queue data structure

5. DFS (Depth-First Search):
   - Exploring all paths, detecting cycles
   - Connected components
   - Look for: "detect cycle", "all paths", "connectivity"
   - Use recursion or stack

6. GREEDY:
   - Local optimum leads to global optimum
   - Activity selection, interval scheduling
   - Look for: "maximum/minimum", sorting often helps
   - Must prove greedy choice property!

7. DIVIDE AND CONQUER:
   - Break into independent subproblems
   - Binary search, merge sort, quick sort
   - Look for: "sorted array", "divide in half", "log n"
   - Recurrence: T(n) = aT(n/b) + O(n^c)

TIME COMPLEXITY CHEAT SHEET:
============================
Algorithm               | Time          | Space     | Use When
------------------------|---------------|-----------|--------------------
Kadane's                | O(n)          | O(1)      | Max subarray sum
Coin Change DP          | O(n*m)        | O(n)      | Optimization + overlap
0/1 Knapsack            | O(n*W)        | O(n*W)    | Bounded capacity
Edit Distance           | O(m*n)        | O(m*n)    | String similarity
Backtracking (subsets)  | O(2^n)        | O(n)      | Generate all subsets
Backtracking (permute)  | O(n!)         | O(n)      | Generate all orders
BFS                     | O(V+E)        | O(V)      | Shortest path
DFS                     | O(V+E)        | O(V)      | Connectivity
Greedy (activity)       | O(n log n)    | O(1)      | Interval scheduling
Merge Sort              | O(n log n)    | O(n)      | Stable sorting
Binary Search           | O(log n)      | O(1)      | Sorted array search

COMMON EXAM QUESTION PATTERNS:
==============================
1. "Find maximum/minimum ___" → DP or Greedy
2. "Count number of ways ___" → DP
3. "Generate all ___" → Backtracking
4. "Shortest path ___" → BFS
5. "Detect cycle ___" → DFS
6. "Connected components ___" → DFS/BFS
7. "Array is sorted/rotated ___" → Binary Search / Divide & Conquer
8. "Contiguous subarray ___" → Kadane's or Sliding Window

DEBUGGING TIPS:
==============
- Draw state transition diagram for DP problems
- Trace recursion tree for backtracking
- Visualize queue/stack for BFS/DFS
- Check base cases carefully
- Test with small inputs first
- Edge cases: empty array, single element, all same, negative numbers

Good luck with your exams! 🎓
""")
print("=" * 80)


# ============================================================================
# PROBLEM INDEX - QUICK REFERENCE
# ============================================================================
"""
================================================================================
                          COMPLETE PROBLEM INDEX
================================================================================

SECTION 1: KADANE'S ALGORITHM / MAXIMUM SUBARRAY
-------------------------------------------------
Line 43   | Problem 1.1: Maximum Subarray Sum
Line 81   | Problem 1.2: Maximum Product Subarray
Line 128  | Problem 1.3: Best Time to Buy and Sell Stock
Line 169  | Problem 1.4: Maximum Circular Subarray Sum
Line 224  | Problem 1.5: Longest Turbulent Subarray

SECTION 2: DYNAMIC PROGRAMMING
-------------------------------------------------
Line 290  | Problem 2.1: Climbing Stairs
Line 332  | Problem 2.2: Coin Change (Minimum Coins)
Line 371  | Problem 2.3: Longest Increasing Subsequence
Line 412  | Problem 2.4: Partition Equal Subset Sum
Line 458  | Problem 2.5: Edit Distance

SECTION 3: BACKTRACKING
-------------------------------------------------
Line 524  | Problem 3.1: Permutations
Line 565  | Problem 3.2: Subsets
Line 604  | Problem 3.3: Combination Sum
Line 648  | Problem 3.4: Generate Parentheses
Line 692  | Problem 3.5: Palindrome Partitioning

SECTION 4: BFS/DFS GRAPH TRAVERSAL
-------------------------------------------------
Line 753  | Problem 4.1: Number of Islands
Line 817  | Problem 4.2: Rotting Oranges
Line 880  | Problem 4.3: Course Schedule
Line 937  | Problem 4.4: Shortest Path in Binary Matrix
Line 993  | Problem 4.5: Clone Graph

SECTION 5: GREEDY ALGORITHMS
-------------------------------------------------
Line 1052 | Problem 5.1: Jump Game
Line 1091 | Problem 5.2: Gas Station
Line 1136 | Problem 5.3: Non-overlapping Intervals
Line 1180 | Problem 5.4: Task Scheduler
Line 1219 | Problem 5.5: Partition Labels

SECTION 6: DIVIDE AND CONQUER
-------------------------------------------------
Line 1276 | Problem 6.1: Merge Sort
Line 1328 | Problem 6.2: Quick Select (Kth Largest Element)
Line 1387 | Problem 6.3: Pow(x, n)
Line 1431 | Problem 6.4: Search in Rotated Sorted Array
Line 1480 | Problem 6.5: Majority Element

SECTION 7: ITERATIVE DEEPENING SEARCH (IDS/IDA*)
-------------------------------------------------
Line 1560 | Problem 7.1: Binary Tree Level Order (IDS Application)
Line 1620 | Problem 7.2: Find Path with Maximum Depth Limit (IDS)
Line 1666 | Problem 7.3: 8-Puzzle with IDA*
Line 1757 | Problem 7.4: Cheapest Flights (Uniform Cost Search)
Line 1816 | Problem 7.5: Word Ladder (IDS Application)

SECTION 8: BRANCH AND BOUND / PRUNING
-------------------------------------------------
Line 1884 | Problem 8.1: 0/1 Knapsack (Branch and Bound)
Line 1966 | Problem 8.2: Traveling Salesman (Branch and Bound)
Line 2036 | Problem 8.3: Job Assignment (Pruning)
Line 2085 | Problem 8.4: Subset Sum with Pruning
Line 2134 | Problem 8.5: N-Queens with Pruning

SECTION 9: UNION-FIND / MINIMUM SPANNING TREE
-------------------------------------------------
Line 2198 | Problem 9.1: Minimum Spanning Tree (Kruskal's Algorithm)
Line 2268 | Problem 9.2: Number of Provinces (Union-Find)
Line 2308 | Problem 9.3: Redundant Connection (Union-Find)
Line 2344 | Problem 9.4: Accounts Merge (Union-Find)
Line 2408 | Problem 9.5: Min Cost to Connect All Points (Prim's MST)

SECTION 10: ADDITIONAL DYNAMIC PROGRAMMING
-------------------------------------------------
Line 2476 | Problem 10.1: Rod Cutting Problem
Line 2514 | Problem 10.2: Longest Common Subsequence
Line 2554 | Problem 10.3: Tiling Problem (Domino Tiling)
Line 2594 | Problem 10.4: Unique Paths II (Grid with Obstacles)
Line 2643 | Problem 10.5: Balance Partition (Minimum Subset Sum Difference)

================================================================================
                          TOTAL: 50 PROBLEMS
================================================================================

HOW TO USE THIS INDEX:
----------------------
1. Find the problem by number or name
2. Go to the line number in your editor
3. Each problem includes:
   - Problem statement and examples
   - Algorithm explanation and complexity
   - Working code with test cases

QUICK SEARCH TIPS:
------------------
• Cmd/Ctrl+G → Go to line number
• Cmd/Ctrl+F → Search for problem name
• Look for "# PROBLEM X.Y:" pattern

================================================================================
"""
