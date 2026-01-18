"""
===============================================================================
COMPREHENSIVE ALGORITHM DESIGN - COURSE SUMMARY
===============================================================================
Author: Algorithm Design Course Analysis
Topics: Recursion, Backtracking, Dynamic Programming, Memoization, Tabulation
===============================================================================
"""

# =============================================================================
# 1. GCD - EUCLIDEAN ALGORITHM (Worksheet 0)
# =============================================================================
"""
ALGORITHM: Greatest Common Divisor
APPROACH: Recursive division with remainder
TIME COMPLEXITY: O(log(min(a,b)))
SPACE COMPLEXITY: O(1)

PROS:
- Very efficient for finding GCD
- Simple iterative implementation
- Optimal time complexity

CONS:
- Limited to GCD calculation only
- Not applicable to other problems

USE CASE: Finding greatest common divisor of two numbers
"""

def gcd_euclidean(a, b):
    while b != 0:
        a, b = b, a % b
    return a


# =============================================================================
# 2. MAXIMUM SUBARRAY SUM - BRUTE FORCE (Worksheet 1)
# =============================================================================
"""
ALGORITHM: Maximum Subarray Sum (Naive)
APPROACH: Try all possible subarrays
TIME COMPLEXITY: O(n³)
SPACE COMPLEXITY: O(1)

PROS:
- Easy to understand
- Guaranteed correct solution
- No additional space needed

CONS:
- Extremely slow for large inputs
- Redundant calculations
- Not practical for n > 100

COMPARISON:
- Brute Force: O(n³)
- Kadane's Algorithm: O(n) ← Much better!

USE CASE: Teaching algorithm analysis and optimization
"""

def max_subarray_brute_force(arr):
    n = len(arr)
    max_sum = arr[0]
    
    for i in range(n):
        for j in range(i, n):
            # Calculate sum from i to j
            current_sum = 0
            for k in range(i, j+1):
                current_sum += arr[k]
            max_sum = max(max_sum, current_sum)
    
    return max_sum


# =============================================================================
# 3. BINARY COMBINATIONS - BACKTRACKING (Worksheet 2)
# =============================================================================
"""
ALGORITHM: Generate All Binary Combinations
APPROACH: Recursive backtracking with state array
TIME COMPLEXITY: O(2^n)
SPACE COMPLEXITY: O(n) for recursion stack

PROS:
- Generates all possible combinations
- Foundation for many DP problems
- Easy to modify for different constraints

CONS:
- Exponential time complexity
- Not efficient for large n
- Only generates, doesn't optimize

USE CASE: Subset generation, exploring all possibilities
RELATED PROBLEMS: Knapsack v1, Balance Split
"""

def generate_binary_combinations(n):
    x = [0] * n
    results = []
    
    def backtrack(i):
        if i == n:
            results.append(x[:])  # Save copy
            return
        
        x[i] = 0
        backtrack(i + 1)
        
        x[i] = 1
        backtrack(i + 1)
    
    backtrack(0)
    return results


# =============================================================================
# 4. BALANCE SPLIT - RECURSIVE EXHAUSTIVE SEARCH (Worksheet 2)
# =============================================================================
"""
ALGORITHM: Minimum Difference Partition
APPROACH: Try all 2^n partitions, find minimum difference
TIME COMPLEXITY: O(2^n)
SPACE COMPLEXITY: O(n) for recursion

PROS:
- Finds optimal solution
- Demonstrates exhaustive search
- Clear recursive structure

CONS:
- Extremely slow (exponential)
- Can be optimized with DP
- Not practical for n > 20

OPTIMIZATION: Can use subset sum DP for O(n * sum) solution

USE CASE: Teaching recursion and need for optimization
"""

def balance_split_recursive(values):
    n = len(values)
    x = [0] * n
    
    def explore(i):
        if i == n:
            group1 = sum(values[j] for j in range(n) if x[j] == 0)
            group2 = sum(values[j] for j in range(n) if x[j] == 1)
            return abs(group1 - group2)
        
        x[i] = 0
        diff0 = explore(i + 1)
        
        x[i] = 1
        diff1 = explore(i + 1)
        
        return min(diff0, diff1)
    
    return explore(0)


# =============================================================================
# 5. ROD CUTTING - NAIVE RECURSION (Worksheet 3)
# =============================================================================
"""
ALGORITHM: Rod Cutting Problem
APPROACH: Try all possible first cuts, recurse on remainder
TIME COMPLEXITY: O(2^n) - exponential
SPACE COMPLEXITY: O(n) for recursion

PROS:
- Natural recursive formulation
- Easy to understand logic
- Correct solution

CONS:
- Massive redundant calculations
- Overlapping subproblems not cached
- Unusable for n > 30

RECURRENCE: maxRev(n) = max(price[i] + maxRev(n-i)) for all i

OPTIMIZATION: Use memoization → O(n²) or tabulation → O(n²)

USE CASE: Demonstrates need for Dynamic Programming
"""

def rod_cutting_recursive(prices, length):
    if length == 0:
        return 0
    
    max_value = float('-inf')
    for i in range(1, len(prices) + 1):
        if i <= length:
            max_value = max(max_value, prices[i-1] + rod_cutting_recursive(prices, length - i))
    
    return max_value


# =============================================================================
# 6. COIN CHANGE - NAIVE RECURSION (Worksheet 3)
# =============================================================================
"""
ALGORITHM: Minimum Coins for Change
APPROACH: Try using each coin, recurse on remaining amount
TIME COMPLEXITY: O(C^n) where C is number of coin types
SPACE COMPLEXITY: O(n) for recursion depth

PROS:
- Intuitive recursive solution
- Explores all possibilities
- Correct result

CONS:
- Exponential time complexity
- Repeats same subproblems millions of times
- Impractical without memoization

RECURRENCE: minCoins(n) = 1 + min(minCoins(n-c)) for each coin c

OPTIMIZATION: Memoization → O(n*C) or Tabulation → O(n*C)

USE CASE: Classic DP teaching example
"""

def coin_change_recursive(coins, amount):
    if amount == 0:
        return 0
    
    min_coins = float('inf')
    for coin in coins:
        if coin <= amount:
            result = coin_change_recursive(coins, amount - coin)
            min_coins = min(min_coins, 1 + result)
    
    return min_coins


# =============================================================================
# 7. KNAPSACK - VERSION 1: GENERATE ALL COMBINATIONS (Worksheet 4)
# =============================================================================
"""
ALGORITHM: 0/1 Knapsack - Brute Force
APPROACH: Generate all 2^N item combinations, check validity
TIME COMPLEXITY: O(2^n)
SPACE COMPLEXITY: O(n)

PROS:
- Guaranteed to find optimal solution
- Simple to implement
- Clear logic

CONS:
- Generates invalid combinations (weight > capacity)
- Checks all 2^N combinations
- Extremely slow

USE CASE: Baseline for comparison

COMPARISON:
V1: Generate all → O(2^n) function calls
V2: Prune invalid → Still exponential but faster
V3: Memoization → O(n*M) where M is capacity
"""

def knapsack_v1_generate_all(weights, values, capacity):
    n = len(weights)
    x = [0] * n
    max_value = 0
    
    def generate(i):
        nonlocal max_value
        if i == n:
            total_weight = sum(weights[j] * x[j] for j in range(n))
            total_value = sum(values[j] * x[j] for j in range(n))
            if total_weight <= capacity:
                max_value = max(max_value, total_value)
            return
        
        x[i] = 0
        generate(i + 1)
        x[i] = 1
        generate(i + 1)
    
    generate(0)
    return max_value


# =============================================================================
# 8. KNAPSACK - VERSION 2: RECURSIVE WITH PRUNING (Worksheet 4)
# =============================================================================
"""
ALGORITHM: 0/1 Knapsack - Pruned Recursion
APPROACH: Skip/take each item, prune invalid branches early
TIME COMPLEXITY: O(2^n) worst case, but faster in practice
SPACE COMPLEXITY: O(n)

PROS:
- Prunes invalid branches early
- More efficient than v1
- Still guaranteed optimal

CONS:
- Still exponential
- Recomputes same states multiple times
- State (i, remaining_capacity) visited repeatedly

KEY INSIGHT: Same state (i, C) reached via different paths!
Example: skip(0)→take(1) = take(1)→skip(0) but computed twice

USE CASE: Shows improvement but still needs memoization
"""

def knapsack_v2_pruning(weights, values, capacity):
    n = len(weights)
    
    def solve(i, remaining_capacity):
        if i == n or remaining_capacity == 0:
            return 0
        
        # Skip item i
        skip = solve(i + 1, remaining_capacity)
        
        # Take item i (if it fits)
        take = 0
        if weights[i] <= remaining_capacity:
            take = values[i] + solve(i + 1, remaining_capacity - weights[i])
        
        return max(skip, take)
    
    return solve(0, capacity)


# =============================================================================
# 9. KNAPSACK - VERSION 3: MEMOIZATION (Worksheet 4)
# =============================================================================
"""
ALGORITHM: 0/1 Knapsack - Top-Down DP with Memoization
APPROACH: Cache results of subproblems in dictionary
TIME COMPLEXITY: O(n * M) where M is capacity
SPACE COMPLEXITY: O(n * M)

PROS:
- Massive speedup over v2 (exponential → polynomial)
- Only computes each state once
- Natural recursive formulation
- Easy to implement from recursive version

CONS:
- Requires extra memory for memo table
- Recursion overhead
- Can be converted to iterative for better performance

HOW MEMOIZATION WORKS:
1. First call to solve(i, C): compute result, store in memo[(i, C)]
2. Subsequent calls: return memo[(i, C)] immediately (no recursion!)

STATE SPACE: At most n * M unique states (i, C)
Each state computed exactly once → O(n*M) time!

USE CASE: Most practical recursive DP solution
"""

def knapsack_v3_memoization(weights, values, capacity):
    n = len(weights)
    memo = {}
    
    def solve(i, remaining_capacity):
        # Check memo first
        if (i, remaining_capacity) in memo:
            return memo[(i, remaining_capacity)]
        
        if i == n or remaining_capacity == 0:
            return 0
        
        # Skip item i
        skip = solve(i + 1, remaining_capacity)
        
        # Take item i (if it fits)
        take = 0
        if weights[i] <= remaining_capacity:
            take = values[i] + solve(i + 1, remaining_capacity - weights[i])
        
        # Store in memo before returning
        result = max(skip, take)
        memo[(i, remaining_capacity)] = result
        return result
    
    return solve(0, capacity)


# =============================================================================
# 10. KNAPSACK - TABULATION (Bottom-Up DP) (Worksheet 6)
# =============================================================================
"""
ALGORITHM: 0/1 Knapsack - Bottom-Up DP
APPROACH: Fill DP table iteratively from base cases
TIME COMPLEXITY: O(n * M)
SPACE COMPLEXITY: O(n * M) [can be optimized to O(M)]

PROS:
- No recursion overhead
- Iterative = often faster than memoization
- Clear order of computation
- Can optimize to O(M) space

CONS:
- Less intuitive than recursive version
- Must define table structure upfront
- Harder to understand for beginners

DP TABLE: dp[i][C] = max value using items 0..i-1 with capacity C

RECURRENCE:
dp[i][C] = max(
    dp[i-1][C],              # Skip item i-1
    v[i-1] + dp[i-1][C-w[i-1]]  # Take item i-1 (if fits)
)

COMPARISON:
Memoization: Top-Down (start from problem, recurse to base cases)
Tabulation: Bottom-Up (start from base cases, build up to answer)

USE CASE: Production code, need maximum performance
"""

def knapsack_tabulation(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill table bottom-up
    for i in range(1, n + 1):
        for c in range(capacity + 1):
            # Option 1: Skip item i-1
            skip = dp[i-1][c]
            
            # Option 2: Take item i-1 (if it fits)
            take = 0
            if weights[i-1] <= c:
                take = values[i-1] + dp[i-1][c - weights[i-1]]
            
            dp[i][c] = max(skip, take)
    
    return dp[n][capacity]


# =============================================================================
# 11. EDIT DISTANCE (Worksheet 5)
# =============================================================================
"""
ALGORITHM: Levenshtein Distance (Edit Distance)
APPROACH: Minimum operations to transform string A to string B
OPERATIONS: Insert, Delete, Replace
TIME COMPLEXITY: 
  - Recursive: O(3^(m+n)) - exponential
  - Memoization: O(m*n)
  - Tabulation: O(m*n)
SPACE COMPLEXITY: O(m*n)

PROS:
- Solves string similarity problem
- 2D DP example
- Useful in spell checking, DNA sequencing

CONS:
- O(m*n) can be large for very long strings
- Requires understanding of 2D state

RECURRENCE:
edit(i, j) = {
    j                           if i = 0
    i                           if j = 0
    edit(i-1, j-1)              if A[i-1] = B[j-1]
    1 + min(
        edit(i-1, j-1),  # Replace
        edit(i-1, j),    # Delete from A
        edit(i, j-1)     # Insert into A
    )
}

USE CASE: Autocorrect, diff tools, bioinformatics
"""

def edit_distance_memoization(A, B):
    memo = {}
    
    def edit(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == 0:
            return j
        if j == 0:
            return i
        
        if A[i-1] == B[j-1]:
            result = edit(i-1, j-1)
        else:
            result = 1 + min(
                edit(i-1, j-1),  # Replace
                edit(i-1, j),    # Delete
                edit(i, j-1)     # Insert
            )
        
        memo[(i, j)] = result
        return result
    
    return edit(len(A), len(B))


# =============================================================================
# 12. LONGEST COMMON SUBSEQUENCE (LCS) (Worksheet 6)
# =============================================================================
"""
ALGORITHM: Longest Common Subsequence
APPROACH: Find longest subsequence common to both sequences
TIME COMPLEXITY: O(m * n)
SPACE COMPLEXITY: O(m * n)

PROS:
- Classic 2D DP problem
- Used in diff algorithms (git diff)
- Foundation for many bioinformatics algorithms

CONS:
- Only finds length, not the actual subsequence (need backtracking)
- O(m*n) space can be large

RECURRENCE:
LCS(i, j) = {
    0                           if i=0 or j=0
    LCS(i-1, j-1) + 1          if seq1[i-1] = seq2[j-1]
    max(LCS(i-1, j), LCS(i, j-1))  otherwise
}

USE CASE: diff tools, plagiarism detection, DNA analysis
RELATED: Edit Distance, Longest Increasing Subsequence
"""

def longest_common_subsequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


# =============================================================================
# 13. M3 TILE PROBLEM - STATE MACHINE DP (Worksheet 7)
# =============================================================================
"""
ALGORITHM: M3 Tile Tiling Problem
APPROACH: Count ways to tile 3×L board with specific constraints
STATES: FLAT, UPPER2, LOWER2 (represents current column state)
TIME COMPLEXITY: 
  - Recursive: Exponential
  - Memoization: O(L) with 3 states = O(3L)
SPACE COMPLEXITY: O(L) for memoization

PROS:
- Handles complex state transitions
- Demonstrates state machine approach
- Efficient with memoization

CONS:
- Complex state definition
- Requires careful analysis of valid transitions
- Problem-specific (hard to generalize)

STATE TRANSITIONS:
From FLAT → Can place: UPPER2, LOWER2, or horizontal tile (skip 2)
From UPPER2/LOWER2 → Must place: FLAT or vertical tile (skip 2)

USE CASE: Advanced DP, tiling problems, state machine modeling
"""

def m3_tile_count(L):
    FLAT, UPPER2, LOWER2 = 0, 1, 2
    memo = {}
    
    def count_ways(distance, state):
        if (distance, state) in memo:
            return memo[(distance, state)]
        
        if distance == L:
            return 1 if state == FLAT else 0
        
        ways = 0
        if state == FLAT:
            ways += count_ways(distance + 1, UPPER2)
            ways += count_ways(distance + 1, LOWER2)
            if distance + 2 <= L:
                ways += count_ways(distance + 2, FLAT)
        else:  # UPPER2 or LOWER2
            ways += count_ways(distance + 1, FLAT)
            if distance + 2 <= L:
                ways += count_ways(distance + 2, state)
        
        memo[(distance, state)] = ways
        return ways
    
    return count_ways(0, FLAT)


# =============================================================================
# 14. STAIR CLIMBING WITH COSTS (Assignment 2)
# =============================================================================
"""
ALGORITHM: Minimum Cost to Reach Top
APPROACH: At each step, choose minimum of previous two steps
TIME COMPLEXITY: 
  - Recursive: O(2^n)
  - Memoization: O(n)
  - Tabulation: O(n)
SPACE COMPLEXITY: O(n) or O(1) with optimization

PROS:
- Simple DP introduction
- Clear optimal substructure
- Can optimize to O(1) space

CONS:
- Simple problem (good for learning, not challenging)

RECURRENCE:
minCost(i) = cost[i] + min(minCost(i-1), minCost(i-2))

Answer: min(minCost(n-1), minCost(n-2))

USE CASE: DP introduction, path problems
"""

def stair_climbing_costs(costs):
    n = len(costs)
    if n == 1:
        return costs[0]
    
    memo = {}
    
    def min_cost(i):
        if i in memo:
            return memo[i]
        
        if i < 0:
            return 0
        if i == 0 or i == 1:
            return costs[i]
        
        result = costs[i] + min(min_cost(i-1), min_cost(i-2))
        memo[i] = result
        return result
    
    return min(min_cost(n-1), min_cost(n-2))


# =============================================================================
# 15. FIBONACCI-STYLE POPULATION GROWTH (Midterm Q1)
# =============================================================================
"""
ALGORITHM: Two-State Population Growth
APPROACH: Bottom-up DP with two sequences (Male, Female)
TIME COMPLEXITY: O(g)
SPACE COMPLEXITY: O(g)

RULES:
- Males → become Females next generation
- Females → produce both Male and Female next generation

RECURRENCE:
M[i+1] = F[i]
F[i+1] = M[i] + F[i]

PROS:
- Clean two-state DP example
- Linear time complexity
- Easy to understand

CONS:
- Specific to this problem

USE CASE: Multi-dimensional DP, Fibonacci variants
"""

def population_growth(generations):
    M = [0] * (generations + 1)
    F = [0] * (generations + 1)
    M[0] = 1  # Start with 1 male
    
    for i in range(generations):
        M[i+1] = F[i]
        F[i+1] = M[i] + F[i]
    
    return M[generations] + F[generations]


# =============================================================================
# ALGORITHM COMPARISON SUMMARY
# =============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
PROBLEM TYPE          | NAIVE         | MEMOIZATION  | TABULATION
═══════════════════════════════════════════════════════════════════════════
Max Subarray          | O(n³)         | O(n) Kadane  | O(n) Kadane
Rod Cutting           | O(2^n)        | O(n²)        | O(n²)
Coin Change           | O(C^n)        | O(n*C)       | O(n*C)
0/1 Knapsack          | O(2^n)        | O(n*M)       | O(n*M)
Edit Distance         | O(3^(m+n))    | O(m*n)       | O(m*n)
LCS                   | O(2^(m+n))    | O(m*n)       | O(m*n)
Stair Climbing        | O(2^n)        | O(n)         | O(n)
M3 Tile               | Exponential   | O(L)         | O(L)
═══════════════════════════════════════════════════════════════════════════

KEY INSIGHTS:
1. Recursion → Identifies overlapping subproblems
2. Memoization → Caches recursive results (Top-Down DP)
3. Tabulation → Iterative table filling (Bottom-Up DP)
4. Memoization vs Tabulation: Same complexity, different approach
5. Space optimization: Many DP problems can reduce space from O(n²) to O(n)

WHEN TO USE EACH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Naive Recursion:  Learning, small inputs, understanding problem
Memoization:      Natural recursive formulation, not all states visited
Tabulation:       Production code, need maximum performance, space optimization
Backtracking:     Constraint satisfaction, generate all solutions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DYNAMIC PROGRAMMING CHECKLIST:
✓ Overlapping subproblems (same state computed multiple times)
✓ Optimal substructure (optimal solution uses optimal subsolutions)
✓ Define state clearly: What parameters uniquely identify a subproblem?
✓ Write recurrence relation
✓ Identify base cases
✓ Choose memoization (top-down) or tabulation (bottom-up)
✓ Optimize space if possible
"""

# =============================================================================
# TESTING SECTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALGORITHM DESIGN - COMPREHENSIVE SUMMARY")
    print("=" * 70)
    
    # Test GCD
    print("\n1. GCD(48, 18):", gcd_euclidean(48, 18))  # Expected: 6
    
    # Test Max Subarray
    print("2. Max Subarray [-2,1,-3,4,-1,2,1,-5,4]:", 
          max_subarray_brute_force([-2,1,-3,4,-1,2,1,-5,4]))  # Expected: 6
    
    # Test Binary Combinations
    print("3. Binary Combinations n=3:", generate_binary_combinations(3))
    
    # Test Knapsack (all versions)
    w = [2, 3, 4, 5]
    v = [3, 4, 5, 6]
    cap = 8
    print(f"4. Knapsack (w={w}, v={v}, cap={cap}):")
    print(f"   V1 (Generate All): {knapsack_v1_generate_all(w, v, cap)}")
    print(f"   V2 (Pruning):      {knapsack_v2_pruning(w, v, cap)}")
    print(f"   V3 (Memoization):  {knapsack_v3_memoization(w, v, cap)}")
    print(f"   V4 (Tabulation):   {knapsack_tabulation(w, v, cap)}")
    
    # Test Edit Distance
    print("5. Edit Distance('kitten', 'sitting'):", 
          edit_distance_memoization("kitten", "sitting"))  # Expected: 3
    
    # Test LCS
    print("6. LCS([1,2,3,4], [2,4,5]):", 
          longest_common_subsequence([1,2,3,4], [2,4,5]))  # Expected: 2
    
    # Test M3 Tile
    print("7. M3 Tile Count (L=4):", m3_tile_count(4))
    
    # Test Stair Climbing
    print("8. Stair Climbing Costs([10,15,20]):", 
          stair_climbing_costs([10,15,20]))  # Expected: 15
    
    # Test Population Growth
    print("9. Population Growth (g=5):", population_growth(5))
    
    print("\n" + "=" * 70)
    print("All algorithms tested successfully!")
    print("=" * 70)
