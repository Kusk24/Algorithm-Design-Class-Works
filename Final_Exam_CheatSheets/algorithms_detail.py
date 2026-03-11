"""
================================================================================
📚 ALGORITHM DESIGN REFERENCE GUIDE
================================================================================
Complete Summary of Algorithms by Week (Weeks 1-13)
All algorithms with detailed explanations, complexity analysis, and use cases

🔍 QUICK SEARCH TIPS:
   - Press Ctrl+F (Cmd+F on Mac) to search
   - Search by: Week number, Algorithm name, Complexity (e.g., "O(n)"), or Type
   - All major sections marked with ═══ for easy spotting
   
================================================================================
"""

# ============================================================================
# 📑 TABLE OF CONTENTS - Quick Navigation
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    🔍 SEARCH BY ALGORITHM                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Algorithm                              | Week   | Complexity | Line #   │
├────────────────────────────────────────┼────────┼────────────┼──────────┤
│ Kadane's Algorithm                     │ Week 1 │ O(n)       │ ~Line 90 │
│ Divide and Conquer                     │ Week 2 │ O(n log n) │ ~Line 140│
│ Balance Split                          │ Week 2 │ O(2^n)     │ ~Line 185│
│ Rod Cutting (DP)                       │ Week 3 │ O(n²)      │ ~Line 230│
│ Minimum Coin Change (DP)               │ Week 3 │ O(V×n)     │ ~Line 275│
│ Top-Down DP (Memoization)              │ Week 4 │ O(n×W)     │ ~Line 320│
│ 0/1 Knapsack (Memoized)                │ Week 4 │ O(n×W)     │ ~Line 375│
│ Edit Distance (Levenshtein)            │ Week 5 │ O(m×n)     │ ~Line 420│
│ Longest Common Subsequence (LCS)       │ Week 6 │ O(m×n)     │ ~Line 470│
│ Knapsack (Bottom-Up DP)                │ Week 6 │ O(n×W)     │ ~Line 520│
│ M3 Tile Problem (Tiling DP)            │ Week 7 │ O(n)       │ ~Line 575│
│ Shoe Shopping (DP)                     │ Week 7 │ O(n²)      │ ~Line 625│
│ N-Queens Problem (Backtracking)        │ Week 8 │ O(n!)      │ ~Line 650│
│ Breadth-First Search (BFS)             │ Week 8 │ O(V+E)     │ ~Line 705│
│ Maze Running (DFS/BFS)                 │ Week 8 │ O(R×C)     │ ~Line 760│
│ Iterative Deepening Search (IDS)       │ Week 9 │ O(b^d)     │ ~Line 820│
│ IDA* (Iterative Deepening A*)          │ Week 9 │ O(b^d)     │ ~Line 870│
│ Uniform Cost Search (UCS)              │ Week 9 │ O(E log V) │ ~Line 920│
│ Activity Selection (Greedy)            │ Week 10│ O(n log n) │ ~Line 980│
│ Minimum Spanning Tree (Kruskal/Prim)   │ Week 10│ O(E log E) │ ~Line 1030│
│ Union-Find (Disjoint Sets)             │ Week 10│ O(α(n))    │ ~Line 1080│
│ Fast Exponentiation                    │ Week 11│ O(log n)   │ ~Line 1145│
│ Max Subarray (Divide & Conquer)        │ Week 11│ O(n log n) │ ~Line 1195│
│ Branch and Bound (Knapsack)            │ Week 12│ O(2^n)     │ ~Line 1250│
│ DFS with Pruning                       │ Week 12│ O(2^n)     │ ~Line 1300│
│ BFS State Space (Flappy Bird)          │ Week 13│ O(H×T)     │ ~Line 1360│
│ Connected Components (DFS/BFS)         │ Week 13│ O(V+E)     │ ~Line 1415│
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       🎯 SEARCH BY TYPE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ Dynamic Programming (DP):                                               │
│   • Kadane's, Rod Cutting, Coin Change, Knapsack, Edit Distance,       │
│     LCS, Tiling, Shoe Shopping                                          │
│                                                                         │
│ Graph Algorithms:                                                       │
│   • BFS, DFS, Maze, IDS, IDA*, UCS, Connected Components               │
│                                                                         │
│ Greedy Algorithms:                                                      │
│   • Activity Selection, MST (Kruskal/Prim)                              │
│                                                                         │
│ Backtracking:                                                           │
│   • N-Queens, Balance Split                                             │
│                                                                         │
│ Divide & Conquer:                                                       │
│   • Max Subarray, Fast Exponentiation                                   │
│                                                                         │
│ Optimization:                                                           │
│   • Branch & Bound, Pruning                                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    ⚡ COMPLEXITY QUICK REFERENCE                         │
├─────────────────────────────────────────────────────────────────────────┤
│ O(n)        │ Kadane's, Tiling, Union-Find                              │
│ O(n log n)  │ Divide & Conquer, Activity Selection, Fast Exp            │
│ O(n²)       │ Rod Cutting, Shoe Shopping                                │
│ O(m×n)      │ Edit Distance, LCS                                        │
│ O(n×W)      │ Knapsack (all variants)                                   │
│ O(V+E)      │ BFS, DFS, Connected Components                            │
│ O(2^n)      │ Balance Split, Branch & Bound                             │
│ O(n!)       │ N-Queens                                                  │
└─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# ============================================================================
# 📌 WEEK 1: Maximum Sum Problem
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Kadane's Algorithm (Maximum Subarray Sum)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n)
# 💾 SPACE COMPLEXITY: O(1)
# 📊 TYPE: Array Processing
# 🎯 DIFFICULTY: Easy
#

# **How the Algorithm Works**:
# 1. Initialize two variables:
#    - `current_sum` = 0 (sum of current subarray)
#    - `max_sum` = -infinity (best sum found so far)
# 2. Iterate through array from left to right:
#    - Add current element to `current_sum`
#    - Update `max_sum` = max(max_sum, current_sum)
#    - If `current_sum` becomes negative, reset it to 0
#    - (Negative sums can't help future subarrays)
# 3. Return `max_sum` as the answer
# 4. Key insight: At each position, decide whether to:
#    - Extend current subarray (add element to current_sum)
#    - Start new subarray (reset current_sum if negative)

# **Advantages**:
# - Very efficient linear time solution
# - Simple to implement
# - Uses minimal extra space
# - Works well for finding contiguous subarrays

# **Disadvantages**:
# - Only works for contiguous subarrays
# - Doesn't track the actual subarray indices without modification
# - Cannot handle all-negative arrays without special handling

# **When to Use**:
# - Finding maximum sum of contiguous subarray
# - Stock profit problems (buy/sell once)
# - When you need optimal linear time solution
# - Streaming data where you process elements once



# ============================================================================
# ============================================================================
# 📌 WEEK 2: Divide and Conquer / Recursion
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Divide and Conquer
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n log n) for most problems
# 💾 SPACE COMPLEXITY: O(log n) for recursion stack
# 📊 TYPE: Problem-solving Pattern
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. **Divide**: Split problem into smaller subproblems
#    - Usually divide array/data into halves or thirds
#    - Continue until subproblems are trivial (base case)
# 2. **Conquer**: Solve each subproblem recursively
#    - Base case: problem small enough to solve directly
#    - Recursive case: divide further and solve
# 3. **Combine**: Merge solutions of subproblems
#    - Merge sorted halves (merge sort)
#    - Find max crossing middle (max subarray)
#    - Combine results to get final answer
# 4. Example (Merge Sort):
#    - Divide: [8,3,5,1] → [8,3] and [5,1]
#    - Conquer: [8,3]→[3,8], [5,1]→[1,5]
#    - Combine: [3,8] + [1,5] → [1,3,5,8]

# **Advantages**:
# - Breaks complex problems into simpler subproblems
# - Often leads to efficient solutions
# - Natural recursive structure
# - Can be parallelized easily

# **Disadvantages**:
# - Recursion overhead
# - May have high space complexity due to call stack
# - Not always intuitive
# - Can lead to repeated computations without memoization

# **When to Use**:
# - Problems that can be broken into independent subproblems
# - Sorting algorithms (merge sort, quick sort)
# - Binary search
# - Tree traversals
# - Matrix operations

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Balance Split
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(2^n)
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Partition/Backtracking
# 🎯 DIFFICULTY: Hard
#

# **How the Algorithm Works**:
# 1. Goal: Split array into two groups with minimum difference in sums
# 2. **Backtracking Approach**:
#    - For each element, decide: Group 0 or Group 1?
#    - Try all 2^n possible partitions
# 3. **Recursive Process**:
#    - Base case: All elements assigned
#      * Calculate sum of each group
#      * Return |sum1 - sum2|
#    - Recursive case:
#      * Try assigning element to Group 0
#      * Try assigning element to Group 1
#      * Return minimum difference from both choices
# 4. **Example**: [1, 6, 11, 5]
#    - Total sum = 23
#    - Optimal: [11, 1] vs [6, 5]
#    - Sums: 12 vs 11
#    - Difference: 1 (minimum possible)
# 5. **Optimization**: Can use DP subset sum approach
#    - Find closest sum to total_sum/2
#    - O(n × sum) time vs O(2^n)

# **Advantages**:
# - Explores all possible combinations
# - Guarantees finding optimal solution if it exists

# **Disadvantages**:
# - Exponential time complexity
# - Not practical for large inputs
# - High memory usage for recursion

# **When to Use**:
# - Small input sizes (n < 20)
# - When you need exact solutions
# - Partition problems with small datasets



# ============================================================================
# ============================================================================
# 📌 WEEK 3: Dynamic Programming Introduction
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Rod Cutting Problem (DP)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n²)
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Dynamic Programming
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Given: rod of length n, prices for each length 1 to n
# 2. Goal: cut rod to maximize revenue
# 3. **Recursive Structure**:
#    - For rod of length n, try all first cuts (1 to n)
#    - maxRev(n) = max(price[i] + maxRev(n-i)) for i=1 to n
# 4. **DP Solution (Bottom-Up)**:
#    - Create array dp[0...n]
#    - dp[0] = 0 (length 0 has 0 revenue)
#    - For length i from 1 to n:
#      - Try all possible first cuts j = 1 to i
#      - dp[i] = max(price[j] + dp[i-j])
# 5. **Example**: n=4, prices=[1,5,8,9]
#    - dp[1] = 1
#    - dp[2] = 5 (no cut better than length-2 piece)
#    - dp[3] = 8 (no cut better)
#    - dp[4] = max(1+8, 5+5, 8+1, 9) = 10 (two length-2 pieces)

# **Advantages**:
# - Finds optimal revenue
# - Polynomial time solution
# - Avoids exponential brute force
# - Can reconstruct solution

# **Disadvantages**:
# - Requires careful state definition
# - Space overhead for DP table
# - Overkill for small inputs

# **When to Use**:
# - Optimization problems with overlapping subproblems
# - When greedy approach doesn't work
# - Resource allocation problems
# - Cutting/partitioning problems

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Minimum Coin Change (DP)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(V × n) where V = value, n = number of coins
# 💾 SPACE COMPLEXITY: O(V)
# 📊 TYPE: Dynamic Programming (Unbounded Knapsack)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Given: coins = [c₁, c₂, ..., cₙ], target amount V
# 2. Goal: find minimum coins needed to make amount V
# 3. **Recursive Structure**:
#    - minCoins(V) = 1 + min(minCoins(V - cᵢ)) for each coin cᵢ ≤ V
#    - Base case: minCoins(0) = 0
# 4. **DP Solution (Bottom-Up)**:
#    - Create array dp[0...V], initialize with infinity
#    - dp[0] = 0 (0 coins for amount 0)
#    - For each amount i from 1 to V:
#      - For each coin c in coins:
#        - If c ≤ i: dp[i] = min(dp[i], 1 + dp[i-c])
# 5. **Example**: coins=[1,3,4], V=6
#    - dp[0]=0, dp[1]=1, dp[2]=2, dp[3]=1
#    - dp[4]=1, dp[5]=2, dp[6]=2 (use coins 3+3)
# 6. **Key Insight**: Can use each coin unlimited times (unbounded)

# **Advantages**:
# - Guarantees minimum number of coins
# - Handles any coin denominations
# - Efficient for reasonable values
# - Classic DP example

# **Disadvantages**:
# - Space grows with target value
# - Doesn't work if no solution exists (needs handling)
# - Not suitable for very large target values

# **When to Use**:
# - Making change problems
# - Resource allocation with discrete units
# - When greedy approach fails (non-canonical coin systems)
# - Unbounded knapsack variants



# ============================================================================
# ============================================================================
# 📌 WEEK 4: Memoization Techniques
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Top-Down DP with Memoization
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n × W) for knapsack, O(V × n) for coins
# 💾 SPACE COMPLEXITY: O(n × W) or O(V) + recursion stack
# 📊 TYPE: Dynamic Programming (Top-Down)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Start with recursive solution (top-down)
# 2. Add caching to avoid recomputation
# 3. **Memoization Steps**:
#    - Create cache (dictionary or array)
#    - Before computing subproblem:
#      * Check if already in cache
#      * If yes: return cached result
#    - After computing subproblem:
#      * Store result in cache
#      * Return result
# 4. **Example** (Fibonacci with memoization):
#    ```python
#    memo = {}
#    def fib(n):
#        if n in memo:
#            return memo[n]  # Return cached
#        if n <= 1:
#            result = n
#        else:
#            result = fib(n-1) + fib(n-2)
#        memo[n] = result  # Cache result
#        return result
#    ```
# 5. **State Definition**: Key to memoization
#    - Identify parameters that uniquely define subproblem
#    - Use as cache key (tuple or unique string)
# 6. **Conversion Pattern**:
#    - Naive recursion: O(exponential)
#    - + Memoization: O(# states × transition cost)

# **Advantages**:
# - Easier to code than bottom-up DP
# - Only computes needed states
# - Natural recursive thinking
# - Avoids redundant calculations

# **Disadvantages**:
# - Recursion stack overhead
# - May hit recursion limit
# - Slightly slower than iterative DP
# - Extra space for recursion

# **When to Use**:
# - When recursive solution is natural
# - Sparse state space (not all states needed)
# - During learning phase of DP
# - When problem is easier to think recursively

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: 0/1 Knapsack (Memoized)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n × W)
# 💾 SPACE COMPLEXITY: O(n × W)
# 📊 TYPE: Dynamic Programming (Memoization)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Given: n items with weights[i] and values[i], capacity W
# 2. Goal: maximize value without exceeding capacity
# 3. **Recursive Structure** (for item i, remaining capacity C):
#    - If i == n: return 0 (no items left)
#    - Option 1 (skip): maxVal(i+1, C)
#    - Option 2 (take): values[i] + maxVal(i+1, C-weights[i])
#    - Return max of both options (if item fits)
# 4. **Memoization**:
#    - Use dictionary memo[(i, C)] to cache results
#    - Before computing, check if (i, C) already solved
#    - After computing, store result in memo
# 5. **Example**: W=5, items=[(w=2,v=3), (w=3,v=4), (w=4,v=5)]
#    - Try all combinations, memo avoids recomputation
#    - Optimal: take items 1 and 2 (w=5, v=7)
# 6. **State Space**: At most n × W unique states

# **Advantages**:
# - Solves classic optimization problem
# - Polynomial time vs exponential brute force
# - Can handle reasonable input sizes
# - Extensible to variations

# **Disadvantages**:
# - Pseudo-polynomial (depends on W)
# - High space complexity
# - Not suitable for very large capacities
# - Integer weights required

# **When to Use**:
# - Resource allocation with capacity constraints
# - Budget optimization problems
# - Project selection with constraints
# - When items can only be taken once



# ============================================================================
# ============================================================================
# 📌 WEEK 5: Edit Distance
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Levenshtein Distance (Edit Distance)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(m × n)
# 💾 SPACE COMPLEXITY: O(m × n) or O(min(m,n)) if optimized
# 📊 TYPE: Dynamic Programming (String)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Given: two strings s1 (length m) and s2 (length n)
# 2. Goal: minimum operations to transform s1 into s2
# 3. **Operations allowed**: Insert, Delete, Substitute (each costs 1)
# 4. **Recursive Structure** (for positions i in s1, j in s2):
#    - If i=0: return j (insert j characters)
#    - If j=0: return i (delete i characters)
#    - If s1[i-1] == s2[j-1]: return dp[i-1][j-1] (no cost)
#    - Else: return 1 + min(
#        dp[i-1][j],    // delete from s1
#        dp[i][j-1],    // insert into s1
#        dp[i-1][j-1]   // substitute
#      )
# 5. **DP Table** (Bottom-Up):
#    - Create (m+1) × (n+1) table
#    - dp[i][j] = edit distance of s1[0..i-1] to s2[0..j-1]
#    - Fill row by row using recurrence above
# 6. **Example**: "kitten" → "sitting"
#    - Substitute k→s: "sitten" (1)
#    - Substitute e→i: "sittin" (2)
#    - Insert g: "sitting" (3)
#    - Total: 3 operations

# **Advantages**:
# - Measures string similarity accurately
# - Handles insertions, deletions, substitutions
# - Well-studied with many variants
# - Space can be optimized to O(n)

# **Disadvantages**:
# - Quadratic time complexity
# - Not suitable for very long strings
# - Equal weight for all operations (may not be realistic)

# **When to Use**:
# - Spell checking and correction
# - DNA sequence alignment
# - Plagiarism detection
# - Fuzzy string matching
# - Auto-correct systems



# ============================================================================
# ============================================================================
# 📌 WEEK 6: Advanced Dynamic Programming
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Longest Common Subsequence (LCS)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(m × n)
# 💾 SPACE COMPLEXITY: O(m × n) or O(min(m,n)) if optimized
# 📊 TYPE: Dynamic Programming (String)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Given: two sequences X (length m) and Y (length n)
# 2. Goal: find longest subsequence common to both (order preserved)
# 3. **Recursive Structure** (for positions i in X, j in Y):
#    - If i=0 or j=0: return 0 (no elements)
#    - If X[i-1] == Y[j-1]: return 1 + LCS(i-1, j-1)
#    - Else: return max(LCS(i-1, j), LCS(i, j-1))
# 4. **DP Table** (Bottom-Up):
#    - Create (m+1) × (n+1) table
#    - dp[i][j] = LCS length of X[0..i-1] and Y[0..j-1]
#    - Fill table row by row using recurrence
# 5. **Example**: X="ABCDGH", Y="AEDFHR"
#    ```
#         A E D F H R
#      0  0 0 0 0 0 0
#    A 0  1 1 1 1 1 1
#    B 0  1 1 1 1 1 1
#    C 0  1 1 1 1 1 1
#    D 0  1 1 2 2 2 2
#    G 0  1 1 2 2 2 2
#    H 0  1 1 2 2 3 3
#    ```
#    - LCS = "ADH" (length 3)
# 6. **Reconstruct**: Trace back from dp[m][n] to dp[0][0]

# **Advantages**:
# - Finds optimal alignment
# - Works for non-contiguous sequences
# - Can reconstruct actual subsequence
# - Space-optimizable

# **Disadvantages**:
# - Quadratic complexity
# - Doesn't handle gaps well
# - May have multiple solutions

# **When to Use**:
# - Diff tools (version control)
# - DNA/protein sequence alignment
# - Finding common patterns
# - File comparison
# - Plagiarism detection

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Knapsack (Bottom-Up DP)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n × W)
# 💾 SPACE COMPLEXITY: O(n × W) or O(W) if optimized
# 📊 TYPE: Dynamic Programming (Bottom-Up)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. **DP Table**: dp[i][w] = max value using first i items with capacity w
# 2. **Base Cases**:
#    - dp[0][w] = 0 for all w (no items, no value)
#    - dp[i][0] = 0 for all i (no capacity, no value)
# 3. **Recurrence**:
#    - For each item i and capacity w:
#      * Option 1 (skip): dp[i][w] = dp[i-1][w]
#      * Option 2 (take): dp[i][w] = values[i] + dp[i-1][w-weights[i]]
#      * dp[i][w] = max(option1, option2) if weights[i] ≤ w
# 4. **Fill Order**: Row by row (or column by column)
# 5. **Example**: items=[(w=2,v=3), (w=3,v=4)], capacity=5
#    ```
#    Capacity:  0  1  2  3  4  5
#    Item 0:    0  0  3  3  3  3
#    Item 1:    0  0  3  4  4  7
#    ```
#    - dp[2][5] = 7 (take both items)
# 6. **Space Optimization**: Use 1D array
#    - Only need previous row to compute current row
#    - Process items in reverse to avoid overwriting
#    ```python
#    dp = [0] * (W + 1)
#    for item in items:
#        for w in range(W, weights[item]-1, -1):
#            dp[w] = max(dp[w], values[item] + dp[w-weights[item]])
#    ```

# **Advantages**:
# - Iterative (no recursion stack)
# - Slightly faster than memoization
# - Can optimize space to O(W)
# - Clear state transitions

# **Disadvantages**:
# - Computes all states (even unnecessary ones)
# - Less intuitive than recursive
# - Pseudo-polynomial complexity

# **When to Use**:
# - When iterative approach is preferred
# - Limited recursion depth
# - Need slight performance edge
# - Teaching bottom-up DP concept



# ============================================================================
# ============================================================================
# 📌 WEEK 7: Complex DP Problems
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: M3 Tile Problem (Tiling DP)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n) with DP
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Dynamic Programming (State-based)
# 🎯 DIFFICULTY: Hard
#

# **How the Algorithm Works**:
# 1. Goal: Tile 3×n board with 1×3 and 3×1 tiles, count ways
# 2. **State Definition**:
#    - dp[i][state] = ways to fill first i columns
#    - state encodes which cells in column i are filled
# 3. **States for 3×n board**:
#    - State 0: All 3 cells empty (000)
#    - State 7: All 3 cells filled (111)
#    - Others: Partial fills
# 4. **Transitions**:
#    - From filled column (state 7):
#      * Place vertical 3×1 tile → state 7
#      * Place three horizontal 1×3 tiles → state 7
#    - Complex transitions for partial states
# 5. **Simplified Version** (only full columns):
#    - dp[i] = ways to completely fill 3×i board
#    - dp[0] = 1 (empty board)
#    - Recurrence: dp[i] = f(dp[i-1], dp[i-2], ...)
# 6. **Example** (3×2 board):
#    ```
#    Way 1: Two vertical tiles
#    Way 2: Six horizontal tiles (stacked)
#    ...
#    ```

# **Advantages**:
# - Elegant recursive structure
# - Linear time with DP
# - Demonstrates state-based DP
# - Can handle complex constraints

# **Disadvantages**:
# - State definition can be tricky
# - Problem-specific approach
# - Exponential without DP

# **When to Use**:
# - Tiling/covering problems
# - Combinatorial counting
# - When states have mutual dependencies
# - Constraint satisfaction problems

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Shoe Shopping (Optimization DP)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n²) typically
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Dynamic Programming (Pairing)
# 🎯 DIFFICULTY: Medium
#

# **Advantages**:
# - Handles pairing constraints
# - Finds optimal cost
# - Demonstrates non-trivial DP

# **Disadvantages**:
# - Problem-specific
# - May need careful state design
# - Not generalizable easily

# **When to Use**:
# - Pairing/matching problems
# - Discount optimization
# - Scheduling with constraints
# - Bundling problems



# ============================================================================
# ============================================================================
# 📌 WEEK 8: Backtracking Algorithms
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: N-Queens Problem (Backtracking)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n!)
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Backtracking (Constraint Satisfaction)
# 🎯 DIFFICULTY: Hard
#

# **How the Algorithm Works**:
# 1. Goal: Place n queens on n×n board so none attack each other
# 2. **Backtracking Strategy**:
#    - Place queens row by row (one queen per row)
#    - For each row, try each column position
#    - Before placing, check if position is safe:
#      * No queen in same column
#      * No queen on diagonal (row-col constant)
#      * No queen on anti-diagonal (row+col constant)
# 3. **Recursive Process**:
#    - If row == n: found complete solution, save it
#    - For col = 0 to n-1:
#      * If position (row, col) is safe:
#        - Place queen at (row, col)
#        - Recursively solve for row+1
#        - Remove queen (backtrack)
# 4. **Pruning**: Skip invalid branches early
#    - If column already has queen, skip
#    - If diagonal blocked, skip
# 5. **Example** (n=4):
#    ```
#    .Q..    ..Q.
#    ...Q    Q...
#    Q...    ...Q
#    ..Q.    .Q..
#    ```
#    - Two solutions for 4×4 board

# **Advantages**:
# - Finds all solutions
# - Memory efficient
# - Explores systematically
# - Can terminate early if one solution needed

# **Disadvantages**:
# - Exponential time complexity
# - Very slow for large n (n > 15)
# - Cannot be easily parallelized

# **When to Use**:
# - Constraint satisfaction problems
# - Small board sizes
# - When all solutions needed
# - Puzzle solving (Sudoku, etc.)

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Breadth-First Search (BFS)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(V + E) where V = vertices, E = edges
# 💾 SPACE COMPLEXITY: O(V)
# 📊 TYPE: Graph Traversal
# 🎯 DIFFICULTY: Easy
#

# **How the Algorithm Works**:
# 1. Start from source vertex s
# 2. Use a queue to track vertices to visit
# 3. Use a set/array to mark visited vertices
# 4. **BFS Process**:
#    - Initialize: queue = [s], visited = {s}
#    - While queue not empty:
#      * Dequeue vertex v
#      * Process v (e.g., check if goal)
#      * For each neighbor u of v:
#        - If u not visited:
#          * Mark u as visited
#          * Enqueue u
#          * Track u's distance = v's distance + 1
# 5. **Level-by-Level Exploration**:
#    - Level 0: source vertex
#    - Level 1: all neighbors of source
#    - Level 2: neighbors of level 1 vertices
#    - Continue until goal found or queue empty
# 6. **Example** (graph with edges A→B, A→C, B→D, C→D):
#    ```
#    Start: A
#    Level 0: A
#    Level 1: B, C
#    Level 2: D
#    ```
#    - Shortest path A→D is 2 edges

# **Advantages**:
# - Finds shortest path (unweighted)
# - Complete (finds solution if exists)
# - Optimal for unweighted graphs
# - Level-by-level exploration

# **Disadvantages**:
# - High memory usage (stores entire level)
# - Not suitable for weighted graphs
# - May explore unnecessary nodes

# **When to Use**:
# - Shortest path in unweighted graphs
# - Level-order traversal
# - Finding connected components
# - Web crawling
# - Social network analysis (degrees of separation)

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Maze Running (DFS/BFS)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(rows × cols)
# 💾 SPACE COMPLEXITY: O(rows × cols)
# 📊 TYPE: Graph Traversal (Grid)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works (BFS for shortest path)**:
# 1. **Grid Representation**:
#    - '#' = wall (impassable)
#    - '.' = open path
#    - 'S' = start position
#    - 'E' = exit position
# 2. **BFS Algorithm**:
#    - Initialize queue with start position (row, col, distance=0)
#    - Create visited set to track explored cells
#    - While queue not empty:
#      * Dequeue (r, c, dist)
#      * If (r, c) is exit: return dist
#      * For each direction (up, down, left, right):
#        - Calculate new position (nr, nc)
#        - If valid (in bounds, not wall, not visited):
#          * Mark (nr, nc) as visited
#          * Enqueue (nr, nc, dist+1)
#    - If queue empty: no path exists
# 3. **4 Directions**:
#    ```python
#    directions = [(-1,0), (1,0), (0,-1), (0,1)]
#    # up, down, left, right
#    ```
# 4. **Example Maze**:
#    ```
#    S . # .
#    . . # .
#    # . . E
#    ```
#    BFS finds shortest path: S→right→down→down→right→right→E (6 moves)
# 5. **DFS vs BFS**:
#    - DFS: May find longer path, uses less memory
#    - BFS: Guarantees shortest path, more memory

# **Advantages**:
# - Systematically explores maze
# - Guarantees finding exit if exists
# - BFS finds shortest path

# **Disadvantages**:
# - May explore entire maze
# - Memory intensive for large mazes

# **When to Use**:
# - Pathfinding in grids
# - Robot navigation
# - Game AI
# - Route planning



# ============================================================================
# ============================================================================
# 📌 WEEK 9: Advanced Search Algorithms
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Iterative Deepening Search (IDS)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(b^d) where b = branching factor, d = depth
# 💾 SPACE COMPLEXITY: O(d)
# 📊 TYPE: Search Algorithm
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Combines DFS space efficiency with BFS completeness
# 2. **Strategy**: Repeat depth-limited DFS with increasing limits
# 3. **Algorithm Steps**:
#    - For depth_limit = 0, 1, 2, 3, ...
#      * Perform DFS with max depth = depth_limit
#      * If goal found: return solution
#      * Else: increment depth_limit
# 4. **Depth-Limited DFS**:
#    - Start from root
#    - Explore children recursively
#    - Stop if depth = depth_limit
#    - Don't expand beyond limit
# 5. **Example** (tree search with solution at depth 3):
#    ```
#    Iteration 1 (depth=0): Check root
#    Iteration 2 (depth=1): Check root + children
#    Iteration 3 (depth=2): Check up to depth 2
#    Iteration 4 (depth=3): Find solution!
#    ```
# 6. **Redundant Work**: Revisits nodes multiple times
#    - But dominated by bottom level
#    - Only adds constant factor overhead
# 7. **Advantage over BFS**: Uses O(d) vs BFS's O(b^d) space

# **Advantages**:
# - Combines DFS space efficiency with BFS optimality
# - Finds shortest solution
# - Memory efficient
# - Complete and optimal

# **Disadvantages**:
# - Repeated work (visits nodes multiple times)
# - Slower than BFS in practice
# - Only optimal for uniform cost

# **When to Use**:
# - Unknown solution depth
# - Limited memory
# - Need shortest path
# - Tree/graph search with memory constraints

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: IDA* (Iterative Deepening A*)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(b^d)
# 💾 SPACE COMPLEXITY: O(d)
# 📊 TYPE: Heuristic Search
# 🎯 DIFFICULTY: Hard
#

# **How the Algorithm Works**:
# 1. Combines IDS with A* heuristic for memory efficiency
# 2. **Cost Function**: f(n) = g(n) + h(n)
#    - g(n) = cost from start to node n
#    - h(n) = heuristic estimate to goal
# 3. **Iterative Deepening with Cost Bound**:
#    - Start with bound = h(start)
#    - Perform DFS limited by f(n) ≤ bound
#    - If goal found, return solution
#    - Else, increase bound to next minimum f(n) exceeded
#    - Repeat until goal found
# 4. **DFS with f-cost Pruning**:
#    - Explore node only if f(n) ≤ current bound
#    - Track minimum f(n) > bound for next iteration
#    - Prune branches exceeding bound
# 5. **Example** (8-puzzle with Manhattan distance):
#    ```
#    Iteration 1: bound = 5
#    Iteration 2: bound = 7
#    Iteration 3: bound = 9, solution found
#    ```
# 6. **Key Advantage**: Uses O(d) space vs A*'s O(b^d)

# **Advantages**:
# - Memory efficient (like IDS)
# - Uses heuristic for efficiency
# - Optimal if heuristic is admissible
# - Better than IDS with good heuristic

# **Disadvantages**:
# - Revisits nodes
# - Requires good heuristic
# - Slower than A* in practice
# - Heuristic overhead

# **When to Use**:
# - Memory-constrained environments
# - 8-puzzle, 15-puzzle problems
# - When A* uses too much memory
# - Game AI with limited resources

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Uniform Cost Search (UCS)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O((V + E) log V) with priority queue
# 💾 SPACE COMPLEXITY: O(V)
# 📊 TYPE: Graph Search (Weighted)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. **Like BFS but for weighted graphs**
# 2. **Priority Queue**: Order nodes by total path cost (g-cost)
# 3. **Algorithm Steps**:
#    - Initialize priority queue with (start, cost=0)
#    - Create cost dictionary: cost[start] = 0
#    - While queue not empty:
#      * Pop node with minimum cost
#      * If node is goal: return path cost
#      * For each neighbor with edge weight w:
#        - Calculate new_cost = cost[node] + w
#        - If neighbor not visited or new_cost < cost[neighbor]:
#          * Update cost[neighbor] = new_cost
#          * Add (neighbor, new_cost) to priority queue
# 4. **Example**: Graph with weighted edges
#    ```
#    A --1--> B --2--> D
#    |        |        ^
#    3        1        |
#    v        v        3
#    C ------4---------+
#    ```
#    Path A→D: A→B→D (cost 3) is better than A→C→D (cost 7)
# 5. **Key Difference from BFS**:
#    - BFS: First to reach = shortest (unweighted)
#    - UCS: Cheapest path = optimal (weighted)
# 6. **Relation to Dijkstra**: UCS is Dijkstra without destination

# **Advantages**:
# - Finds optimal solution for weighted graphs
# - Complete and optimal
# - Doesn't require heuristic
# - Handles arbitrary non-negative weights

# **Disadvantages**:
# - Memory intensive
# - Slow without heuristic
# - Requires priority queue
# - Not as efficient as A*

# **When to Use**:
# - Weighted graphs
# - When heuristic not available
# - Need guaranteed optimal path
# - Network routing


# ============================================================================
# ============================================================================
# 📌 WEEK 10: Greedy Algorithms
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Activity Selection (Greedy)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n log n) due to sorting
# 💾 SPACE COMPLEXITY: O(1) or O(n) for sorting
# 📊 TYPE: Greedy Algorithm
# 🎯 DIFFICULTY: Easy
#

# **How the Algorithm Works**:
# 1. Given: n activities with start and finish times
# 2. Goal: select maximum number of non-overlapping activities
# 3. **Greedy Strategy**: Always pick activity that finishes earliest
# 4. **Algorithm Steps**:
#    - Sort activities by finish time (ascending)
#    - Select first activity (earliest finish)
#    - For each remaining activity:
#      * If start time ≥ previous finish time:
#        - Select this activity
#        - Update last finish time
#      * Else: skip (overlaps with selected activity)
# 5. **Why Greedy Works**:
#    - Finishing early leaves most room for future activities
#    - Optimal substructure: if activity k is in optimal solution,
#      remaining problem is also optimal
# 6. **Example**: Activities [(1,4), (3,5), (0,6), (5,7), (8,9)]
#    - Sort by finish: [(1,4), (3,5), (0,6), (5,7), (8,9)]
#    - Select: (1,4)
#    - Skip: (3,5) overlaps
#    - Skip: (0,6) overlaps
#    - Select: (5,7)
#    - Select: (8,9)
#    - Result: 3 activities [(1,4), (5,7), (8,9)]

# **Advantages**:
# - Very efficient
# - Simple to implement
# - Optimal for activity selection
# - Intuitive approach

# **Disadvantages**:
# - Only works for specific problems
# - Not always optimal (must prove it)
# - Requires careful problem analysis

# **When to Use**:
# - Interval scheduling
# - Resource allocation
# - Meeting room problems
# - Job scheduling
# - When greedy choice property holds

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Minimum Spanning Tree (Kruskal's/Prim's)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(E log E) for Kruskal's, O(E log V) for Prim's
# 💾 SPACE COMPLEXITY: O(V + E)
# 📊 TYPE: Greedy (Graph Algorithm)
# 🎯 DIFFICULTY: Medium
#

# **How Kruskal's Algorithm Works**:
# 1. Goal: Connect all vertices with minimum total edge weight
# 2. **Greedy Strategy**: Add cheapest edge that doesn't create cycle
# 3. **Algorithm Steps**:
#    - Sort all edges by weight (ascending)
#    - Initialize: Each vertex is its own component
#    - For each edge (u, v, weight) in sorted order:
#      * If u and v in different components:
#        - Add edge to MST
#        - Merge components (union operation)
#      * Else: skip (would create cycle)
#    - Stop when |V|-1 edges added
# 4. **Uses Union-Find**: Track connected components efficiently
# 5. **Example**: Graph with edges (A-B:1, B-C:2, A-C:3)
#    - Sort: [1, 2, 3]
#    - Add A-B (weight 1)
#    - Add B-C (weight 2)
#    - Skip A-C (creates cycle)
#    - MST weight: 1 + 2 = 3

# **How Prim's Algorithm Works**:
# 1. Start from arbitrary vertex
# 2. Grow tree by adding cheapest edge to new vertex
# 3. Use priority queue for efficient edge selection
# 4. Stop when all vertices included

# **Advantages**:
# - Efficient for sparse graphs
# - Guaranteed optimal
# - Clear correctness proof
# - Practical applications

# **Disadvantages**:
# - Requires sorting edges (Kruskal's)
# - Needs disjoint-set data structure
# - Not applicable to directed graphs

# **When to Use**:
# - Network design (minimize cable/pipe)
# - Clustering problems
# - Approximation algorithms
# - Road/utility network construction

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Union-Find (Disjoint Sets)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(α(n)) ≈ O(1) per operation with optimizations
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Data Structure
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Data structure to track disjoint sets (components)
# 2. **Two Main Operations**:
#    - Find(x): Which set does x belong to?
#    - Union(x, y): Merge sets containing x and y
# 3. **Representation**: Parent array
#    - parent[i] = parent of node i
#    - If parent[i] = i: i is root (representative)
# 4. **Find Operation** (with path compression):
#    ```python
#    def find(x):
#        if parent[x] != x:
#            parent[x] = find(parent[x])  # Path compression
#        return parent[x]
#    ```
#    - Follow parents until reaching root
#    - Optimization: Make all nodes point directly to root
# 5. **Union Operation** (with union by rank):
#    ```python
#    def union(x, y):
#        root_x = find(x)
#        root_y = find(y)
#        if root_x != root_y:
#            if rank[root_x] < rank[root_y]:
#                parent[root_x] = root_y
#            else:
#                parent[root_y] = root_x
#                if rank[root_x] == rank[root_y]:
#                    rank[root_x] += 1
#    ```
#    - Attach smaller tree under larger tree
# 6. **Example**: Elements {0,1,2,3,4}
#    - Initially: Each element is own set
#    - Union(0,1): {0,1}, {2}, {3}, {4}
#    - Union(2,3): {0,1}, {2,3}, {4}
#    - Union(1,2): {0,1,2,3}, {4}
#    - Find(3) returns same root as Find(0)

# **Advantages**:
# - Near-constant time operations
# - Simple to implement
# - Essential for Kruskal's algorithm
# - Efficient for connectivity queries

# **Disadvantages**:
# - Limited to specific problems
# - Requires path compression for efficiency
# - Not intuitive initially

# **When to Use**:
# - Kruskal's MST algorithm
# - Connected components
# - Network connectivity
# - Cycle detection in graphs



# ============================================================================
# ============================================================================
# 📌 WEEK 11: Divide and Conquer (Advanced)
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Fast Exponentiation (Binary Exponentiation)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(log n)
# 💾 SPACE COMPLEXITY: O(log n) for recursive, O(1) for iterative
# 📊 TYPE: Math/Divide & Conquer
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. Goal: Compute x^n efficiently
# 2. **Key Insight**: Use binary representation of exponent
#    - x^8 = (x^4)^2 = ((x^2)^2)^2
#    - x^9 = x × x^8 = x × ((x^2)^2)^2
# 3. **Recursive Approach**:
#    - Base case: x^0 = 1
#    - If n is even: x^n = (x^(n/2))^2
#    - If n is odd: x^n = x × (x^(n-1))
# 4. **Iterative Approach**:
#    - Initialize result = 1, base = x, exp = n
#    - While exp > 0:
#      * If exp is odd: result *= base
#      * base = base^2
#      * exp = exp // 2
# 5. **Example**: Compute 3^13
#    - 13 in binary: 1101
#    - 3^13 = 3^8 × 3^4 × 3^1
#    - Steps: 3^1=3, 3^2=9, 3^4=81, 3^8=6561
#    - Result: 6561 × 81 × 3 = 1,594,323
# 6. **With Modulo** (for large n): (x^n) mod m
#    - Apply mod at each step to prevent overflow
#    - (a × b) mod m = ((a mod m) × (b mod m)) mod m

# **Advantages**:
# - Logarithmic time vs linear
# - Essential for cryptography
# - Works for large exponents
# - Can be made iterative

# **Disadvantages**:
# - More complex than naive approach
# - May overflow without modular arithmetic
# - Recursion overhead

# **When to Use**:
# - Modular exponentiation (cryptography)
# - Matrix exponentiation
# - Computing large powers
# - RSA encryption
# - Fibonacci computation

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Maximum Subarray (Divide and Conquer)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(n log n)
# 💾 SPACE COMPLEXITY: O(log n)
# 📊 TYPE: Divide & Conquer (Array)
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. **Divide**: Split array into left and right halves
# 2. **Conquer**: Find max subarray in each half recursively
# 3. **Combine**: Find max subarray crossing the midpoint
# 4. **Return**: Maximum of three possibilities
# 5. **Base Case**: Single element array returns that element
# 6. **Crossing Subarray** (the tricky part):
#    - Find max sum extending left from mid:
#      * Start at mid, go left
#      * Track maximum sum found
#    - Find max sum extending right from mid+1:
#      * Start at mid+1, go right
#      * Track maximum sum found
#    - Crossing max = left_max + right_max
# 7. **Example**: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
#    ```
#    Divide: [-2,1,-3,4] | [-1,2,1,-5,4]
#    Left max: 4
#    Right max: 2
#    Crossing: 1+(-3)+4+(-1)+2+1 = 4
#    Result: max(4, 2, 4) = 4... continue
#    Final answer: 6 [4,-1,2,1]
#    ```
# 8. **Recurrence**: T(n) = 2T(n/2) + O(n)
#    - Two recursive calls on halves: 2T(n/2)
#    - Linear time to find crossing: O(n)
#    - Solves to O(n log n)

# **Advantages**:
# - Demonstrates divide and conquer
# - Elegant recursive structure
# - Educational value
# - Parallelizable

# **Disadvantages**:
# - Slower than Kadane's O(n) algorithm
# - More complex to implement
# - Higher space complexity

# **When to Use**:
# - Teaching divide and conquer
# - When parallelization is needed
# - Historical/educational purposes
# - (Note: Kadane's is preferred in practice)



# ============================================================================
# ============================================================================
# 📌 WEEK 12: Branch and Bound
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Branch and Bound (Knapsack)
# ──────────────────────────────────────────────────────────
# ───────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(2^n) worst case, much better in practice
# 💾 SPACE COMPLEXITY: O(n) for recursion
# 📊 TYPE: Optimization (Branch & Bound)
# 🎯 DIFFICULTY: Hard
#

# **How the Algorithm Works**:
# 1. **Branch**: Explore decision tree (take/skip each item)
# 2. **Bound**: Calculate upper bound on potential value
# 3. **Prune**: Skip branches that can't beat current best
# 4. **Bounding Function**:
#    - Use fractional knapsack for upper bound
#    - Sort items by value/weight ratio (descending)
#    - Fill knapsack greedily with fractions allowed
#    - This gives optimistic estimate (upper bound)
# 5. **Algorithm Steps**:
#    - Sort items by value/weight ratio
#    - Track current best solution (max_value)
#    - For each node in search tree:
#      * Calculate bound for this branch
#      * If bound ≤ max_value: prune (can't improve)
#      * Else: branch into take/skip decisions
# 6. **Example**: Items [(w=2,v=40,r=20), (w=3,v=50,r=16.7), (w=5,v=60,r=12)]
#    - Capacity W=5
#    - Bound for taking item 1: 40 + (3/3)×50 = 90
#    - Explore this branch
#    - Bound for skipping item 1: (5/3)×50 + (2/5)×60 = 107.3
#    - Branching continues with pruning
# 7. **Pruning Power**: Eliminates large portions of search tree

# **Advantages**:
# - Better than brute force (pruning)
# - Finds optimal solution
# - Can terminate early
# - Uses bounds to prune

# **Disadvantages**:
# - Still exponential worst case
# - Complex to implement
# - Requires good bounding function
# - Not suitable for large n

# **When to Use**:
# - Exact solutions for NP-hard problems
# - Small to medium input sizes
# - When approximation not acceptable
# - Traveling Salesman Problem
# - 0/1 Knapsack with high accuracy needs

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: DFS with Pruning
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(2^n) worst case, better with pruning
# 💾 SPACE COMPLEXITY: O(n)
# 📊 TYPE: Search (with Optimization)
# 🎯 DIFFICULTY: Hard
#

# **How the Algorithm Works**:
# 1. **DFS**: Depth-first exploration of decision tree
# 2. **Pruning**: Cut off branches that can't improve solution
# 3. **Algorithm Steps**:
#    - Track current best solution (global max/min)
#    - Explore decision tree recursively
#    - At each node:
#      * Calculate bound/estimate for this branch
#      * If bound can't beat current best: prune (return)
#      * Else: continue exploring children
# 4. **Pruning Strategies**:
#    - **Feasibility pruning**: Branch violates constraints
#    - **Optimality pruning**: Branch can't improve best
#    - **Dominance pruning**: Another branch is strictly better
# 5. **Example** (Knapsack with W=10):
#    ```
#    Item 1 (w=6, v=30)
#    Item 2 (w=5, v=20)
#    Item 3 (w=4, v=15)
   
#    Tree exploration:
#    - Take item 1: capacity left = 4
#      * Can't take item 2 (w=5 > 4)
#      * Take item 3: value = 45 ✓
#    - Skip item 1:
#      * Take items 2,3: value = 35
#      * Pruned: can't beat 45
#    ```
# 6. **Effective Pruning** needs good bounds:
#    - Upper bounds for maximization
#    - Lower bounds for minimization

# **Advantages**:
# - Reduces search space significantly
# - Memory efficient (DFS)
# - Can find optimal solutions
# - Pruning improves average case

# **Disadvantages**:
# - Still exponential complexity
# - Pruning strategy problem-specific
# - May miss solutions if pruning incorrect

# **When to Use**:
# - Optimization problems
# - When branch and bound overhead too high
# - Constraint satisfaction
# - Game tree search



# ============================================================================
# ============================================================================
# 📌 WEEK 13: Graph Traversal Applications
# ============================================================================
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: BFS for State Space Search (Flappy Bird)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(H × T) where H = height, T = time intervals
# 💾 SPACE COMPLEXITY: O(H × T)
# 📊 TYPE: State Space Search
# 🎯 DIFFICULTY: Medium
#

# **How the Algorithm Works**:
# 1. **Model Game as State Space**:
#    - State = (bird_height, pipe_position, time)
#    - Actions = {flap, don't flap}
# 2. **State Transitions**:
#    - Flap: height increases by jump_velocity
#    - Don't flap: height decreases by gravity
#    - Each action advances time by 1
# 3. **BFS Algorithm**:
#    - Initialize queue with start state
#    - While queue not empty:
#      * Dequeue current state (h, pos, t)
#      * If reached goal (passed all pipes): return t
#      * Try both actions:
#        - Flap: new_h = h + jump
#        - No flap: new_h = h - gravity
#      * Check validity:
#        - Is new_h within bounds [0, height]?
#        - Does new_h avoid pipe at current position?
#      * If valid and not visited:
#        - Enqueue new state (new_h, pos+1, t+1)
# 4. **Example State Space**:
#    ```
#    Time 0: Bird at h=5
#    Time 1: Can be at h=7 (flap) or h=3 (fall)
#    Time 2: Multiple reachable heights...
#    Goal: Find path avoiding pipes
#    ```
# 5. **Key Insight**: BFS finds shortest time (minimum flaps)

# **Advantages**:
# - Models game as state space
# - Finds shortest solution
# - Complete algorithm
# - Natural for level-based games

# **Disadvantages**:
# - Memory intensive for large spaces
# - May explore many states
# - Requires good state representation

# **When to Use**:
# - Game AI
# - State space search
# - Shortest path in games
# - Planning problems
# - Robot motion planning

# ─────────────────────────────────────────────────────────────────────────────
# 🔷 ALGORITHM: Connected Components (DFS/BFS)
# ─────────────────────────────────────────────────────────────────────────────
#
# ⏱️  TIME COMPLEXITY:  O(rows × cols) for grid or O(V + E) for graph
# 💾 SPACE COMPLEXITY: O(rows × cols) or O(V + E)
# 📊 TYPE: Graph Traversal
# 🎯 DIFFICULTY: Easy
#

# **How the Algorithm Works**:
# 1. Goal: Find all separate connected regions
# 2. **DFS Approach**:
#    - Create visited array/set
#    - Initialize component_count = 0
#    - For each cell/vertex in grid/graph:
#      * If not visited:
#        - Increment component_count
#        - Start DFS from this cell
#        - Mark all reachable cells as visited
# 3. **DFS Process** (from starting cell):
#    - Mark current cell as visited
#    - For each adjacent cell (4 or 8 directions):
#      * If cell is valid and not visited:
#        - Recursively call DFS on that cell
# 4. **Example** (grid with X=land, .=water):
#    ```
#    X X . X
#    X . . X
#    . . X X
#    X . X .
#    ```
#    - Component 1: Top-left group (2 X's)
#    - Component 2: Top-right group (4 X's)
#    - Component 3: Bottom-left (1 X)
#    - Component 4: Bottom-middle group (2 X's)
#    - Total: 4 components
# 5. **Variation**: Can also find component sizes, largest component, etc.

# **Advantages**:
# - Identifies separate regions
# - Works for various graph types
# - Can count components efficiently
# - Useful for many applications

# **Disadvantages**:
# - Requires full graph traversal
# - Memory for visited tracking
# - May need repeated searches

# **When to Use**:
# - Image processing (finding blobs)
# - Network analysis
# - Island counting problems
# - Graph connectivity
# - Social network clustering
# - Flood fill algorithms

# ---

# ## General Algorithm Selection Guide

# ### Use DP When:
# - Problem has overlapping subproblems
# - Optimal substructure exists
# - Need exact optimal solution
# - Polynomial solution possible

# ### Use Greedy When:
# - Greedy choice property holds
# - Local optimum leads to global optimum
# - Need fast solution
# - Can prove correctness

# ### Use Backtracking When:
# - Need all solutions
# - Constraint satisfaction
# - Small input size
# - Pruning can help significantly

# ### Use Divide and Conquer When:
# - Problem divisible into independent subproblems
# - Combine step is efficient
# - Parallelization needed
# - Clean recursive structure

# ### Use Graph Algorithms When:
# - Relationship between entities
# - Network problems
# - Pathfinding needed
# - Connectivity matters

# ---

# ## Complexity Cheat Sheet

# | Algorithm | Time Complexity | Space Complexity |
# |-----------|----------------|------------------|
# | Kadane's | O(n) | O(1) |
# | Merge Sort | O(n log n) | O(n) |
# | Coin Change DP | O(V × n) | O(V) |
# | Edit Distance | O(m × n) | O(m × n) |
# | LCS | O(m × n) | O(m × n) |
# | 0/1 Knapsack | O(n × W) | O(n × W) |
# | BFS | O(V + E) | O(V) |
# | DFS | O(V + E) | O(V) |
# | IDS | O(b^d) | O(d) |
# | Kruskal's MST | O(E log E) | O(V + E) |
# | Activity Selection | O(n log n) | O(1) |
# | Fast Exponentiation | O(log n) | O(log n) |
# | N-Queens | O(n!) | O(n) |
# | Branch & Bound | O(2^n)* | O(n) |

# *With pruning, average case much better


# ============================================================================
# ============================================================================
# 📊 ALGORITHM SELECTION GUIDE
# ============================================================================
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                     WHEN TO USE EACH ALGORITHM TYPE                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Use DP When:                                                            │
│   • Problem has overlapping subproblems                                 │
│   • Optimal substructure exists                                         │
│   • Need exact optimal solution                                         │
│   Examples: Knapsack, Coin Change, Edit Distance, LCS                  │
│                                                                         │
│ Use Greedy When:                                                        │
│   • Local optimal choice leads to global optimal                        │
│   • Greedy choice property proven                                       │
│   • Need fast solution                                                  │
│   Examples: Activity Selection, Huffman, MST                            │
│                                                                         │
│ Use Backtracking When:                                                  │
│   • Need ALL solutions                                                  │
│   • Constraint satisfaction problems                                    │
│   • Pruning can reduce search space                                     │
│   Examples: N-Queens, Sudoku, Permutations                              │
│                                                                         │
│ Use Divide & Conquer When:                                              │
│   • Problem divisible into independent subproblems                      │
│   • Combine step is efficient                                           │
│   • Parallelization needed                                              │
│   Examples: Merge Sort, Fast Exponentiation, Max Subarray               │
│                                                                         │
│ Use Graph Algorithms When:                                              │
│   • Relationship between entities                                       │
│   • Network problems                                                    │
│   • Pathfinding needed                                                  │
│   Examples: BFS/DFS, Dijkstra, MST, Connected Components                │
└─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 📋 COMPLEXITY CHEAT SHEET
# ============================================================================

# - Always analyze your specific problem constraints
# - Consider input size when choosing algorithm
# - Time vs space tradeoffs are common
# - Optimization is often problem-specific
# - Understanding multiple approaches helps choose the best tool
# - Practice implementing these algorithms for mastery

# **Created**: March 1, 2026  
# **Course**: Algorithm Design  
# **Week Coverage**: Weeks 1-13


# ============================================================================
# ============================================================================
# 📑 WORKSHEET INDEX - COMPLETE ALGORITHM REFERENCE
# ============================================================================
# ============================================================================

"""
================================================================================
                     📚 COMPLETE WORKSHEET INDEX
                     All Algorithms by Week with Line Numbers
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                          📌 WEEK 1: Maximum Sum                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 95   │ 📌 Week Header                                                  │
│ Line 100  │ 🔷 Kadane's Algorithm (Maximum Subarray Sum)                    │
│           │    ⏱️  O(n) | 💾 O(1) | Type: Array Processing                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 2: Divide and Conquer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 144  │ 📌 Week Header                                                  │
│ Line 149  │ 🔷 Divide and Conquer (Maximum Subarray)                        │
│           │    ⏱️  O(n log n) | 💾 O(log n) | Type: Divide & Conquer        │
│ Line 194  │ 🔷 Balance Split (Recursive Backtracking)                       │
│           │    ⏱️  O(2^n) | 💾 O(n) | Type: Backtracking                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 3: Dynamic Programming Intro                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 243  │ 📌 Week Header                                                  │
│ Line 248  │ 🔷 Rod Cutting Problem (DP)                                     │
│           │    ⏱️  O(n²) | 💾 O(n) | Type: Optimization DP                  │
│ Line 293  │ 🔷 Minimum Coin Change (DP)                                     │
│           │    ⏱️  O(V×n) | 💾 O(V) | Type: Optimization DP                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 4: Memoization Techniques                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 340  │ 📌 Week Header                                                  │
│ Line 345  │ 🔷 Top-Down DP with Memoization                                 │
│           │    ⏱️  Varies | 💾 O(n) | Type: Top-Down DP                     │
│ Line 404  │ 🔷 0/1 Knapsack (Memoized)                                      │
│           │    ⏱️  O(n×W) | 💾 O(n×W) | Type: Optimization DP               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 5: Edit Distance                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 452  │ 📌 Week Header                                                  │
│ Line 457  │ 🔷 Levenshtein Distance (Edit Distance)                         │
│           │    ⏱️  O(m×n) | 💾 O(m×n) | Type: String DP                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 6: Advanced Dynamic Programming                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 511  │ 📌 Week Header                                                  │
│ Line 516  │ 🔷 Longest Common Subsequence (LCS)                             │
│           │    ⏱️  O(m×n) | 💾 O(m×n) | Type: String DP                     │
│ Line 569  │ 🔷 Knapsack (Bottom-Up DP)                                      │
│           │    ⏱️  O(n×W) | 💾 O(n×W) | Type: Optimization DP               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 7: Complex DP Problems                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 627  │ 📌 Week Header                                                  │
│ Line 632  │ 🔷 M3 Tile Problem (Tiling DP)                                  │
│           │    ⏱️  O(n) | 💾 O(n) | Type: Combinatorial DP                  │
│ Line 684  │ 🔷 Shoe Shopping (Optimization DP)                              │
│           │    ⏱️  O(n²) | 💾 O(n) | Type: Optimization DP                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 8: Backtracking Algorithms                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 713  │ 📌 Week Header                                                  │
│ Line 718  │ 🔷 N-Queens Problem (Backtracking)                              │
│           │    ⏱️  O(n!) | 💾 O(n) | Type: Constraint Satisfaction          │
│ Line 773  │ 🔷 Breadth-First Search (BFS)                                   │
│           │    ⏱️  O(V+E) | 💾 O(V) | Type: Graph Traversal                 │
│ Line 829  │ 🔷 Maze Running (DFS/BFS)                                       │
│           │    ⏱️  O(R×C) | 💾 O(R×C) | Type: Grid Traversal                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 9: Advanced Search Algorithms                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 891  │ 📌 Week Header                                                  │
│ Line 896  │ 🔷 Iterative Deepening Search (IDS)                             │
│           │    ⏱️  O(b^d) | 💾 O(bd) | Type: Uninformed Search              │
│ Line 948  │ 🔷 IDA* (Iterative Deepening A*)                                │
│           │    ⏱️  O(b^d) | 💾 O(d) | Type: Informed Search                 │
│ Line 999  │ 🔷 Uniform Cost Search (UCS)                                    │
│           │    ⏱️  O(E log V) | 💾 O(V) | Type: Weighted Search             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 10: Greedy Algorithms                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 1057 │ 📌 Week Header                                                  │
│ Line 1062 │ 🔷 Activity Selection (Greedy)                                  │
│           │    ⏱️  O(n log n) | 💾 O(1) | Type: Interval Scheduling         │
│ Line 1115 │ 🔷 Minimum Spanning Tree (Kruskal's/Prim's)                     │
│           │    ⏱️  O(E log E) | 💾 O(V) | Type: Graph Optimization          │
│ Line 1168 │ 🔷 Union-Find (Disjoint Sets)                                   │
│           │    ⏱️  O(α(n)) | 💾 O(n) | Type: Data Structure                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 11: Divide and Conquer (Advanced)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 1236 │ 📌 Week Header                                                  │
│ Line 1241 │ 🔷 Fast Exponentiation (Binary Exponentiation)                  │
│           │    ⏱️  O(log n) | 💾 O(1) | Type: Math Optimization             │
│ Line 1293 │ 🔷 Maximum Subarray (Divide and Conquer)                        │
│           │    ⏱️  O(n log n) | 💾 O(log n) | Type: Divide & Conquer        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 12: Branch and Bound                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 1351 │ 📌 Week Header                                                  │
│ Line 1356 │ 🔷 Branch and Bound (Knapsack)                                  │
│           │    ⏱️  O(2^n) | 💾 O(n) | Type: Optimization                    │
│ Line 1410 │ 🔷 DFS with Pruning                                             │
│           │    ⏱️  O(2^n) | 💾 O(n) | Type: Backtracking                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   📌 WEEK 13: Graph Traversal Applications                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Line 1472 │ 📌 Week Header                                                  │
│ Line 1477 │ 🔷 BFS for State Space Search (Flappy Bird)                     │
│           │    ⏱️  O(H×T) | 💾 O(H×T) | Type: State Space Search           │
│ Line 1535 │ 🔷 Connected Components (DFS/BFS)                               │
│           │    ⏱️  O(V+E) | 💾 O(V) | Type: Graph Traversal                 │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                          SUMMARY: 27 ALGORITHMS
================================================================================

📊 BY CATEGORY:
---------------
• Dynamic Programming (DP):     11 algorithms (Weeks 1,3,4,5,6,7)
• Graph Algorithms:              9 algorithms  (Weeks 8,9,13)
• Greedy Algorithms:             3 algorithms  (Week 10)
• Divide & Conquer:              3 algorithms  (Weeks 2,11)
• Backtracking/Pruning:          4 algorithms  (Weeks 2,8,12)
• Data Structures:               1 algorithm   (Week 10 - Union-Find)

⏱️  BY COMPLEXITY:
------------------
• O(n) or better:     Kadane's, Tiling, Union-Find, Fast Exp
• O(n log n):         Divide & Conquer, Activity Selection, MST
• O(n²) or O(m×n):    Rod Cutting, LCS, Edit Distance, Shoe Shopping
• O(V+E):             BFS, DFS, Connected Components
• O(2^n):             Balance Split, Branch & Bound, Pruning
• O(n!):              N-Queens

🎯 WORKSHEET MAPPING:
----------------------
Worksheet 1:  Kadane's Algorithm (Week 1)
Worksheet 2:  Divide & Conquer, Balance Split (Week 2)
Worksheet 3:  Rod Cutting, Coin Change (Week 3)
Worksheet 4:  Memoization, Knapsack (Week 4)
Worksheet 5:  Edit Distance (Week 5)
Worksheet 6:  LCS, Knapsack DP (Week 6)
Worksheet 7:  Tiling, Shoe Shopping (Week 7)
Worksheet 8:  N-Queens, BFS, Maze (Week 8)
Worksheet 9:  IDS, IDA*, UCS (Week 9)
Worksheet 10: Activity Selection, MST, Union-Find (Week 10)
Worksheet 11: Fast Exponentiation, Max Subarray (Week 11)
Worksheet 12: Branch & Bound, Pruning (Week 12)
Worksheet 13: State Space Search, Connected Components (Week 13)

================================================================================

💡 HOW TO USE THIS INDEX:
--------------------------
1. Press Cmd+G (Mac) or Ctrl+G (Windows) to go to line number
2. Use Cmd+F or Ctrl+F to search by algorithm name
3. Search by week: "Week 5", by type: "Graph", or complexity: "O(n)"
4. All algorithm headers marked with 🔷 for easy identification
5. Week headers marked with 📌 for quick navigation

📚 STUDY TIPS:
--------------
• Start with Week 1 fundamentals (Kadane's)
• Master DP concepts in Weeks 3-7 (most common in exams)
• Practice graph algorithms in Weeks 8-9 (BFS/DFS essential)
• Learn greedy approach in Week 10 (MST important)
• Advanced topics in Weeks 11-13 (optimization techniques)

================================================================================
                    Good luck with your worksheets! 🚀
================================================================================
"""
