"""
================================================================================
🎯 PURE ALGORITHMS REFERENCE - COMPLETE CODEBASE COLLECTION
================================================================================
This file contains all algorithm explanations used in the codebase
with detailed analysis and usage notes (NO CODE - explanations only).

Organization (Alphabetical by Algorithm Name):
  1.  Activity Selection (Greedy)
  2.  Binary Exponentiation (Fast Power)
  3.  Branch and Bound
  4.  Breadth-First Search (BFS)
  5.  Coin Change Problem
  6.  Depth-First Search (DFS)
  7.  Edit Distance (Levenshtein)
  8.  Euclidean Algorithm (GCD)
  9.  Fenwick Tree (Binary Indexed Tree)
  10. Frog Problem
  11. IDA* (Iterative Deepening A*)
  12. Iterative Deepening Search (IDS)
  13. Kadane's Algorithm
  14. Knapsack 0/1 Problem
  15. Kruskal's Algorithm (MST)
  16. Longest Common Subsequence (LCS)
  17. Maximum Subarray (Divide & Conquer)
  18. Merge Sort (for Inversion Count)
  19. N-Queens Problem
  20. Rod Cutting Problem
  21. Selection Sort
  22. Stair Climbing Problem
  23. Subset Sum / Balance Split
  24. Tiling Problem (M3 Tiles)
  25. Union-Find (Disjoint Set)

Each algorithm includes:
  ✓ Location: Where it's used in the codebase
  ✓ How it works: Simple explanation
  ✓ Advantages: When to use it
  ✓ Disadvantages: When to avoid it
  ✓ Time Complexity
  ✓ Space Complexity
================================================================================
"""


# ============================================================================
# 📋 ALGORITHM 1: ACTIVITY SELECTION (GREEDY)
# ============================================================================
"""
LOCATION:
  - worksheet10/activity_selection_v1.py
  - worksheet10/activity_selection_v2.py
  - algorithms_detail.py (Week 10)

HOW IT WORKS:
  Select maximum number of non-overlapping activities. Greedy approach:
  always pick activity with earliest finish time. Sort by finish time,
  then greedily select if compatible.

ADVANTAGES:
  ✓ Optimal solution with O(n log n) time
  ✓ Much faster than DP O(n²) approach
  ✓ Simple to implement
  ✓ Greedy choice property proven

DISADVANTAGES:
  ✗ Only maximizes count, not value
  ✗ Requires sorting first
  ✗ Doesn't work if activities have weights
  ✗ Greedy doesn't always work for variants

TIME COMPLEXITY: O(n log n) due to sorting
SPACE COMPLEXITY: O(1) ignoring input storage
"""


# ============================================================================
# 📋 ALGORITHM 2: BINARY EXPONENTIATION (FAST POWER)
# ============================================================================
"""
LOCATION:
  - worksheet11/exponentiation_v1.py (Iterative)
  - worksheet11/exponentiation_v2.py (Recursive)
  - algorithms_detail.py (Week 11)

HOW IT WORKS:
  Compute x^n efficiently by exploiting binary representation of n.
  If n is even: x^n = (x^(n/2))²
  If n is odd: x^n = x × x^(n-1)
  Reduces O(n) multiplications to O(log n).

ADVANTAGES:
  ✓ O(log n) time - huge improvement over O(n)
  ✓ Works for matrices, modular arithmetic
  ✓ Simple to implement
  ✓ Essential for cryptography

DISADVANTAGES:
  ✗ Requires careful handling of negative exponents
  ✗ Potential overflow for large numbers
  ✗ Recursive version uses O(log n) stack space

TIME COMPLEXITY: O(log n)
SPACE COMPLEXITY: O(1) iterative, O(log n) recursive
"""


# ============================================================================
# 📋 ALGORITHM 3: BRANCH AND BOUND
# ============================================================================
"""
LOCATION:
  - worksheet12/dfsKnapsack_branch_and_bound.py
  - worksheet12/KnapsackBound.py
  - algorithms_detail.py (Week 12)

HOW IT WORKS:
  Exhaustive search with pruning using bounds. Calculate upper bound (optimistic
  estimate) of best solution in subtree. If bound ≤ current best, prune branch.
  For knapsack: use fractional relaxation for upper bound.

ADVANTAGES:
  ✓ Finds optimal solution
  ✓ Prunes large portions of search space
  ✓ Works when DP not applicable
  ✓ Theoretical guarantee of optimality

DISADVANTAGES:
  ✗ Still exponential worst case O(2^n)
  ✗ Depends heavily on bound quality
  ✗ Complex to implement
  ✗ Can be slow in practice

TIME COMPLEXITY: O(2^n) worst case, much better with good bounds
SPACE COMPLEXITY: O(n) for recursion stack
"""


# ============================================================================
# 📋 ALGORITHM 4: BREADTH-FIRST SEARCH (BFS)
# ============================================================================
"""
LOCATION:
  - worksheet8/maze_running.py
  - worksheet13/flappybird.py
  - algorithms_detail.py (Week 8)
  - algorithms_example.py (Section 4)

HOW IT WORKS:
  Explores graph level by level using queue. Visits all neighbors of current
  node before moving to next level. Guarantees shortest path in unweighted
  graphs.

ADVANTAGES:
  ✓ Finds shortest path in unweighted graphs
  ✓ Complete - always finds solution if exists
  ✓ Good for finding nearby solutions
  ✓ Can detect cycles easily

DISADVANTAGES:
  ✗ Uses O(V) space for queue
  ✗ Slower than DFS for deep solutions
  ✗ Not suitable for weighted graphs (use Dijkstra)
  ✗ Explores many unnecessary nodes if solution is deep

TIME COMPLEXITY: O(V + E) where V = vertices, E = edges
SPACE COMPLEXITY: O(V) for queue and visited set
"""


# ============================================================================
# 📋 ALGORITHM 5: COIN CHANGE PROBLEM
# ============================================================================
"""
LOCATION:
  - worksheet3/minCoin-v1.py (Recursive)
  - worksheet3/minCoin-v2.py (DP)
  - worksheet5/minCoin-v3.py (Optimized)
  - algorithms_detail.py (Week 3)

HOW IT WORKS:
  Given coin denominations and target value, find minimum number of coins
  to make that value. DP table dp[i] stores min coins for value i.
  For each value, try using each coin and take minimum.

ADVANTAGES:
  ✓ Finds optimal solution (minimum coins)
  ✓ Works with any coin denominations
  ✓ Can handle large values efficiently
  ✓ Can be modified to count number of ways

DISADVANTAGES:
  ✗ O(V×n) time where V is target value
  ✗ Doesn't work if no solution exists (returns infinity)
  ✗ Assumes unlimited coins of each denomination

TIME COMPLEXITY: O(V×n) where V = value, n = number of coin types
SPACE COMPLEXITY: O(V) - DP array for all values 0 to V
"""


# ============================================================================
# 📋 ALGORITHM 6: DEPTH-FIRST SEARCH (DFS)
# ============================================================================
"""
LOCATION:
  - worksheet8/maze_running.py
  - worksheet12/* (in backtracking)
  - algorithms_detail.py (Week 8)
  - algorithms_example.py (Section 4)

HOW IT WORKS:
  Explores graph by going as deep as possible before backtracking. Uses stack
  (or recursion). Good for exploring all paths, detecting cycles, topological
  sorting.

ADVANTAGES:
  ✓ Uses O(h) space where h is depth (better than BFS)
  ✓ Good for exploring all possible paths
  ✓ Natural with recursion
  ✓ Fast for deep solutions

DISADVANTAGES:
  ✗ May not find shortest path
  ✗ Can get stuck in deep branches
  ✗ Requires cycle detection to avoid infinite loops
  ✗ Not optimal for shortest path problems

TIME COMPLEXITY: O(V + E)
SPACE COMPLEXITY: O(V) in worst case (all vertices on stack)
"""


# ============================================================================
# 📋 ALGORITHM 7: EDIT DISTANCE (LEVENSHTEIN)
# ============================================================================
"""
LOCATION:
  - worksheet5/edit_distance_v1.py (Recursive)
  - worksheet5/edit_distance_v2.py (DP)
  - algorithms_detail.py (Week 5)

HOW IT WORKS:
  Calculate minimum edits (insert, delete, replace) to transform one string
  to another. DP table dp[i][j] stores min edits for first i chars of string1
  to first j chars of string2. Compare characters and take minimum of three
  operations.

ADVANTAGES:
  ✓ Measures string similarity accurately
  ✓ Works for any strings (any length)
  ✓ Can reconstruct actual edit sequence
  ✓ Used in spell checkers, DNA analysis

DISADVANTAGES:
  ✗ O(m×n) time - slow for very long strings
  ✗ O(m×n) space for 2D table
  ✗ All operations have same cost (can be modified)
  ✗ Doesn't consider semantic meaning

TIME COMPLEXITY: O(m×n) where m, n = string lengths
SPACE COMPLEXITY: O(m×n) for 2D table, can optimize to O(min(m,n))
"""


# ============================================================================
# 📋 ALGORITHM 8: EUCLIDEAN ALGORITHM (GCD)
# ============================================================================
"""
LOCATION:
  - worksheet0/SourceCode-ALDS1_1_B.py
  - algorithms_detail.py (Week 0)

HOW IT WORKS:
  Find Greatest Common Divisor using repeated division. GCD(a,b) = GCD(b, a%b)
  until remainder is 0. Based on fact that GCD divides both numbers and
  their difference.

ADVANTAGES:
  ✓ Very efficient O(log min(a,b))
  ✓ Simple recursive or iterative implementation
  ✓ Ancient algorithm (300 BC)
  ✓ Extends to Extended Euclidean (finds x,y where ax+by=gcd)

DISADVANTAGES:
  ✗ Only works for integers
  ✗ Requires modulo operation
  ✗ Not useful for floating point

TIME COMPLEXITY: O(log min(a,b))
SPACE COMPLEXITY: O(1) iterative, O(log n) recursive
"""


# ============================================================================
# 📋 ALGORITHM 9: FENWICK TREE (BINARY INDEXED TREE)
# ============================================================================
"""
LOCATION:
  - assignment4/inversion_count_BIT.py
  - algorithms_detail.py

HOW IT WORKS:
  Tree structure for efficient range queries and point updates. Each node
  stores cumulative value for range of indices. Uses binary representation
  for parent-child relationships. Updates and queries in O(log n).

ADVANTAGES:
  ✓ O(log n) for both update and query
  ✓ Space efficient O(n)
  ✓ Easy to implement
  ✓ Useful for cumulative frequencies

DISADVANTAGES:
  ✗ Only works for cumulative operations
  ✗ Less versatile than segment tree
  ✗ Not intuitive to understand
  ✗ 1-indexed (common implementation)

TIME COMPLEXITY: O(log n) per operation
SPACE COMPLEXITY: O(n)
"""


# ============================================================================
# 📋 ALGORITHM 10: FROG PROBLEM
# ============================================================================
"""
LOCATION:
  - assignment3/frog1.py
  - assignment4/questions_of_3&4.py
  - algorithms_example.py

HOW IT WORKS:
  Frog on stone 1 wants to reach stone n. Can jump 1 or 2 stones forward.
  Jumping from stone i to j costs |height[i] - height[j]|.
  DP finds minimum total cost path.

ADVANTAGES:
  ✓ Simple 1D DP solution
  ✓ O(n) time complexity
  ✓ Can be extended to k jumps
  ✓ Intuitive problem structure

DISADVANTAGES:
  ✗ Limited to forward jumps only
  ✗ Only 1 or 2 stones at a time
  ✗ Doesn't handle general graphs

TIME COMPLEXITY: O(n)
SPACE COMPLEXITY: O(n) for DP array
"""


# ============================================================================
# 📋 ALGORITHM 11: IDA* (ITERATIVE DEEPENING A*)
# ============================================================================
"""
LOCATION:
  - worksheet9/8puzzle_IDAstar_example.py
  - algorithms_detail.py (Week 9)

HOW IT WORKS:
  Combines IDS with A* heuristic. Uses cost threshold instead of depth limit.
  f(n) = g(n) + h(n) where g = cost so far, h = heuristic estimate.
  Increases threshold to minimum f-value that exceeded previous threshold.

ADVANTAGES:
  ✓ O(bd) space like IDS
  ✓ Uses heuristic for faster search
  ✓ Optimal if heuristic is admissible
  ✓ Better than IDS for large state spaces

DISADVANTAGES:
  ✗ Revisits nodes (like IDS)
  ✗ Requires good heuristic function
  ✗ Slower than A* if memory available
  ✗ Complex to implement correctly

TIME COMPLEXITY: O(b^d) but often much better with good heuristic
SPACE COMPLEXITY: O(bd)
"""


# ============================================================================
# 📋 ALGORITHM 12: ITERATIVE DEEPENING SEARCH (IDS)
# ============================================================================
"""
LOCATION:
  - worksheet9/8puzzle_IDS_example.py
  - algorithms_detail.py (Week 9)

HOW IT WORKS:
  Combines benefits of BFS (completeness, optimality) and DFS (space efficiency).
  Performs depth-limited DFS repeatedly with increasing depth limits.
  Finds solution at depth d after exploring depths 0,1,2,...,d.

ADVANTAGES:
  ✓ Space complexity O(bd) like DFS
  ✓ Complete and optimal like BFS
  ✓ Good when depth unknown
  ✓ No heuristic needed

DISADVANTAGES:
  ✗ Revisits nodes multiple times
  ✗ Time complexity appears wasteful
  ✗ Slower than BFS with large branching factor
  ✗ Not suitable when depth is very large

TIME COMPLEXITY: O(b^d) where b = branching factor, d = depth
SPACE COMPLEXITY: O(bd)
"""


# ============================================================================
# 📋 ALGORITHM 13: KADANE'S ALGORITHM
# ============================================================================
"""
LOCATION:
  - worksheet1/maxsum_v3.py
  - worksheet11/maxSubSum.py (D&C comparison)
  - algorithms_detail.py (Week 1)
  - algorithms_example.py (Section 1)

HOW IT WORKS:
  Scans array left to right, at each position decides whether to:
  - Extend current subarray (add current element)
  - Start new subarray (reset to current element)
  Keeps track of maximum sum seen so far.

ADVANTAGES:
  ✓ Optimal O(n) time complexity - single pass
  ✓ Simple to implement and understand
  ✓ Constant O(1) space - only two variables needed
  ✓ Works with negative numbers
  ✓ Can be modified to track start/end indices

DISADVANTAGES:
  ✗ Only works for contiguous subarrays
  ✗ Doesn't find the actual subarray (additional tracking needed)
  ✗ Not suitable for 2D arrays (need different approach)

TIME COMPLEXITY: O(n) - single pass through array
SPACE COMPLEXITY: O(1) - only stores max_so_far and max_ending_here
"""


# ============================================================================
# 📋 ALGORITHM 14: KNAPSACK 0/1 PROBLEM
# ============================================================================
"""
LOCATION:
  - worksheet4/knapsack_v1.py (Backtracking)
  - worksheet4/knapsack_v2.py (Memoization)
  - worksheet4/knapsack_v3.py (Bottom-Up DP)
  - worksheet6/v2_mm.py, v3_dp.py
  - worksheet12/* (Branch & Bound variants)
  - algorithms_detail.py (Week 4, 6)

HOW IT WORKS:
  Given items with weights and values, and weight capacity W, maximize total
  value without exceeding capacity. DP table dp[i][w] = max value using first
  i items with capacity w. For each item, choose to include or exclude.

ADVANTAGES:
  ✓ Guarantees optimal solution
  ✓ Pseudo-polynomial time O(n×W)
  ✓ Can reconstruct which items to take
  ✓ Works for any weights/values

DISADVANTAGES:
  ✗ Not truly polynomial (depends on W magnitude)
  ✗ O(n×W) space for 2D table (can optimize to O(W))
  ✗ Only handles integer weights efficiently
  ✗ Each item can only be used once (0/1 constraint)

TIME COMPLEXITY: O(n×W) where n = items, W = capacity
SPACE COMPLEXITY: O(n×W) for 2D table, O(W) for space-optimized 1D
"""


# ============================================================================
# 📋 ALGORITHM 15: KRUSKAL'S ALGORITHM (MST)
# ============================================================================
"""
LOCATION:
  - worksheet10/mst.py
  - algorithms_detail.py (Week 10)

HOW IT WORKS:
  Find minimum cost tree connecting all vertices. Kruskal's: sort edges by
  weight, add edge if it doesn't create cycle (uses Union-Find). Greedy
  choice: always add cheapest valid edge.

ADVANTAGES:
  ✓ Optimal solution guaranteed
  ✓ O(E log E) time complexity
  ✓ Works on disconnected graphs (finds forest)
  ✓ Simple with Union-Find data structure

DISADVANTAGES:
  ✗ Requires sorting all edges
  ✗ Needs Union-Find implementation
  ✗ Not efficient for dense graphs (use Prim's)
  ✗ Doesn't handle directed graphs

TIME COMPLEXITY: O(E log E) for sorting + O(E α(V)) for union-find
SPACE COMPLEXITY: O(V + E) for graph storage and union-find
"""


# ============================================================================
# 📋 ALGORITHM 16: LONGEST COMMON SUBSEQUENCE (LCS)
# ============================================================================
"""
LOCATION:
  - worksheet6/LCS.py
  - algorithms_detail.py (Week 6)
  - algorithms_example.py

HOW IT WORKS:
  Find longest subsequence present in both strings (not necessarily contiguous).
  DP table dp[i][j] stores LCS length for first i chars of str1 and first j
  chars of str2. If characters match, extend LCS; otherwise take max of
  excluding one character from either string.

ADVANTAGES:
  ✓ Finds optimal longest common subsequence
  ✓ Works for any strings
  ✓ Can reconstruct actual subsequence
  ✓ Used in diff tools, bioinformatics

DISADVANTAGES:
  ✗ O(m×n) time complexity
  ✗ O(m×n) space (can be optimized to O(min(m,n)))
  ✗ Finds subsequence not substring
  ✗ Doesn't consider order importance

TIME COMPLEXITY: O(m×n) where m, n = string lengths
SPACE COMPLEXITY: O(m×n) for 2D table
"""


# ============================================================================
# 📋 ALGORITHM 17: MAXIMUM SUBARRAY (DIVIDE & CONQUER)
# ============================================================================
"""
LOCATION:
  - worksheet11/maxSubSum.py
  - algorithms_detail.py (Week 11)

HOW IT WORKS:
  Divide array in half. Maximum subarray is either:
  1. Entirely in left half
  2. Entirely in right half  
  3. Crosses the middle
  Recursively solve left/right, calculate crossing, return maximum.

ADVANTAGES:
  ✓ O(n log n) time
  ✓ Demonstrates divide & conquer paradigm
  ✓ No extra space needed
  ✓ Good for teaching/learning

DISADVANTAGES:
  ✗ Slower than Kadane's O(n) algorithm
  ✗ More complex to implement
  ✗ O(log n) recursion depth
  ✗ Not practical when Kadane's exists

TIME COMPLEXITY: O(n log n)
SPACE COMPLEXITY: O(log n) for recursion stack
"""


# ============================================================================
# 📋 ALGORITHM 18: MERGE SORT (FOR INVERSION COUNT)
# ============================================================================
"""
LOCATION:
  - assignment4/inversion_count.py
  - algorithms_detail.py (Week 11)
  - algorithms_example.py

HOW IT WORKS:
  Divide array in half recursively, sort each half, merge sorted halves.
  During merge, count inversions: when element from right half is smaller,
  all remaining left elements form inversions with it.

ADVANTAGES:
  ✓ O(n log n) guaranteed (not worst-case like quicksort)
  ✓ Stable sort (preserves order of equal elements)
  ✓ Counts inversions efficiently
  ✓ Good for linked lists

DISADVANTAGES:
  ✗ O(n) extra space for merging
  ✗ Not in-place algorithm
  ✗ Slower than quicksort in practice (constant factors)
  ✗ Slower than insertion sort for small arrays

TIME COMPLEXITY: O(n log n)
SPACE COMPLEXITY: O(n) for temporary array
"""


# ============================================================================
# 📋 ALGORITHM 19: N-QUEENS PROBLEM
# ============================================================================
"""
LOCATION:
  - worksheet8/n_queens.py
  - worksheet8/print_queens.py
  - algorithms_detail.py (Week 8)

HOW IT WORKS:
  Place n queens on n×n board so no two attack each other. Try placing queen
  in each row, backtrack if no valid position. Check diagonals and columns
  for conflicts.

ADVANTAGES:
  ✓ Finds all valid solutions
  ✓ Can optimize with symmetry
  ✓ Pruning makes it practical for n ≤ 15
  ✓ Classic backtracking example

DISADVANTAGES:
  ✗ O(n!) time complexity
  ✗ Very slow for large n (n > 20)
  ✗ Requires significant pruning for efficiency
  ✗ High memory for solution storage

TIME COMPLEXITY: O(n!) with pruning
SPACE COMPLEXITY: O(n²) for board
"""


# ============================================================================
# 📋 ALGORITHM 20: ROD CUTTING PROBLEM
# ============================================================================
"""
LOCATION:
  - worksheet3/maxRev-v1.py (Recursive)
  - worksheet3/maxRev-v2.py (DP)
  - algorithms_detail.py (Week 3)

HOW IT WORKS:
  Given rod of length n and prices for different lengths, find maximum 
  revenue by cutting rod into pieces. DP table dp[i] stores max revenue 
  for length i. For each length, try all possible first cuts and take maximum.

ADVANTAGES:
  ✓ Finds optimal solution guaranteed
  ✓ Handles any price structure
  ✓ Can reconstruct actual cuts made
  ✓ Much faster than exponential recursion

DISADVANTAGES:
  ✗ O(n²) time - slow for very long rods
  ✗ Requires O(n) space for DP table
  ✗ Assumes prices are given (not discovering them)

TIME COMPLEXITY: O(n²) - nested loops for length and cuts
SPACE COMPLEXITY: O(n) - DP array of size n+1
"""


# ============================================================================
# 📋 ALGORITHM 21: SELECTION SORT
# ============================================================================
"""
LOCATION:
  - worksheet0/SourceCode-P2.py
  - algorithms_detail.py (Week 0)

HOW IT WORKS:
  Find minimum element in unsorted part, swap with first element of unsorted
  part. Repeat until array is sorted. Simple but inefficient.

ADVANTAGES:
  ✓ Simple to understand and implement
  ✓ O(1) extra space
  ✓ Performs well on small arrays
  ✓ Minimum number of swaps

DISADVANTAGES:
  ✗ O(n²) time always (even if sorted)
  ✗ Not stable (can change relative order)
  ✗ Not adaptive (doesn't benefit from partial sorting)
  ✗ Many comparisons

TIME COMPLEXITY: O(n²)
SPACE COMPLEXITY: O(1)
"""


# ============================================================================
# 📋 ALGORITHM 22: STAIR CLIMBING PROBLEM
# ============================================================================
"""
LOCATION:
  - assignment2/stair-climbing-v1.py (Recursive)
  - assignment2/stair-climbing-v2.py (Memoization)
  - assignment2/stair-climbing-v3.py (DP)

HOW IT WORKS:
  Count ways to climb n stairs when you can take 1 or 2 steps at a time.
  Recurrence: ways[n] = ways[n-1] + ways[n-2] (Fibonacci sequence)
  Each position can be reached from previous step or two steps back.

ADVANTAGES:
  ✓ Very simple and intuitive
  ✓ O(n) time with DP
  ✓ Can optimize to O(1) space
  ✓ Can be extended to k steps

DISADVANTAGES:
  ✗ Only counts ways, not actual paths
  ✗ Can overflow for large n
  ✗ Assumes all ways are valid (no constraints)

TIME COMPLEXITY: O(n) with DP
SPACE COMPLEXITY: O(n) for DP, O(1) with space optimization
"""


# ============================================================================
# 📋 ALGORITHM 23: SUBSET SUM / BALANCE SPLIT
# ============================================================================
"""
LOCATION:
  - worksheet2/balanceSplit.py
  - algorithms_detail.py (Week 2)

HOW IT WORKS:
  Find if array can be split into two subsets with equal sum. Try including/
  excluding each element, backtrack if sum exceeds target. Target is
  total_sum / 2.

ADVANTAGES:
  ✓ Finds exact solution if exists
  ✓ Can enumerate all valid splits
  ✓ Pruning reduces search space
  ✓ Can add constraints easily

DISADVANTAGES:
  ✗ O(2^n) exponential time
  ✗ Very slow for large arrays
  ✗ DP solution exists and is faster
  ✗ High memory with recursion

TIME COMPLEXITY: O(2^n)
SPACE COMPLEXITY: O(n) for recursion stack
"""


# ============================================================================
# 📋 ALGORITHM 24: TILING PROBLEM (M3 TILES)
# ============================================================================
"""
LOCATION:
  - worksheet7/m3tileBF.py (Brute Force)
  - worksheet7/m3tileMM.py (Memoization)
  - worksheet7/m3tileDP.py (DP)
  - algorithms_detail.py (Week 7)

HOW IT WORKS:
  Count ways to tile a board using tiles of sizes 1×1, 1×2, 1×3.
  DP uses recurrence: ways[n] = ways[n-1] + ways[n-2] + ways[n-3]
  Similar to Fibonacci but with three previous terms.

ADVANTAGES:
  ✓ Linear O(n) time with DP
  ✓ Simple recurrence relation
  ✓ Easy to modify for different tile sizes
  ✓ Counts all valid configurations

DISADVANTAGES:
  ✗ Only counts ways, doesn't enumerate them
  ✗ Limited to 1D tiling (2D more complex)
  ✗ Can overflow for large n (need big integers)

TIME COMPLEXITY: O(n) with DP, O(3^n) with naive recursion
SPACE COMPLEXITY: O(n) for DP array, can optimize to O(1) with variables
"""


# ============================================================================
# 📋 ALGORITHM 25: UNION-FIND (DISJOINT SET)
# ============================================================================
"""
LOCATION:
  - worksheet10/mst.py (used in Kruskal's)
  - algorithms_detail.py (Week 10)

HOW IT WORKS:
  Maintains disjoint sets with operations: union (merge two sets) and find
  (which set contains element). Uses path compression and union by rank
  for near-constant amortized time.

ADVANTAGES:
  ✓ Near O(1) amortized time per operation
  ✓ Essential for Kruskal's MST
  ✓ Detects cycles efficiently
  ✓ Simple implementation

DISADVANTAGES:
  ✗ Not useful for dynamic connectivity
  ✗ Can't easily split sets once merged
  ✗ Requires array-based representation

TIME COMPLEXITY: O(α(n)) amortized per operation (α = inverse Ackermann)
SPACE COMPLEXITY: O(n)
"""


# ============================================================================
# 🎯 ALGORITHM COMPLEXITY COMPARISON TABLE
# ============================================================================
"""
================================================================================
                    QUICK REFERENCE: ALL ALGORITHMS
================================================================================

┌────────────────────────────────────┬──────────────┬─────────────┬───────────┐
│ ALGORITHM NAME                     │ TIME         │ SPACE       │ PARADIGM  │
├────────────────────────────────────┼──────────────┼─────────────┼───────────┤
│ Activity Selection                 │ O(n log n)   │ O(1)        │ Greedy    │
│ Binary Exponentiation              │ O(log n)     │ O(1)        │ D&C       │
│ Branch and Bound                   │ O(2^n)*      │ O(n)        │ Optimize  │
│ Breadth-First Search (BFS)         │ O(V+E)       │ O(V)        │ Graph     │
│ Coin Change                        │ O(V×n)       │ O(V)        │ DP        │
│ Depth-First Search (DFS)           │ O(V+E)       │ O(V)        │ Graph     │
│ Edit Distance                      │ O(m×n)       │ O(m×n)      │ DP        │
│ Euclidean Algorithm (GCD)          │ O(log n)     │ O(1)        │ Math      │
│ Fenwick Tree (per operation)       │ O(log n)     │ O(n)        │ Data Str  │
│ Frog Problem                       │ O(n)         │ O(n)        │ DP        │
│ IDA*                               │ O(b^d)*      │ O(bd)       │ Search    │
│ Iterative Deepening Search         │ O(b^d)       │ O(bd)       │ Search    │
│ Kadane's Algorithm                 │ O(n)         │ O(1)        │ DP/Greedy │
│ Knapsack 0/1                       │ O(n×W)       │ O(n×W)      │ DP        │
│ Kruskal's MST                      │ O(E log E)   │ O(V+E)      │ Greedy    │
│ Longest Common Subsequence         │ O(m×n)       │ O(m×n)      │ DP        │
│ Maximum Subarray (D&C)             │ O(n log n)   │ O(log n)    │ D&C       │
│ Merge Sort                         │ O(n log n)   │ O(n)        │ D&C       │
│ N-Queens                           │ O(n!)        │ O(n²)       │ Backtrack │
│ Rod Cutting                        │ O(n²)        │ O(n)        │ DP        │
│ Selection Sort                     │ O(n²)        │ O(1)        │ Sort      │
│ Stair Climbing                     │ O(n)         │ O(n)        │ DP        │
│ Subset Sum / Balance Split         │ O(2^n)       │ O(n)        │ Backtrack │
│ Tiling Problem                     │ O(n)         │ O(n)        │ DP        │
│ Union-Find (per operation)         │ O(α(n))      │ O(n)        │ Data Str  │
└────────────────────────────────────┴──────────────┴─────────────┴───────────┘

Note: * = often much better in practice with pruning/heuristics
      α(n) = inverse Ackermann function (effectively constant)

================================================================================
                        ALGORITHM SELECTION GUIDE
================================================================================

PROBLEM TYPE                              → RECOMMENDED ALGORITHM
──────────────────────────────────────────────────────────────────────────────
Contiguous subarray maximum               → Kadane's Algorithm
Optimization with overlapping subproblems → Dynamic Programming
Maximize value with capacity constraint   → Knapsack (DP or Branch & Bound)
String similarity/transformation          → Edit Distance, LCS
Count ways to achieve goal                → DP (Tiling, Stair Climbing, Frog)
Shortest path in unweighted graph         → BFS
Explore all paths / possibilities         → DFS, Backtracking
Select non-overlapping activities         → Activity Selection (Greedy)
Connect all nodes minimum cost            → Kruskal's MST
Fast computation of power                 → Binary Exponentiation
Count inversions efficiently              → Merge Sort
Constraint satisfaction puzzle            → Backtracking (N-Queens)
State space search unknown depth          → IDS
State space with good heuristic           → IDA*
Optimization with intelligent pruning     → Branch & Bound
Cycle detection / disjoint sets           → Union-Find
Range queries with point updates          → Fenwick Tree
Greatest common divisor                   → Euclidean Algorithm
Partition into equal sum subsets          → Subset Sum / Balance Split

================================================================================
                    PARADIGM-BASED CLASSIFICATION
================================================================================

DYNAMIC PROGRAMMING:
  • Coin Change Problem
  • Edit Distance (Levenshtein)
  • Frog Problem
  • Kadane's Algorithm
  • Knapsack 0/1 Problem
  • Longest Common Subsequence (LCS)
  • Rod Cutting Problem
  • Stair Climbing Problem
  • Tiling Problem (M3 Tiles)

GRAPH ALGORITHMS:
  • Breadth-First Search (BFS)
  • Depth-First Search (DFS)
  • Kruskal's Algorithm (MST)

GREEDY ALGORITHMS:
  • Activity Selection
  • Kruskal's Algorithm (MST)

DIVIDE & CONQUER:
  • Binary Exponentiation (Fast Power)
  • Maximum Subarray (Divide & Conquer)
  • Merge Sort (for Inversion Count)

BACKTRACKING:
  • N-Queens Problem
  • Subset Sum / Balance Split

SEARCH ALGORITHMS:
  • Breadth-First Search (BFS)
  • Depth-First Search (DFS)
  • IDA* (Iterative Deepening A*)
  • Iterative Deepening Search (IDS)

OPTIMIZATION:
  • Branch and Bound

DATA STRUCTURES:
  • Fenwick Tree (Binary Indexed Tree)
  • Union-Find (Disjoint Set)

SORTING:
  • Merge Sort
  • Selection Sort

MATHEMATICAL:
  • Euclidean Algorithm (GCD)
  • Binary Exponentiation

================================================================================
                        KEY INSIGHTS AND TRADE-OFFS
================================================================================

DP vs GREEDY:
  ✓ DP: Always finds optimal, explores all subproblems, slower
  ✓ Greedy: Fast but needs proof of optimality, makes local choices
  → Use DP when greedy doesn't work (Knapsack, Edit Distance)
  → Use Greedy when proven optimal (Activity Selection, MST)

MEMOIZATION vs TABULATION:
  ✓ Memoization: Top-down, recursive, only computes needed states
  ✓ Tabulation: Bottom-up, iterative, computes all states
  → Use Memoization for sparse state spaces
  → Use Tabulation for better space optimization

BFS vs DFS:
  ✓ BFS: Shortest path, level-order, uses more memory
  ✓ DFS: Memory efficient, explores deeply, natural for backtracking
  → Use BFS for shortest paths
  → Use DFS for all paths or deep exploration

RECURSION vs ITERATION:
  ✓ Recursion: Elegant for trees, divide-and-conquer, natural structure
  ✓ Iteration: Faster, no stack overflow, better for DP
  → Use Recursion for tree/graph problems, divide-and-conquer
  → Use Iteration for DP, when stack depth is concern

SPACE vs TIME:
  ✓ Can often trade space for time (memoization, DP tables)
  ✓ Can trade time for space (recomputation, IDS vs BFS)
  → Choose based on constraints (memory limited? time critical?)

EXACT vs APPROXIMATE:
  ✓ Exact: Branch & Bound, DP - guaranteed optimal but potentially slow
  ✓ Approximate: Greedy, heuristics - fast but may not be optimal
  → Use exact when optimality is critical
  → Use approximate when "good enough" solutions suffice

================================================================================
                        CODEBASE SUMMARY
================================================================================

TOTAL ALGORITHMS: 25 core algorithms
WORKSHEETS: 0-13 (14 weeks)
ASSIGNMENTS: 1-4
REFERENCE FILES: algorithms_detail.py, algorithms_example.py, ALGORITHM_REFERENCE_GUIDE.md

COVERAGE BY WEEK:
  Week 0-1:  Introduction, Kadane's Algorithm
  Week 2-3:  Divide & Conquer, Basic DP (Rod Cutting, Coin Change)
  Week 4-6:  Memoization, Advanced DP (Knapsack, Edit Distance, LCS)
  Week 7:    Complex DP (Tiling, Shoe Shopping)
  Week 8:    Backtracking (N-Queens), Graph (BFS, DFS)
  Week 9:    Advanced Search (IDS, IDA*)
  Week 10:   Greedy (Activity Selection), MST, Union-Find
  Week 11:   Divide & Conquer Advanced (Fast Power, Merge Sort)
  Week 12:   Optimization (Branch & Bound, Pruning)
  Week 13:   State Space Search, Applications

================================================================================
                    For complete code implementations, see:
                    - algorithms_detail.py (27 algorithms with code)
                    - algorithms_example.py (50 practice problems)
================================================================================
"""

print("="*80)
print("Pure Algorithms Reference - Explanations Only (No Code)")
print("="*80)
print("Total Algorithms: 25")
print("Organized Alphabetically by Algorithm Name")
print("Each includes: Location, How It Works, Advantages, Disadvantages, Complexity")
print("="*80)
