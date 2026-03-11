"""
================================================================================
🎯 UNIQUE ALGORITHMS REFERENCE - ALGORITHM TECHNIQUES ONLY
================================================================================
This file contains only UNIQUE algorithm techniques used in the codebase.
Problems like "Coin Change", "Frog", "Knapsack" are NOT algorithms - they are
problems solved using algorithms like Dynamic Programming, Backtracking, etc.

Organization (Alphabetical by Algorithm Technique):
  1.  Activity Selection (Greedy)
  2.  Binary Exponentiation (Divide & Conquer)
  3.  Branch and Bound
  4.  Breadth-First Search (BFS)
  5.  Depth-First Search (DFS)
  6.  DFS with Backtracking
  7.  DFS with Pruning
  8.  Dynamic Programming - 1D
  9.  Dynamic Programming - 2D
  10. Euclidean Algorithm (GCD)
  11. Fenwick Tree (Binary Indexed Tree)
  12. IDA* (Iterative Deepening A*)
  13. Iterative Deepening Search (IDS)
  14. Kadane's Algorithm (Greedy/DP Hybrid)
  15. Kruskal's Algorithm (MST)
  16. Maximum Subarray (Divide & Conquer)
  17. Merge Sort
  18. Selection Sort
  19. Union-Find (Disjoint Set)

Each algorithm includes:
  ✓ Where used: Types of problems it solves
  ✓ How it works: Core technique explanation
  ✓ Advantages
  ✓ Disadvantages
  ✓ Best/Worst/Average complexity
================================================================================
"""


# ============================================================================
# 📋 ALGORITHM 1: ACTIVITY SELECTION (GREEDY)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet10/activity_selection_v1.py
  - worksheet10/activity_selection_v2.py

ALGORITHM TYPE: Greedy Algorithm

HOW IT WORKS:
  Sort items by a criterion (e.g., finish time), then greedily select items
  that don't conflict with previous selections. Makes locally optimal choice
  at each step without reconsidering.

ADVANTAGES:
  ✓ Fast O(n log n) time complexity
  ✓ Simple to implement
  ✓ Works well when greedy choice property holds
  ✓ No need to explore all possibilities

DISADVANTAGES:
  ✗ Only works when greedy choice property is proven
  ✗ Doesn't find optimal solution for all problems
  ✗ Hard to know if greedy approach is correct
  ✗ Usually maximizes count, not weighted value

BEST CASE: O(n log n)
  → Why: Must sort all activities by finish time first
  → When: Always requires sorting, no way to skip

WORST CASE: O(n log n)
  → Why: Sorting dominates the complexity
  → When: Even if all activities selected, still need to sort

AVERAGE TIME: O(n log n)
SPACE: O(1)
"""


# ============================================================================
# 📋 ALGORITHM 2: BINARY EXPONENTIATION (DIVIDE & CONQUER)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet11/exponentiation_v1.py
  - worksheet11/exponentiation_v2.py

ALGORITHM TYPE: Divide & Conquer

HOW IT WORKS:
  Compute x^n by exploiting binary representation. If n is even: x^n = (x^(n/2))²
  If n is odd: x^n = x × x^(n-1). Reduces multiplications from O(n) to O(log n).

ADVANTAGES:
  ✓ Logarithmic time O(log n)
  ✓ Works for matrices, modular arithmetic
  ✓ Essential for cryptography
  ✓ Much faster than naive multiplication

DISADVANTAGES:
  ✗ Requires careful overflow handling
  ✗ Recursive version uses stack space
  ✗ More complex than naive approach

BEST CASE: O(log n)
  → Why: Even if n=1, still processes binary representation
  → When: Always O(log n) regardless of n value

WORST CASE: O(log n)
  → Why: Maximum log₂(n) divisions by 2
  → When: Large n values, but still logarithmic

AVERAGE TIME: O(log n)
SPACE: O(1) iterative, O(log n) recursive
"""


# ============================================================================
# 📋 ALGORITHM 3: BRANCH AND BOUND
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet12/dfsKnapsack_branch_and_bound.py
  - worksheet12/KnapsackBound.py
  - Problems: Knapsack optimization

ALGORITHM TYPE: Optimization with Intelligent Pruning

HOW IT WORKS:
  Exhaustive search enhanced with bounds. Calculate upper bound (optimistic
  estimate) for each branch. If bound ≤ current best solution, prune that
  entire branch. Commonly uses fractional relaxation for bounds.

ADVANTAGES:
  ✓ Finds guaranteed optimal solution
  ✓ Prunes large portions of search space
  ✓ Works when DP not applicable
  ✓ Better than brute force

DISADVANTAGES:
  ✗ Still exponential worst case
  ✗ Heavily depends on bound quality
  ✗ Complex to implement
  ✗ Can be slow in practice

BEST CASE: O(n) with perfect pruning
  → Why: Bounds eliminate nearly all branches immediately
  → When: Tight bounds, optimal solution found early

WORST CASE: O(2^n) when no effective pruning
  → Why: Poor bounds force exploration of all branches
  → When: Loose bounds, optimal solution found late

AVERAGE TIME: Varies with bound quality
SPACE: O(n) recursion stack
"""


# ============================================================================
# 📋 ALGORITHM 4: BREADTH-FIRST SEARCH (BFS)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet8/maze_running.py
  - worksheet13/flappybird.py
  - Problems: Shortest path in unweighted graphs, level-order traversal

ALGORITHM TYPE: Graph Traversal

HOW IT WORKS:
  Explore graph level by level using a queue. Visit all neighbors of current
  node before moving to next level. Guarantees shortest path in unweighted graphs.

ADVANTAGES:
  ✓ Finds shortest path in unweighted graphs
  ✓ Complete - always finds solution if exists
  ✓ Good for nearby solutions
  ✓ Can detect cycles

DISADVANTAGES:
  ✗ Uses O(V) space for queue
  ✗ Slower than DFS for deep solutions
  ✗ Not suitable for weighted graphs
  ✗ Explores many unnecessary nodes if solution is deep

BEST CASE: O(1) if target at start
  → Why: Target found immediately, no traversal needed
  → When: Starting node is the goal node

WORST CASE: O(V + E) visits all vertices
  → Why: Must explore all vertices and edges to find target
  → When: Target at deepest level or doesn't exist

AVERAGE TIME: O(V + E)
SPACE: O(V) for queue and visited set
"""


# ============================================================================
# 📋 ALGORITHM 5: DEPTH-FIRST SEARCH (DFS)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet8/maze_running.py
  - Problems: Path finding, graph exploration, cycle detection

ALGORITHM TYPE: Graph Traversal

HOW IT WORKS:
  Explore graph by going as deep as possible before backtracking. Uses stack
  or recursion. Goes down one path completely before trying alternatives.

ADVANTAGES:
  ✓ Uses O(depth) space - better than BFS
  ✓ Good for exploring all paths
  ✓ Natural with recursion
  ✓ Fast for deep solutions

DISADVANTAGES:
  ✗ May not find shortest path
  ✗ Can get stuck in deep branches
  ✗ Requires cycle detection
  ✗ Not optimal for shortest paths

BEST CASE: O(1) if target at start
  → Why: Target found immediately at first node
  → When: Starting node is the goal

WORST CASE: O(V + E) visits all vertices
  → Why: May traverse entire graph before finding target
  → When: Target in last branch explored or doesn't exist

AVERAGE TIME: O(V + E)
SPACE: O(V) worst case for stack
"""


# ============================================================================
# 📋 ALGORITHM 6: DFS WITH BACKTRACKING
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet8/n_queens.py
  - worksheet8/print_queens.py
  - worksheet2/balanceSplit.py
  - Problems: N-Queens, Subset Sum, Constraint Satisfaction

ALGORITHM TYPE: Exhaustive Search with Pruning

HOW IT WORKS:
  DFS that builds solutions incrementally. At each step, try all possibilities.
  If a partial solution violates constraints, backtrack immediately (prune).
  Continue until all valid solutions found or search space exhausted.

ADVANTAGES:
  ✓ Finds all valid solutions
  ✓ Can find one solution quickly if it exists
  ✓ Pruning reduces search space significantly
  ✓ Works for constraint satisfaction problems

DISADVANTAGES:
  ✗ Exponential time complexity O(b^d)
  ✗ Very slow for large problem sizes
  ✗ May explore many dead ends
  ✗ Memory intensive with recursion

BEST CASE: O(d) finds solution immediately
  → Why: First path tried leads to valid solution
  → When: Lucky ordering, constraints easily satisfied

WORST CASE: O(b^d) explores all branches
  → Why: Must explore entire search tree to find all solutions
  → When: Many constraints, solutions rare or at leaves

AVERAGE TIME: O(b^d) with pruning
SPACE: O(d) recursion depth
"""


# ============================================================================
# 📋 ALGORITHM 7: DFS WITH PRUNING (ADVANCED)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet12/dfsKnapsack_pruning.py
  - worksheet12/dfsKnapsack_branch_and_bound.py
  - Problems: Knapsack with bounds

ALGORITHM TYPE: DFS with Bounds-Based Pruning

HOW IT WORKS:
  Like backtracking but uses bounds/estimates to prune more aggressively.
  Before exploring a branch, calculate upper bound on possible solution quality.
  If bound ≤ current best, prune entire branch. More sophisticated than
  basic constraint checking.

ADVANTAGES:
  ✓ More aggressive pruning than basic backtracking
  ✓ Can solve larger problems than brute force
  ✓ Finds optimal solution
  ✓ Prunes based on solution quality, not just constraints

DISADVANTAGES:
  ✗ Still exponential worst case
  ✗ Requires good bounding function
  ✗ More complex to implement
  ✗ Depends on bound quality

BEST CASE: O(n) with excellent bounds
  → Why: Tight bounds prune nearly all suboptimal branches
  → When: High-quality bounding function, clear optimal path

WORST CASE: O(2^n) with poor bounds
  → Why: Weak bounds allow exploration of most branches
  → When: Loose bounds, many promising-looking branches

AVERAGE TIME: Much better than O(2^n) with good bounds
SPACE: O(n) recursion stack
"""


# ============================================================================
# 📋 ALGORITHM 8: DYNAMIC PROGRAMMING - 1D
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - assignment2/* (Stair Climbing)
  - assignment3/* (Frog)
  - worksheet3/* (Coin Change, Rod Cutting)
  - worksheet7/* (Tiling)
  - worksheet1/* (Kadane's - can be viewed as DP)
  - Problems: Stair Climbing, Frog, Coin Change, Rod Cutting, Tiling

ALGORITHM TYPE: Dynamic Programming

HOW IT WORKS:
  Solve by building solutions to larger problems from smaller subproblems.
  Use 1D array dp[i] where each state depends on previous states.
  Recurrence: dp[i] = f(dp[i-1], dp[i-2], ..., dp[i-k]).
  Common patterns: Fibonacci-like (dp[i] = dp[i-1] + dp[i-2]).

ADVANTAGES:
  ✓ Optimal solution guaranteed
  ✓ Polynomial time (usually O(n) or O(n²))
  ✓ Avoids exponential recursion
  ✓ Space can be optimized to O(1) in many cases

DISADVANTAGES:
  ✗ Requires identifying subproblem structure
  ✗ May compute unnecessary states
  ✗ Not intuitive for complex problems
  ✗ Space overhead for DP table

BEST CASE: O(n) linear scan
  → Why: Simple recurrence like Fibonacci, each state computed once
  → When: dp[i] depends only on constant number of previous states

WORST CASE: O(n²) depending on transitions
  → Why: Each state depends on all previous states
  → When: dp[i] = min/max over all j < i (like Rod Cutting)

AVERAGE TIME: O(n) for simple, O(n²) for complex
SPACE: O(n) for table, can optimize to O(1)
"""


# ============================================================================
# 📋 ALGORITHM 9: DYNAMIC PROGRAMMING - 2D
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet4/* (Knapsack 0/1)
  - worksheet5/* (Edit Distance)
  - worksheet6/* (LCS, Knapsack)
  - Problems: Knapsack, Edit Distance, LCS, Matrix Chain

ALGORITHM TYPE: Dynamic Programming

HOW IT WORKS:
  Use 2D array dp[i][j] where states depend on two parameters.
  Common pattern: dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]).
  Build table bottom-up or use memoization top-down.
  Examples: string matching, resource allocation, grid problems.

ADVANTAGES:
  ✓ Solves problems with two dimensions/parameters
  ✓ Optimal solution guaranteed
  ✓ Handles string/sequence problems well
  ✓ Can reconstruct solution path

DISADVANTAGES:
  ✗ O(n×m) time - slow for large inputs
  ✗ O(n×m) space for table
  ✗ More complex than 1D DP
  ✗ Can be memory intensive

BEST CASE: O(n×m) must fill entire table
  → Why: Every cell depends on previous cells, no shortcuts
  → When: Always - all subproblems must be solved

WORST CASE: O(n×m) must fill entire table
  → Why: Bottom-up DP computes all states regardless
  → When: Always - table must be completely filled

AVERAGE TIME: O(n×m)
SPACE: O(n×m) for table, sometimes optimized to O(min(n,m))
"""


# ============================================================================
# 📋 ALGORITHM 10: EUCLIDEAN ALGORITHM (GCD)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet0/SourceCode-ALDS1_1_B.py

ALGORITHM TYPE: Mathematical Algorithm

HOW IT WORKS:
  Find Greatest Common Divisor using repeated division: GCD(a,b) = GCD(b, a%b)
  until remainder is 0. Based on property that GCD divides both numbers and
  their difference.

ADVANTAGES:
  ✓ Very efficient O(log min(a,b))
  ✓ Simple recursive or iterative
  ✓ Ancient and well-studied (300 BC)
  ✓ Extends to Extended Euclidean

DISADVANTAGES:
  ✗ Only works for integers
  ✗ Requires modulo operation
  ✗ Not applicable to floating point

BEST CASE: O(1) when b divides a
  → Why: First modulo operation gives remainder 0
  → When: a is exact multiple of b (e.g., GCD(12, 6) = 6)

WORST CASE: O(log min(a,b)) for consecutive Fibonacci numbers
  → Why: Maximum divisions needed, worst-case input pattern
  → When: Consecutive Fibonacci numbers (e.g., GCD(89, 55))

AVERAGE TIME: O(log min(a,b))
SPACE: O(1) iterative, O(log n) recursive
"""


# ============================================================================
# 📋 ALGORITHM 11: FENWICK TREE (BINARY INDEXED TREE)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - assignment4/inversion_count_BIT.py
  - Problems: Range sum queries, cumulative frequency

ALGORITHM TYPE: Data Structure

HOW IT WORKS:
  Tree structure for efficient range queries and point updates. Each node
  stores cumulative value for a range. Uses binary representation for
  parent-child relationships. Both update and query in O(log n).

ADVANTAGES:
  ✓ O(log n) for both update and query
  ✓ Space efficient O(n)
  ✓ Simple to implement
  ✓ Good for cumulative operations

DISADVANTAGES:
  ✗ Only works for cumulative associative operations
  ✗ Less versatile than segment tree
  ✗ Not intuitive to understand
  ✗ Usually 1-indexed

BEST CASE: O(log n) per operation
  → Why: Binary tree structure always requires log n jumps
  → When: Always - tree height determines operations

WORST CASE: O(log n) per operation
  → Why: Maximum tree height is log₂(n)
  → When: Always - consistent logarithmic performance

AVERAGE TIME: O(log n) per operation
SPACE: O(n)
"""


# ============================================================================
# 📋 ALGORITHM 12: IDA* (ITERATIVE DEEPENING A*)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet9/8puzzle_IDAstar_example.py
  - Problems: 8-puzzle, state space search with heuristic

ALGORITHM TYPE: Informed Search Algorithm

HOW IT WORKS:
  Combines IDS with A* heuristic. Uses cost threshold f(n) = g(n) + h(n)
  instead of depth limit. Increases threshold to minimum f-value that exceeded
  previous threshold. Memory-efficient A*.

ADVANTAGES:
  ✓ O(bd) space like IDS
  ✓ Uses heuristic for faster search
  ✓ Optimal if heuristic is admissible
  ✓ Better than IDS for large states

DISADVANTAGES:
  ✗ Revisits nodes like IDS
  ✗ Requires good heuristic
  ✗ Slower than A* if memory available
  ✗ Complex to implement

BEST CASE: O(d) with perfect heuristic
  → Why: Heuristic guides directly to goal with no wrong turns
  → When: h(n) perfectly predicts remaining cost

WORST CASE: O(b^d) with poor heuristic
  → Why: Bad heuristic causes exploration like uninformed search
  → When: h(n) = 0 or misleading heuristic values

AVERAGE TIME: Much better than O(b^d) with good heuristic
SPACE: O(bd)
"""


# ============================================================================
# 📋 ALGORITHM 13: ITERATIVE DEEPENING SEARCH (IDS)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet9/8puzzle_IDS_example.py
  - Problems: 8-puzzle, state space search

ALGORITHM TYPE: Uninformed Search Algorithm

HOW IT WORKS:
  Combines BFS completeness with DFS space efficiency. Performs depth-limited
  DFS repeatedly with increasing limits. Explores depths 0, 1, 2, ..., d
  until solution found.

ADVANTAGES:
  ✓ Space O(bd) like DFS
  ✓ Complete and optimal like BFS
  ✓ Good when depth unknown
  ✓ No heuristic needed

DISADVANTAGES:
  ✗ Revisits nodes multiple times
  ✗ Appears wasteful
  ✗ Slower than BFS for large branching
  ✗ Not suitable for very large depth

BEST CASE: O(d) shallow solution
  → Why: Target found at shallow depth, few iterations needed
  → When: Solution exists at depth 1, 2, or 3

WORST CASE: O(b^d) deep solution
  → Why: Must repeat DFS for depths 1, 2, ..., d
  → When: Solution at maximum depth d

AVERAGE TIME: O(b^d)
SPACE: O(bd)
"""


# ============================================================================
# 📋 ALGORITHM 14: KADANE'S ALGORITHM (GREEDY/DP HYBRID)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet1/maxsum_v3.py
  - worksheet11/maxSubSum.py
  - Problems: Maximum subarray sum

ALGORITHM TYPE: Greedy/Dynamic Programming Hybrid

HOW IT WORKS:
  Scan array left to right. At each position, decide: extend current subarray
  (add element) or start new subarray (reset to current element). Track
  maximum sum seen. Can be viewed as DP or greedy.

ADVANTAGES:
  ✓ Optimal O(n) time - single pass
  ✓ Simple to implement
  ✓ Constant O(1) space
  ✓ Works with negative numbers

DISADVANTAGES:
  ✗ Only works for contiguous subarrays
  ✗ Doesn't return subarray indices without modification
  ✗ Not applicable to 2D arrays

BEST CASE: O(n) single pass
  → Why: Always scans entire array once
  → When: Always - must check every element

WORST CASE: O(n) single pass
  → Why: Cannot skip elements, needs to see all values
  → When: Always - linear scan required

AVERAGE TIME: O(n)
SPACE: O(1)
"""


# ============================================================================
# 📋 ALGORITHM 15: KRUSKAL'S ALGORITHM (MST)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet10/mst.py
  - Problems: Minimum Spanning Tree

ALGORITHM TYPE: Greedy Algorithm + Union-Find

HOW IT WORKS:
  Find minimum cost tree connecting all vertices. Sort edges by weight.
  Add edges in order if they don't create cycle. Uses Union-Find to detect
  cycles efficiently.

ADVANTAGES:
  ✓ Optimal solution guaranteed
  ✓ O(E log E) time
  ✓ Works on disconnected graphs
  ✓ Simple with Union-Find

DISADVANTAGES:
  ✗ Requires sorting all edges
  ✗ Needs Union-Find structure
  ✗ Not efficient for dense graphs
  ✗ Doesn't handle directed graphs

BEST CASE: O(E log E) sorting dominates
  → Why: Must sort all edges first, no way to skip
  → When: Always needs to sort, even for simple graphs

WORST CASE: O(E log E) sorting dominates
  → Why: Sorting complexity is constant regardless of graph structure
  → When: Always - Union-Find operations are nearly O(1)

AVERAGE TIME: O(E log E)
SPACE: O(V + E)
"""


# ============================================================================
# 📋 ALGORITHM 16: MAXIMUM SUBARRAY (DIVIDE & CONQUER)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet11/maxSubSum.py
  - Problems: Maximum subarray sum (alternative to Kadane's)

ALGORITHM TYPE: Divide & Conquer

HOW IT WORKS:
  Divide array in half. Maximum subarray is either:
  1. Entirely in left half
  2. Entirely in right half
  3. Crosses the middle
  Recursively solve left/right, calculate crossing, return maximum.

ADVANTAGES:
  ✓ Demonstrates D&C paradigm
  ✓ O(n log n) time
  ✓ No extra space
  ✓ Good for teaching

DISADVANTAGES:
  ✗ Slower than Kadane's O(n)
  ✗ More complex
  ✗ Not practical when better solution exists

BEST CASE: O(n log n)
  → Why: Must divide array log n times and process all elements
  → When: Always - divide and conquer structure is fixed

WORST CASE: O(n log n)
  → Why: Same recurrence regardless of input values
  → When: Always - balanced splits every time

AVERAGE TIME: O(n log n)
SPACE: O(log n) recursion stack
"""


# ============================================================================
# 📋 ALGORITHM 17: MERGE SORT
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - assignment4/inversion_count.py
  - Problems: Sorting, inversion counting

ALGORITHM TYPE: Divide & Conquer Sorting

HOW IT WORKS:
  Divide array in half recursively, sort each half, merge sorted halves.
  During merge, can count inversions: when element from right is smaller,
  all remaining left elements form inversions.

ADVANTAGES:
  ✓ O(n log n) guaranteed (not worst-case like quicksort)
  ✓ Stable sort
  ✓ Counts inversions efficiently
  ✓ Good for linked lists

DISADVANTAGES:
  ✗ O(n) extra space
  ✗ Not in-place
  ✗ Slower than quicksort in practice
  ✗ Overkill for small arrays

BEST CASE: O(n log n)
  → Why: Must divide log n times and merge all n elements
  → When: Always - even sorted arrays need merging

WORST CASE: O(n log n)
  → Why: Merge step always processes all elements
  → When: Always - no worst-case degradation

AVERAGE TIME: O(n log n)
SPACE: O(n) temporary array
"""


# ============================================================================
# 📋 ALGORITHM 18: SELECTION SORT
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet0/SourceCode-P2.py

ALGORITHM TYPE: Sorting Algorithm

HOW IT WORKS:
  Find minimum element in unsorted portion, swap with first element of
  unsorted portion. Repeat until array is sorted. Simple but inefficient.

ADVANTAGES:
  ✓ Simple to understand
  ✓ O(1) extra space
  ✓ Good for small arrays
  ✓ Minimum number of swaps

DISADVANTAGES:
  ✗ O(n²) time always
  ✗ Not stable
  ✗ Not adaptive
  ✗ Many comparisons

BEST CASE: O(n²) even if sorted
  → Why: Always scans remaining array for minimum
  → When: Even if array is sorted, still makes n(n-1)/2 comparisons

WORST CASE: O(n²) even if sorted
  → Why: Not adaptive - doesn't detect sorted input
  → When: Reverse sorted or random, still O(n²)

AVERAGE TIME: O(n²)
SPACE: O(1)
"""


# ============================================================================
# 📋 ALGORITHM 19: UNION-FIND (DISJOINT SET)
# ============================================================================
"""
WHERE USED IN CODEBASE:
  - worksheet10/mst.py (used in Kruskal's)

ALGORITHM TYPE: Data Structure

HOW IT WORKS:
  Maintains disjoint sets with operations: union (merge sets) and find
  (which set contains element). Uses path compression and union by rank
  for near-constant amortized time.

ADVANTAGES:
  ✓ Near O(1) amortized time per operation
  ✓ Essential for Kruskal's MST
  ✓ Efficient cycle detection
  ✓ Simple implementation

DISADVANTAGES:
  ✗ Can't split sets once merged
  ✗ Not useful for dynamic connectivity
  ✗ Requires array representation

BEST CASE: O(α(n)) amortized
  → Why: Path compression makes subsequent operations faster
  → When: Always near-constant with path compression

WORST CASE: O(α(n)) amortized
  → Why: Union by rank prevents degenerate trees
  → When: Always - optimizations guarantee good performance

AVERAGE TIME: O(α(n)) where α = inverse Ackermann (effectively O(1))
SPACE: O(n)
"""


# ============================================================================
# 🎯 ALGORITHM COMPLEXITY COMPARISON TABLE
# ============================================================================
"""
================================================================================
                    QUICK REFERENCE: UNIQUE ALGORITHMS ONLY
================================================================================

┌────────────────────────────────────┬──────────────┬─────────────┬───────────┐
│ ALGORITHM NAME                     │ TIME         │ SPACE       │ PARADIGM  │
├────────────────────────────────────┼──────────────┼─────────────┼───────────┤
│ Activity Selection                 │ O(n log n)   │ O(1)        │ Greedy    │
│ Binary Exponentiation              │ O(log n)     │ O(1)        │ D&C       │
│ Branch and Bound                   │ O(2^n)*      │ O(n)        │ Optimize  │
│ Breadth-First Search (BFS)         │ O(V+E)       │ O(V)        │ Graph     │
│ Depth-First Search (DFS)           │ O(V+E)       │ O(V)        │ Graph     │
│ DFS with Backtracking              │ O(b^d)       │ O(d)        │ Backtrack │
│ DFS with Pruning                   │ O(2^n)*      │ O(n)        │ Pruning   │
│ Dynamic Programming - 1D           │ O(n) - O(n²) │ O(n)        │ DP        │
│ Dynamic Programming - 2D           │ O(m×n)       │ O(m×n)      │ DP        │
│ Euclidean Algorithm (GCD)          │ O(log n)     │ O(1)        │ Math      │
│ Fenwick Tree (per operation)       │ O(log n)     │ O(n)        │ Data Str  │
│ IDA*                               │ O(b^d)*      │ O(bd)       │ Search    │
│ Iterative Deepening Search         │ O(b^d)       │ O(bd)       │ Search    │
│ Kadane's Algorithm                 │ O(n)         │ O(1)        │ DP/Greedy │
│ Kruskal's MST                      │ O(E log E)   │ O(V+E)      │ Greedy    │
│ Maximum Subarray (D&C)             │ O(n log n)   │ O(log n)    │ D&C       │
│ Merge Sort                         │ O(n log n)   │ O(n)        │ D&C       │
│ Selection Sort                     │ O(n²)        │ O(1)        │ Sort      │
│ Union-Find (per operation)         │ O(α(n))      │ O(n)        │ Data Str  │
└────────────────────────────────────┴──────────────┴─────────────┴───────────┘

Note: * = often much better in practice with good pruning/heuristics
      α(n) = inverse Ackermann function (effectively constant)

================================================================================
                        PARADIGM-BASED CLASSIFICATION
================================================================================
(Unique algorithms only - no problem names)

BACKTRACKING:
  • DFS with Backtracking

DATA STRUCTURES:
  • Fenwick Tree
  • Union-Find

DIVIDE & CONQUER:
  • Binary Exponentiation
  • Maximum Subarray
  • Merge Sort

DYNAMIC PROGRAMMING:
  • DP - 1D (used in: Stair Climbing, Frog, Coin Change, Rod Cutting, Tiling)
  • DP - 2D (used in: Knapsack, Edit Distance, LCS)
  • Kadane's Algorithm (hybrid)

GRAPH ALGORITHMS:
  • Breadth-First Search (BFS)
  • Depth-First Search (DFS)
  • Kruskal's MST

GREEDY ALGORITHMS:
  • Activity Selection

MATHEMATICAL:
  • Euclidean Algorithm (GCD)

OPTIMIZATION:
  • Branch and Bound
  • DFS with Pruning

SEARCH ALGORITHMS:
  • IDA*
  • Iterative Deepening Search (IDS)

SORTING:
  • Merge Sort
  • Selection Sort

================================================================================
                        KEY INSIGHTS
================================================================================

ALGORITHM vs PROBLEM:
  ✗ "Coin Change" is NOT an algorithm - it's a problem
  ✗ "Knapsack" is NOT an algorithm - it's a problem
  ✗ "Frog Problem" is NOT an algorithm - it's a problem
  
  ✓ "Dynamic Programming" IS an algorithm - it's a technique
  ✓ "Backtracking" IS an algorithm - it's a technique
  ✓ "Greedy" IS an algorithm - it's a technique

ALGORITHM VARIANTS:
  • DFS has variants: Basic DFS, DFS with Backtracking, DFS with Pruning
  • DP has variants: 1D DP, 2D DP, 3D DP
  • Search has variants: BFS, DFS, IDS, IDA*

CODEBASE MAP:
  • Total unique algorithm techniques: 19
  • Total problem examples: 50+
  • Worksheets: 0-13 (14 weeks)
  • Assignments: 1-4

================================================================================
"""

print("="*80)
print("Unique Algorithms Reference - Algorithm Techniques Only")
print("="*80)
print("Total Unique Algorithms: 19")
print("No Problem Names - Only Algorithm Techniques")
print("="*80)
