"""
================================================================================
📚 ALGORITHM DESIGN - COMPLETE TABLE OF CONTENTS
================================================================================
Comprehensive index of all algorithm problems across worksheets and assignments
Problem Name → Algorithm Type Used
================================================================================
"""

# ============================================================================
# 📋 WORKSHEET 0: Introduction
# ============================================================================
"""
1. GCD (Greatest Common Divisor)
   Algorithm: Euclidean Algorithm (Divide & Conquer)
   Reason: Efficiently finds GCD by repeatedly dividing and taking remainders,
           reducing problem size exponentially (O(log min(a,b)))
   File: worksheet0/SourceCode-ALDS1_1_B.py

2. Selection Sort
   Algorithm: Selection Sort (Sorting)
   Reason: Simple sorting algorithm for teaching basic array manipulation and
           comparison operations, though O(n²) complexity
   File: worksheet0/SourceCode-P2.py
"""

# ============================================================================
# 📋 WORKSHEET 1: Maximum Sum Problem
# ============================================================================
"""
1. Maximum Subarray Sum (Brute Force)
   Algorithm: Brute Force - O(n³)
   Reason: Check all possible subarrays to understand the problem structure,
           demonstrates why optimization is needed
   File: worksheet1/maxsum_v1.py

2. Maximum Subarray Sum (Optimized)
   Algorithm: Prefix Sum - O(n²)
   Reason: Uses prefix sums to avoid recalculating subarray sums repeatedly,
           reducing one level of nested loops
   File: worksheet1/maxsum_v2.py

3. Maximum Subarray Sum (Optimal)
   Algorithm: Kadane's Algorithm - O(n)
   Reason: Optimal solution using dynamic programming principle - tracks running
           maximum by deciding to extend or restart subarray at each position
   File: worksheet1/maxsum_v3.py
"""

# ============================================================================
# 📋 WORKSHEET 2: Divide and Conquer / Recursion
# ============================================================================
"""
1. Balance Split
   Algorithm: Backtracking / Exhaustive Search
   Reason: Must explore all possible subset combinations to find balanced split,
           no greedy shortcut exists for exact equality constraint
   File: worksheet2/balanceSplit.py

2. Task 1 Problem
   Algorithm: Divide and Conquer
   Reason: Problem can be broken into independent subproblems, solved recursively,
           and combined for final solution
   File: worksheet2/task1.py

3. Task 2 Problem
   Algorithm: Recursive Approach
   Reason: Problem has natural recursive structure where solution depends on
           smaller instances of the same problem
   File: worksheet2/task2.py
"""

# ============================================================================
# 📋 WORKSHEET 3: Dynamic Programming Introduction
# ============================================================================
"""
1. Rod Cutting Problem (Recursive)
   Algorithm: Plain Recursion (Exponential)
   Reason: Shows naive approach with overlapping subproblems, motivates need
           for memoization/DP optimization
   File: worksheet3/maxRev-v1.py

2. Rod Cutting Problem (DP)
   Algorithm: Dynamic Programming - O(n²)
   Reason: Has optimal substructure (max revenue = max of all cut choices) and
           overlapping subproblems, perfect for DP optimization
   File: worksheet3/maxRev-v2.py

3. Minimum Coin Change (Recursive)
   Algorithm: Plain Recursion (Exponential)
   Reason: Demonstrates the exponential time problem when recursively trying
           all coin combinations without memoization
   File: worksheet3/minCoin-v1.py

4. Minimum Coin Change (DP)
   Algorithm: Dynamic Programming - O(V×n)
   Reason: Classic DP problem - minimum coins for value V = 1 + min(coins for V-c),
           overlapping subproblems make DP essential
   File: worksheet3/minCoin-v2.py
"""

# ============================================================================
# 📋 WORKSHEET 4: Memoization Techniques
# ============================================================================
"""
1. 0/1 Knapsack (Backtracking)
   Algorithm: Backtracking - O(2^n)
   Reason: Explores all include/exclude combinations for items, shows exponential
           complexity without optimization
   File: worksheet4/knapsack_v1.py

2. 0/1 Knapsack (Memoization)
   Algorithm: Top-Down DP with Memoization - O(n×W)
   Reason: Caches recursive results indexed by (item, remaining_weight) to avoid
           recomputing same subproblems, reduces exponential to polynomial
   File: worksheet4/knapsack_v2.py

3. 0/1 Knapsack (Bottom-Up DP)
   Algorithm: Bottom-Up Dynamic Programming - O(n×W)
   Reason: Iteratively builds solution table from base cases up, more space-efficient
           than recursion and avoids call stack overhead
   File: worksheet4/knapsack_v3.py
"""

# ============================================================================
# 📋 WORKSHEET 5: Edit Distance
# ============================================================================
"""
1. Edit Distance (Recursive)
   Algorithm: Plain Recursion (Exponential)
   Reason: Shows naive approach comparing characters recursively with three choices
           (insert/delete/replace), demonstrates exponential blow-up
   File: worksheet5/edit_distance_v1.py

2. Edit Distance (DP)
   Algorithm: Levenshtein Distance DP - O(m×n)
   Reason: 2D DP table stores min edits for all prefix pairs, optimal substructure:
           edit(i,j) = min of three edit operations on smaller prefixes
   File: worksheet5/edit_distance_v2.py

3. Minimum Coin Change V3
   Algorithm: Dynamic Programming - O(V×n)
   Reason: Refined DP implementation with better space/time optimization techniques
   File: worksheet5/minCoin-v3.py
"""

# ============================================================================
# 📋 WORKSHEET 6: Advanced Dynamic Programming
# ============================================================================
"""
1. Longest Common Subsequence (LCS)
   Algorithm: Dynamic Programming - O(m×n)
   Reason: Must compare all substring pairs to find longest common subsequence,
           DP table naturally captures optimal substructure of the problem
   File: worksheet6/LCS.py

2. Knapsack Memoization (V2)
   Algorithm: Top-Down DP with Memoization
   Reason: Demonstrates advanced memoization with dictionary/hashmap for caching
           (item, weight) state pairs
   File: worksheet6/v2_mm.py

3. Knapsack Bottom-Up (V3)
   Algorithm: Bottom-Up Dynamic Programming
   Reason: More memory-efficient iterative approach, can optimize to 1D array
           since only previous row is needed
   File: worksheet6/v3_dp.py

4. Knapsack Visualization
   Algorithm: DP with Visualization
   Reason: Educational tool to visualize how DP table is filled and decisions
           are made at each step
   File: worksheet6/knapsack_visualization.py
"""

# ============================================================================
# 📋 WORKSHEET 7: Complex DP Problems
# ============================================================================
"""
1. M3 Tile Problem (Brute Force)
   Algorithm: Backtracking - O(3^n)
   Reason: Tries all possible tile placement combinations (3 choices per position),
           shows exponential growth without memoization
   File: worksheet7/m3tileBF.py

2. M3 Tile Problem (Memoization)
   Algorithm: Top-Down DP with Memoization
   Reason: Caches number of ways to tile length n, avoids recalculating same
           subproblems in recursive formula
   File: worksheet7/m3tileMM.py

3. M3 Tile Problem (DP)
   Algorithm: Bottom-Up Dynamic Programming - O(n)
   Reason: Linear DP using recurrence relation: ways[n] = ways[n-1] + ways[n-3],
           optimal for this counting problem
   File: worksheet7/m3tileDP.py

4. Shoe Shopping
   Algorithm: Dynamic Programming - O(n²)
   Reason: Optimization problem with discount constraints, DP tracks minimum cost
           for buying first i shoes considering discount rules
   File: worksheet7/shoeshopping.py
"""

# ============================================================================
# 📋 WORKSHEET 8: Backtracking Algorithms
# ============================================================================
"""
1. N-Queens Problem
   Algorithm: Backtracking - O(n!)
   Reason: Must satisfy complex constraints (no two queens attack each other),
           backtracking systematically explores valid placements with pruning
   File: worksheet8/n_queens.py

2. Print Queens Solutions
   Algorithm: Backtracking with Solution Output
   Reason: Extension of N-Queens that outputs all valid board configurations,
           demonstrates solution enumeration with backtracking
   File: worksheet8/print_queens.py

3. Maze Running
   Algorithm: DFS/BFS - O(rows×cols)
   Reason: BFS finds shortest path in unweighted grid, DFS explores all paths,
           both visit each cell at most once
   File: worksheet8/maze_running.py

4. Example Python Class
   Algorithm: Object-Oriented Programming Example
   Reason: Demonstrates OOP principles for structuring algorithm implementations
           with classes and methods
   File: worksheet8/example_python_class.py
"""

# ============================================================================
# 📋 WORKSHEET 9: Advanced Search Algorithms
# ============================================================================
"""
1. 8-Puzzle with Iterative Deepening Search (IDS)
   Algorithm: IDS - O(b^d)
   Reason: Combines DFS's space efficiency with BFS's optimality guarantee,
           ideal when solution depth is unknown
   File: worksheet9/8puzzle_IDS_example.py

2. 8-Puzzle with IDA* (A* with Iterative Deepening)
   Algorithm: IDA* - O(b^d) with heuristic
   Reason: Uses heuristic (Manhattan distance) to guide search and prune paths,
           memory-efficient alternative to A* for large state spaces
   File: worksheet9/8puzzle_IDAstar_example.py

3. Map Template
   Algorithm: Graph Representation
   Reason: Provides standard data structure for representing graphs (adjacency
           list/matrix) used in search algorithms
   File: worksheet9/map_template.py

4. Simple Priority Queue
   Algorithm: Priority Queue Implementation
   Reason: Essential data structure for UCS, Dijkstra's, A* - always processes
           lowest-cost node first
   File: worksheet9/simplePriorityQueue.py
"""

# ============================================================================
# 📋 WORKSHEET 10: Greedy Algorithms
# ============================================================================
"""
1. Activity Selection (Version 1)
   Algorithm: Greedy - O(n log n)
   Reason: Greedy choice property holds - always pick earliest finishing activity,
           provably optimal and much faster than DP O(n²) approach
   File: worksheet10/activity_selection_v1.py

2. Activity Selection (Version 2)
   Algorithm: Greedy - O(n log n)
   Reason: Alternative greedy implementation (possibly with different tie-breaking
           or sorting approach), demonstrates greedy strategy flexibility
   File: worksheet10/activity_selection_v2.py

3. Minimum Spanning Tree (MST)
   Algorithm: Kruskal's / Prim's - O(E log E)
   Reason: Greedy algorithms that always choose minimum weight edge (Kruskal's)
           or minimum edge from tree (Prim's), proven optimal for MST
   File: worksheet10/mst.py
"""

# ============================================================================
# 📋 WORKSHEET 11: Divide and Conquer (Advanced)
# ============================================================================
"""
1. Fast Exponentiation (Iterative)
   Algorithm: Binary Exponentiation (Iterative) - O(log n)
   Reason: Uses binary representation of exponent to reduce O(n) multiplications
           to O(log n), iterative version avoids recursion overhead
   File: worksheet11/exponentiation_v1.py

2. Fast Exponentiation (Recursive)
   Algorithm: Binary Exponentiation (Recursive) - O(log n)
   Reason: Elegant recursive approach: x^n = (x^(n/2))² if n even, else x·x^(n-1),
           halves problem size each step
   File: worksheet11/exponentiation_v2.py

3. Maximum Subarray (Divide & Conquer)
   Algorithm: Divide and Conquer - O(n log n)
   Reason: Splits array in half, maximum subarray is either in left, right, or
           crosses middle - demonstrates D&C with merging step
   File: worksheet11/maxSubSum.py
"""

# ============================================================================
# 📋 WORKSHEET 12: Branch and Bound / Pruning
# ============================================================================
"""
1. DFS Knapsack (Plain)
   Algorithm: DFS Backtracking - O(2^n)
   Reason: Baseline exhaustive search exploring all include/exclude combinations,
           establishes comparison point for optimizations
   File: worksheet12/dfsKnapsack.py

2. DFS Knapsack with Pruning
   Algorithm: DFS with Pruning - O(2^n) optimized
   Reason: Adds feasibility pruning (weight constraint) and optimality pruning
           (current best), cuts search tree significantly in practice
   File: worksheet12/dfsKnapsack_pruning.py

3. DFS Knapsack Branch and Bound
   Algorithm: Branch and Bound - O(2^n) with bounding
   Reason: Calculates upper bound (fractional knapsack relaxation) to prune
           branches that cannot improve best solution
   File: worksheet12/dfsKnapsack_branch_and_bound.py

4. Knapsack Bound
   Algorithm: Branch and Bound with Upper Bound
   Reason: Implements tight upper bound calculation for aggressive pruning,
           demonstrates theoretical vs practical exponential complexity
   File: worksheet12/KnapsackBound.py
"""

# ============================================================================
# 📋 WORKSHEET 13: Graph Traversal Applications
# ============================================================================
"""
1. Flappy Bird (State Space Search)
   Algorithm: BFS for State Space - O(height×time)
   Reason: Models game as state space (position, time), BFS finds shortest path
           (minimum moves) from start to goal state
   File: worksheet13/flappybird.py
"""

# ============================================================================
# 📋 ASSIGNMENT 1: Basic Algorithms
# ============================================================================
"""
1. Assignment 1 Problems
   Algorithm: Various (DP/Recursion)
   Reason: Mix of problems to practice basic DP concepts and recursive thinking,
           foundational exercises for course
   File: assignment1/assginment1.py

2. Dynamic Programming Problems
   Algorithm: Dynamic Programming
   Reason: Focused DP practice problems to master memoization and tabulation
           techniques covered in early weeks
   File: assignment1/dp.py
"""

# ============================================================================
# 📋 ASSIGNMENT 2: Stair Climbing Variations
# ============================================================================
"""
1. Stair Climbing V1
   Algorithm: Plain Recursion
   Reason: Simple recursive formula (ways[n] = ways[n-1] + ways[n-2]) but
           exponential time due to repeated subproblems
   File: assignment2/stair-climbing-v1.py

2. Stair Climbing V2
   Algorithm: Memoization (Top-Down DP)
   Reason: Adds dictionary to cache results, reduces exponential to linear time
           while keeping intuitive recursive structure
   File: assignment2/stair-climbing-v2.py

3. Stair Climbing V3
   Algorithm: Bottom-Up Dynamic Programming
   Reason: Iterative solution building from base cases (similar to Fibonacci),
           most efficient with O(1) space possible
   File: assignment2/stair-climbing-v3.py
"""

# ============================================================================
# 📋 ASSIGNMENT 3: AtCoder Problems
# ============================================================================
"""
1. Frog 1 Problem
   Algorithm: Dynamic Programming - O(n)
   Reason: Frog can jump 1 or 2 stones, DP tracks minimum cost to reach each stone,
           classic 1D DP: dp[i] = min(dp[i-1] + cost1, dp[i-2] + cost2)
   File: assignment3/frog1.py
"""

# ============================================================================
# 📋 ASSIGNMENT 4: Advanced Problems
# ============================================================================
"""
1. Assignment 4 Main Problems
   Algorithm: Various (DP/Divide & Conquer)
   Reason: Collection of advanced problems testing mastery of multiple algorithmic
           paradigms covered throughout semester
   File: assignment4/assignment4.py

2. Inversion Count (Brute Force)
   Algorithm: Brute Force - O(n²)
   Reason: Checks all pairs (i,j) where i<j to count inversions, demonstrates
           why optimization is critical for large inputs
   File: assignment4/inversion_count_bruteforce.py

3. Inversion Count (Optimal)
   Algorithm: Merge Sort - O(n log n)
   Reason: Counts inversions during merge step of merge sort - when element from
           right subarray is smaller, all remaining left elements form inversions
   File: assignment4/inversion_count.py

4. Inversion Count (Binary Indexed Tree)
   Algorithm: Fenwick Tree (BIT) - O(n log n)
   Reason: Alternative approach using BIT for range queries, counts smaller elements
           to the right efficiently with update and query operations
   File: assignment4/inversion_count_BIT.py

5. Questions 3 & 4
   Algorithm: Frog Problem (DP), Inversion Count (Merge Sort)
   Reason: Problem statements and explanations for assignment questions,
           documents requirements and solution approaches
   File: assignment4/questions_of_3&4.py
"""

# ============================================================================
# 📋 ALGORITHM REFERENCE FILES
# ============================================================================
"""
1. Algorithm Reference Guide (Markdown)
   Description: Complete guide with all 27 algorithms
   Reason: Comprehensive reference organized by weeks, includes complexity analysis,
           pseudocode, and examples for exam preparation
   File: ALGORITHM_REFERENCE_GUIDE.md

2. Algorithm Detail (Python)
   Description: Python commented version with 27 algorithms
   Reason: Executable Python implementations with detailed comments explaining
           each algorithm's logic and design decisions
   File: algorithms_detail.py

3. Algorithm Examples
   Description: 50 exam problems with implementations
   Reason: Practice problems covering all major algorithm types, structured like
           exam questions with complete solutions
   File: algorithms_example.py

4. Algorithm Comparison
   Description: Side-by-side algorithm comparisons
   Reason: Shows different approaches to same problems (e.g., recursion vs DP),
           helps understand trade-offs and when to use each
   File: comparison.py

5. Algorithm Comparison Guide (Markdown)
   Description: Text comparison of algorithm choices
   Reason: Conceptual explanations of WHY to choose one algorithm over another,
           decision-making guidelines without code details
   File: algorithm_comparison_guide.md
"""

# ============================================================================
# 📋 MIDTERM REFERENCE FILES
# ============================================================================
"""
1. Algorithm Summary
   Description: Quick reference for midterm
   Reason: Condensed cheat sheet with key algorithms, complexities, and formulas
           for rapid review before midterm exam
   File: midterm/ALGORITHM_SUMMARY.py

2. Memoization to DP Conversion Guide
   Description: Guide for converting recursion to DP
   Reason: Step-by-step methodology for transforming top-down recursive solutions
           to bottom-up DP, critical skill for exams
   File: midterm/MEMOIZATION_TO_DP_CONVERSION_GUIDE.py

3. Optimization Techniques Explained
   Description: Various optimization strategies
   Reason: Explains pruning, bounding, space optimization, and other techniques
           to improve algorithm efficiency
   File: midterm/OPTIMIZATION_TECHNIQUES_EXPLAINED.py
"""

# ============================================================================
# ============================================================================
# 📊 COMPLETE SUMMARY
# ============================================================================
# ============================================================================

"""
================================================================================
                        QUICK ALGORITHM REFERENCE
================================================================================

📌 BY ALGORITHM TYPE:
=====================

DYNAMIC PROGRAMMING (DP):
  • Kadane's Algorithm (Maximum Subarray)
  • Rod Cutting Problem
  • Minimum Coin Change
  • 0/1 Knapsack
  • Edit Distance (Levenshtein)
  • Longest Common Subsequence (LCS)
  • M3 Tile Problem
  • Shoe Shopping
  • Stair Climbing
  • Frog Problem

GRAPH ALGORITHMS:
  • Breadth-First Search (BFS)
  • Depth-First Search (DFS)
  • Maze Running
  • Iterative Deepening Search (IDS)
  • IDA* (Iterative Deepening A*)
  • Uniform Cost Search (UCS)
  • Connected Components
  • Minimum Spanning Tree (Kruskal's/Prim's)

GREEDY ALGORITHMS:
  • Activity Selection
  • Minimum Spanning Tree

DIVIDE & CONQUER:
  • Maximum Subarray (D&C approach)
  • Fast Exponentiation (Binary)
  • Merge Sort (for Inversion Count)

BACKTRACKING:
  • N-Queens Problem
  • Balance Split
  • Subset Generation

OPTIMIZATION TECHNIQUES:
  • Branch and Bound
  • DFS with Pruning
  • Memoization (Top-Down)
  • Tabulation (Bottom-Up)

DATA STRUCTURES:
  • Union-Find (Disjoint Sets)
  • Priority Queue
  • Fenwick Tree (Binary Indexed Tree)

================================================================================

⏱️  BY TIME COMPLEXITY:
========================

O(n):
  • Kadane's Algorithm
  • M3 Tiling (DP)
  • Frog Problem
  • Union-Find (amortized)

O(n log n):
  • Merge Sort (Inversion Count)
  • Fast Exponentiation
  • Activity Selection
  • MST (Kruskal's)
  • Maximum Subarray (D&C)

O(n²) or O(m×n):
  • Rod Cutting
  • Shoe Shopping
  • Edit Distance
  • LCS
  • Prefix Sum Approach

O(n×W) - Pseudo-polynomial:
  • 0/1 Knapsack (all variants)

O(V+E) - Graph algorithms:
  • BFS
  • DFS
  • Connected Components

O(2^n) - Exponential:
  • Balance Split (Backtracking)
  • Branch and Bound
  • Subset problems

O(n!) - Factorial:
  • N-Queens

================================================================================

📚 BY WORKSHEET/ASSIGNMENT:
===========================

Worksheet 1:  Maximum Sum (Kadane's)
Worksheet 2:  Divide & Conquer, Balance Split
Worksheet 3:  Rod Cutting, Coin Change (DP Introduction)
Worksheet 4:  Knapsack with Memoization
Worksheet 5:  Edit Distance
Worksheet 6:  LCS, Knapsack Bottom-Up
Worksheet 7:  M3 Tiling, Shoe Shopping
Worksheet 8:  N-Queens, BFS, Maze
Worksheet 9:  IDS, IDA*, UCS
Worksheet 10: Activity Selection, MST, Union-Find
Worksheet 11: Fast Exponentiation, Max Subarray
Worksheet 12: Branch & Bound, Pruning
Worksheet 13: State Space Search, Connected Components

Assignment 1: Basic DP
Assignment 2: Stair Climbing (3 versions)
Assignment 3: Frog Problem
Assignment 4: Inversion Count (3 approaches)

================================================================================

🎯 PROBLEM PATTERNS:
====================

Array Maximum/Minimum:
  → Kadane's Algorithm

Optimization with Constraints:
  → Dynamic Programming (Knapsack, Rod Cutting, Coin Change)

Pathfinding:
  → BFS (shortest path), DFS (all paths), IDS, IDA*

String Similarity:
  → Edit Distance, LCS

Counting Ways/Combinations:
  → Dynamic Programming (Tiling, Stair Climbing)

Scheduling/Selection:
  → Greedy (Activity Selection)

Graph Connectivity:
  → DFS, BFS, Union-Find

Constraint Satisfaction:
  → Backtracking (N-Queens)

Exact Optimal Solution:
  → Branch and Bound

Array Inversions:
  → Merge Sort, Fenwick Tree

================================================================================

💡 QUICK TIPS:
==============

1. If problem has "maximum/minimum" + "contiguous subarray"
   → Try Kadane's Algorithm first

2. If problem has "count ways" or "minimum steps"
   → Consider Dynamic Programming

3. If problem requires "all solutions"
   → Use Backtracking

4. If problem is on a "graph" or "grid"
   → Use BFS (shortest path) or DFS (exploration)

5. If problem has "overlapping subproblems"
   → Use Memoization or DP

6. If "greedy choice property" can be proven
   → Use Greedy Algorithm (faster)

7. If dealing with "tree structures" or "recursive definitions"
   → Consider Divide & Conquer

8. If need "optimal solution" from exponential space
   → Use Branch and Bound

================================================================================

📖 STUDY RECOMMENDATIONS:
=========================

ESSENTIAL (Must Master):
  ✓ Kadane's Algorithm
  ✓ Dynamic Programming basics (Knapsack, Coin Change)
  ✓ BFS and DFS
  ✓ Backtracking (N-Queens)

IMPORTANT (Common in Exams):
  ✓ Edit Distance
  ✓ LCS
  ✓ Memoization techniques
  ✓ Activity Selection (Greedy)

ADVANCED (Good to Know):
  ✓ IDS and IDA*
  ✓ Branch and Bound
  ✓ Union-Find
  ✓ Fast Exponentiation

================================================================================
                    Total: 60+ Algorithm Implementations
                    Across 13 Worksheets + 4 Assignments
================================================================================

Created: March 2026
Course: Algorithm Design
Coverage: Complete Semester (Weeks 1-13)

Good luck with your studies! 🚀
"""

print("=" * 80)
print("Algorithm Table of Contents Loaded")
print("=" * 80)
print("Total Problems Indexed: 60+")
print("Worksheets: 13 (Week 0-13)")
print("Assignments: 4")
print("Reference Files: 8")
print("=" * 80)
