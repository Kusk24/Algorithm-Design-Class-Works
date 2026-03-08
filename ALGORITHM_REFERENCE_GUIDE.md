# Algorithm Design Reference Guide
## Complete Summary of Algorithms by Week

---

## Week 1: Maximum Sum Problem

### Algorithm: Kadane's Algorithm (Maximum Subarray Sum)

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

**Advantages**:
- Very efficient linear time solution
- Simple to implement
- Uses minimal extra space
- Works well for finding contiguous subarrays

**Disadvantages**:
- Only works for contiguous subarrays
- Doesn't track the actual subarray indices without modification
- Cannot handle all-negative arrays without special handling

**When to Use**:
- Finding maximum sum of contiguous subarray
- Stock profit problems (buy/sell once)
- When you need optimal linear time solution
- Streaming data where you process elements once

---

## Week 2: Divide and Conquer / Recursion

### Algorithm: Divide and Conquer

**Time Complexity**: O(n log n) for most problems  
**Space Complexity**: O(log n) for recursion stack

**Advantages**:
- Breaks complex problems into simpler subproblems
- Often leads to efficient solutions
- Natural recursive structure
- Can be parallelized easily

**Disadvantages**:
- Recursion overhead
- May have high space complexity due to call stack
- Not always intuitive
- Can lead to repeated computations without memoization

**When to Use**:
- Problems that can be broken into independent subproblems
- Sorting algorithms (merge sort, quick sort)
- Binary search
- Tree traversals
- Matrix operations

### Algorithm: Balance Split

**Time Complexity**: O(2^n)  
**Space Complexity**: O(n)

**Advantages**:
- Explores all possible combinations
- Guarantees finding optimal solution if it exists

**Disadvantages**:
- Exponential time complexity
- Not practical for large inputs
- High memory usage for recursion

**When to Use**:
- Small input sizes (n < 20)
- When you need exact solutions
- Partition problems with small datasets

---

## Week 3: Dynamic Programming Introduction

### Algorithm: Rod Cutting Problem (DP)

**Time Complexity**: O(n²)  
**Space Complexity**: O(n)

**Advantages**:
- Finds optimal revenue
- Polynomial time solution
- Avoids exponential brute force
- Can reconstruct solution

**Disadvantages**:
- Requires careful state definition
- Space overhead for DP table
- Overkill for small inputs

**When to Use**:
- Optimization problems with overlapping subproblems
- When greedy approach doesn't work
- Resource allocation problems
- Cutting/partitioning problems

### Algorithm: Minimum Coin Change (DP)

**Time Complexity**: O(V × n) where V = value, n = number of coins  
**Space Complexity**: O(V)

**Advantages**:
- Guarantees minimum number of coins
- Handles any coin denominations
- Efficient for reasonable values
- Classic DP example

**Disadvantages**:
- Space grows with target value
- Doesn't work if no solution exists (needs handling)
- Not suitable for very large target values

**When to Use**:
- Making change problems
- Resource allocation with discrete units
- When greedy approach fails (non-canonical coin systems)
- Unbounded knapsack variants

---

## Week 4: Memoization Techniques

### Algorithm: Top-Down DP with Memoization

**Time Complexity**: O(n × W) for knapsack, O(V × n) for coins  
**Space Complexity**: O(n × W) or O(V) + recursion stack

**Advantages**:
- Easier to code than bottom-up DP
- Only computes needed states
- Natural recursive thinking
- Avoids redundant calculations

**Disadvantages**:
- Recursion stack overhead
- May hit recursion limit
- Slightly slower than iterative DP
- Extra space for recursion

**When to Use**:
- When recursive solution is natural
- Sparse state space (not all states needed)
- During learning phase of DP
- When problem is easier to think recursively

### Algorithm: 0/1 Knapsack (Memoized)

**Time Complexity**: O(n × W)  
**Space Complexity**: O(n × W)

**Advantages**:
- Solves classic optimization problem
- Polynomial time vs exponential brute force
- Can handle reasonable input sizes
- Extensible to variations

**Disadvantages**:
- Pseudo-polynomial (depends on W)
- High space complexity
- Not suitable for very large capacities
- Integer weights required

**When to Use**:
- Resource allocation with capacity constraints
- Budget optimization problems
- Project selection with constraints
- When items can only be taken once

---

## Week 5: Edit Distance

### Algorithm: Levenshtein Distance (Edit Distance)

**Time Complexity**: O(m × n)  
**Space Complexity**: O(m × n) or O(min(m,n)) if optimized

**Advantages**:
- Measures string similarity accurately
- Handles insertions, deletions, substitutions
- Well-studied with many variants
- Space can be optimized to O(n)

**Disadvantages**:
- Quadratic time complexity
- Not suitable for very long strings
- Equal weight for all operations (may not be realistic)

**When to Use**:
- Spell checking and correction
- DNA sequence alignment
- Plagiarism detection
- Fuzzy string matching
- Auto-correct systems

---

## Week 6: Advanced Dynamic Programming

### Algorithm: Longest Common Subsequence (LCS)

**Time Complexity**: O(m × n)  
**Space Complexity**: O(m × n) or O(min(m,n)) if optimized

**Advantages**:
- Finds optimal alignment
- Works for non-contiguous sequences
- Can reconstruct actual subsequence
- Space-optimizable

**Disadvantages**:
- Quadratic complexity
- Doesn't handle gaps well
- May have multiple solutions

**When to Use**:
- Diff tools (version control)
- DNA/protein sequence alignment
- Finding common patterns
- File comparison
- Plagiarism detection

### Algorithm: Knapsack (Bottom-Up DP)

**Time Complexity**: O(n × W)  
**Space Complexity**: O(n × W) or O(W) if optimized

**Advantages**:
- Iterative (no recursion stack)
- Slightly faster than memoization
- Can optimize space to O(W)
- Clear state transitions

**Disadvantages**:
- Computes all states (even unnecessary ones)
- Less intuitive than recursive
- Pseudo-polynomial complexity

**When to Use**:
- When iterative approach is preferred
- Limited recursion depth
- Need slight performance edge
- Teaching bottom-up DP concept

---

## Week 7: Complex DP Problems

### Algorithm: M3 Tile Problem (Tiling DP)

**Time Complexity**: O(n) with DP  
**Space Complexity**: O(n)

**Advantages**:
- Elegant recursive structure
- Linear time with DP
- Demonstrates state-based DP
- Can handle complex constraints

**Disadvantages**:
- State definition can be tricky
- Problem-specific approach
- Exponential without DP

**When to Use**:
- Tiling/covering problems
- Combinatorial counting
- When states have mutual dependencies
- Constraint satisfaction problems

### Algorithm: Shoe Shopping (Optimization DP)

**Time Complexity**: O(n²) typically  
**Space Complexity**: O(n)

**Advantages**:
- Handles pairing constraints
- Finds optimal cost
- Demonstrates non-trivial DP

**Disadvantages**:
- Problem-specific
- May need careful state design
- Not generalizable easily

**When to Use**:
- Pairing/matching problems
- Discount optimization
- Scheduling with constraints
- Bundling problems

---

## Week 8: Backtracking Algorithms

### Algorithm: N-Queens Problem (Backtracking)

**Time Complexity**: O(n!)  
**Space Complexity**: O(n)

**Advantages**:
- Finds all solutions
- Memory efficient
- Explores systematically
- Can terminate early if one solution needed

**Disadvantages**:
- Exponential time complexity
- Very slow for large n (n > 15)
- Cannot be easily parallelized

**When to Use**:
- Constraint satisfaction problems
- Small board sizes
- When all solutions needed
- Puzzle solving (Sudoku, etc.)

### Algorithm: Breadth-First Search (BFS)

**Time Complexity**: O(V + E) where V = vertices, E = edges  
**Space Complexity**: O(V)

**Advantages**:
- Finds shortest path (unweighted)
- Complete (finds solution if exists)
- Optimal for unweighted graphs
- Level-by-level exploration

**Disadvantages**:
- High memory usage (stores entire level)
- Not suitable for weighted graphs
- May explore unnecessary nodes

**When to Use**:
- Shortest path in unweighted graphs
- Level-order traversal
- Finding connected components
- Web crawling
- Social network analysis (degrees of separation)

### Algorithm: Maze Running (DFS/BFS)

**Time Complexity**: O(rows × cols)  
**Space Complexity**: O(rows × cols)

**Advantages**:
- Systematically explores maze
- Guarantees finding exit if exists
- BFS finds shortest path

**Disadvantages**:
- May explore entire maze
- Memory intensive for large mazes

**When to Use**:
- Pathfinding in grids
- Robot navigation
- Game AI
- Route planning

---

## Week 9: Advanced Search Algorithms

### Algorithm: Iterative Deepening Search (IDS)

**Time Complexity**: O(b^d) where b = branching factor, d = depth  
**Space Complexity**: O(d)

**Advantages**:
- Combines DFS space efficiency with BFS optimality
- Finds shortest solution
- Memory efficient
- Complete and optimal

**Disadvantages**:
- Repeated work (visits nodes multiple times)
- Slower than BFS in practice
- Only optimal for uniform cost

**When to Use**:
- Unknown solution depth
- Limited memory
- Need shortest path
- Tree/graph search with memory constraints

### Algorithm: IDA* (Iterative Deepening A*)

**Time Complexity**: O(b^d)  
**Space Complexity**: O(d)

**Advantages**:
- Memory efficient (like IDS)
- Uses heuristic for efficiency
- Optimal if heuristic is admissible
- Better than IDS with good heuristic

**Disadvantages**:
- Revisits nodes
- Requires good heuristic
- Slower than A* in practice
- Heuristic overhead

**When to Use**:
- Memory-constrained environments
- 8-puzzle, 15-puzzle problems
- When A* uses too much memory
- Game AI with limited resources

### Algorithm: Uniform Cost Search (UCS)

**Time Complexity**: O((V + E) log V) with priority queue  
**Space Complexity**: O(V)

**Advantages**:
- Finds optimal solution for weighted graphs
- Complete and optimal
- Doesn't require heuristic
- Handles arbitrary non-negative weights

**Disadvantages**:
- Memory intensive
- Slow without heuristic
- Requires priority queue
- Not as efficient as A*

**When to Use**:
- Weighted graphs
- When heuristic not available
- Need guaranteed optimal path
- Network routing

---

## Week 10: Greedy Algorithms

### Algorithm: Activity Selection (Greedy)

**Time Complexity**: O(n log n) due to sorting  
**Space Complexity**: O(1) or O(n) for sorting

**Advantages**:
- Very efficient
- Simple to implement
- Optimal for activity selection
- Intuitive approach

**Disadvantages**:
- Only works for specific problems
- Not always optimal (must prove it)
- Requires careful problem analysis

**When to Use**:
- Interval scheduling
- Resource allocation
- Meeting room problems
- Job scheduling
- When greedy choice property holds

### Algorithm: Minimum Spanning Tree (Kruskal's/Prim's)

**Time Complexity**: O(E log E) for Kruskal's, O(E log V) for Prim's  
**Space Complexity**: O(V + E)

**Advantages**:
- Efficient for sparse graphs
- Guaranteed optimal
- Clear correctness proof
- Practical applications

**Disadvantages**:
- Requires sorting edges (Kruskal's)
- Needs disjoint-set data structure
- Not applicable to directed graphs

**When to Use**:
- Network design (minimize cable/pipe)
- Clustering problems
- Approximation algorithms
- Road/utility network construction

### Algorithm: Union-Find (Disjoint Sets)

**Time Complexity**: O(α(n)) ≈ O(1) per operation with optimizations  
**Space Complexity**: O(n)

**Advantages**:
- Near-constant time operations
- Simple to implement
- Essential for Kruskal's algorithm
- Efficient for connectivity queries

**Disadvantages**:
- Limited to specific problems
- Requires path compression for efficiency
- Not intuitive initially

**When to Use**:
- Kruskal's MST algorithm
- Connected components
- Network connectivity
- Cycle detection in graphs

---

## Week 11: Divide and Conquer (Advanced)

### Algorithm: Fast Exponentiation (Binary Exponentiation)

**Time Complexity**: O(log n)  
**Space Complexity**: O(log n) for recursive, O(1) for iterative

**Advantages**:
- Logarithmic time vs linear
- Essential for cryptography
- Works for large exponents
- Can be made iterative

**Disadvantages**:
- More complex than naive approach
- May overflow without modular arithmetic
- Recursion overhead

**When to Use**:
- Modular exponentiation (cryptography)
- Matrix exponentiation
- Computing large powers
- RSA encryption
- Fibonacci computation

### Algorithm: Maximum Subarray (Divide and Conquer)

**Time Complexity**: O(n log n)  
**Space Complexity**: O(log n)

**Advantages**:
- Demonstrates divide and conquer
- Elegant recursive structure
- Educational value
- Parallelizable

**Disadvantages**:
- Slower than Kadane's O(n) algorithm
- More complex to implement
- Higher space complexity

**When to Use**:
- Teaching divide and conquer
- When parallelization is needed
- Historical/educational purposes
- (Note: Kadane's is preferred in practice)

---

## Week 12: Branch and Bound

### Algorithm: Branch and Bound (Knapsack)

**Time Complexity**: O(2^n) worst case, much better in practice  
**Space Complexity**: O(n) for recursion

**Advantages**:
- Better than brute force (pruning)
- Finds optimal solution
- Can terminate early
- Uses bounds to prune

**Disadvantages**:
- Still exponential worst case
- Complex to implement
- Requires good bounding function
- Not suitable for large n

**When to Use**:
- Exact solutions for NP-hard problems
- Small to medium input sizes
- When approximation not acceptable
- Traveling Salesman Problem
- 0/1 Knapsack with high accuracy needs

### Algorithm: DFS with Pruning

**Time Complexity**: O(2^n) worst case, better with pruning  
**Space Complexity**: O(n)

**Advantages**:
- Reduces search space significantly
- Memory efficient (DFS)
- Can find optimal solutions
- Pruning improves average case

**Disadvantages**:
- Still exponential complexity
- Pruning strategy problem-specific
- May miss solutions if pruning incorrect

**When to Use**:
- Optimization problems
- When branch and bound overhead too high
- Constraint satisfaction
- Game tree search

---

## Week 13: Graph Traversal Applications

### Algorithm: BFS for State Space Search (Flappy Bird)

**Time Complexity**: O(H × T) where H = height, T = time intervals  
**Space Complexity**: O(H × T)

**Advantages**:
- Models game as state space
- Finds shortest solution
- Complete algorithm
- Natural for level-based games

**Disadvantages**:
- Memory intensive for large spaces
- May explore many states
- Requires good state representation

**When to Use**:
- Game AI
- State space search
- Shortest path in games
- Planning problems
- Robot motion planning

### Algorithm: Connected Components (DFS/BFS)

**Time Complexity**: O(rows × cols) for grid  
**Space Complexity**: O(rows × cols)

**Advantages**:
- Identifies separate regions
- Works for various graph types
- Can count components efficiently
- Useful for many applications

**Disadvantages**:
- Requires full graph traversal
- Memory for visited tracking
- May need repeated searches

**When to Use**:
- Image processing (finding blobs)
- Network analysis
- Island counting problems
- Graph connectivity
- Social network clustering
- Flood fill algorithms

---

## General Algorithm Selection Guide

### Use DP When:
- Problem has overlapping subproblems
- Optimal substructure exists
- Need exact optimal solution
- Polynomial solution possible

### Use Greedy When:
- Greedy choice property holds
- Local optimum leads to global optimum
- Need fast solution
- Can prove correctness

### Use Backtracking When:
- Need all solutions
- Constraint satisfaction
- Small input size
- Pruning can help significantly

### Use Divide and Conquer When:
- Problem divisible into independent subproblems
- Combine step is efficient
- Parallelization needed
- Clean recursive structure

### Use Graph Algorithms When:
- Relationship between entities
- Network problems
- Pathfinding needed
- Connectivity matters

---

## Complexity Cheat Sheet

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Kadane's | O(n) | O(1) |
| Merge Sort | O(n log n) | O(n) |
| Coin Change DP | O(V × n) | O(V) |
| Edit Distance | O(m × n) | O(m × n) |
| LCS | O(m × n) | O(m × n) |
| 0/1 Knapsack | O(n × W) | O(n × W) |
| BFS | O(V + E) | O(V) |
| DFS | O(V + E) | O(V) |
| IDS | O(b^d) | O(d) |
| Kruskal's MST | O(E log E) | O(V + E) |
| Activity Selection | O(n log n) | O(1) |
| Fast Exponentiation | O(log n) | O(log n) |
| N-Queens | O(n!) | O(n) |
| Branch & Bound | O(2^n)* | O(n) |

*With pruning, average case much better

---

## Final Notes

- Always analyze your specific problem constraints
- Consider input size when choosing algorithm
- Time vs space tradeoffs are common
- Optimization is often problem-specific
- Understanding multiple approaches helps choose the best tool
- Practice implementing these algorithms for mastery

**Created**: March 1, 2026  
**Course**: Algorithm Design  
**Week Coverage**: Weeks 1-13
