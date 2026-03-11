"""
================================================================================
ALGORITHM COMPARISONS - Side-by-Side Analysis
================================================================================
Author: Algorithm Design Course
Purpose: Compare similar algorithms to understand trade-offs and when to use each

This file contains detailed comparisons of algorithms that solve similar problems
but with different approaches, complexities, and use cases.

TABLE OF CONTENTS:
==================
1. BFS vs DFS (Graph Traversal)
2. Kadane's vs Divide & Conquer (Maximum Subarray)
3. Top-Down (Memoization) vs Bottom-Up DP
4. Greedy vs Dynamic Programming
5. IDS vs BFS vs DFS (Search Algorithms)
6. IDA* vs A* vs IDS (Heuristic Search)
7. Kruskal's vs Prim's (Minimum Spanning Tree)
8. Backtracking vs Dynamic Programming
9. Branch and Bound vs Backtracking
10. Iterative vs Recursive Approaches

Each comparison includes:
- Algorithm implementations
- Complexity analysis
- Actual runtime tests
- When to use each
- Advantages and disadvantages
================================================================================
"""

import time
import random
from collections import deque
from typing import List, Set, Tuple, Dict
import heapq

# ============================================================================
# COMPARISON 1: BFS vs DFS (Graph Traversal)
# ============================================================================

print("=" * 80)
print("COMPARISON 1: BFS vs DFS")
print("=" * 80)

"""
SCENARIO: Finding path in a graph

KEY DIFFERENCES:
- BFS: Uses queue (FIFO) - explores level by level
- DFS: Uses stack/recursion (LIFO) - explores depth first

TIME COMPLEXITY: Both O(V + E)
SPACE COMPLEXITY: 
  - BFS: O(V) - queue can hold entire level
  - DFS: O(h) - recursion depth h (height of tree)
"""

def bfs_path(graph, start, goal):
    """BFS finds shortest path (unweighted)"""
    queue = deque([(start, [start])])
    visited = {start}
    nodes_explored = 0
    
    while queue:
        node, path = queue.popleft()
        nodes_explored += 1
        
        if node == goal:
            return path, nodes_explored
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None, nodes_explored

def dfs_path(graph, start, goal):
    """DFS may find longer path but uses less memory"""
    visited = set()
    nodes_explored = [0]  # Use list to modify in nested function
    
    def dfs_helper(node, path):
        nodes_explored[0] += 1
        if node == goal:
            return path
        
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result = dfs_helper(neighbor, path + [neighbor])
                if result:
                    return result
        
        return None
    
    result = dfs_helper(start, [start])
    return result, nodes_explored[0]

# Test graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("\nGraph structure:")
print("    A")
print("   / \\")
print("  B   C")
print(" /|   |")
print("D E - F")
print()

print("Finding path from A to F:")
bfs_result, bfs_explored = bfs_path(graph, 'A', 'F')
dfs_result, dfs_explored = dfs_path(graph, 'A', 'F')

print(f"BFS: Path = {bfs_result}, Nodes explored = {bfs_explored}")
print(f"DFS: Path = {dfs_result}, Nodes explored = {dfs_explored}")
print(f"\nBFS path length: {len(bfs_result)} (optimal)")
print(f"DFS path length: {len(dfs_result)} (may be longer)")

print("\n✓ WHEN TO USE:")
print("BFS:")
print("  - Need shortest path (unweighted)")
print("  - Solution likely near start")
print("  - Need to explore by levels")
print("\nDFS:")
print("  - Memory constrained")
print("  - Solution likely deep in tree")
print("  - Exploring all paths")
print("  - Game trees, backtracking")
print()


# ============================================================================
# COMPARISON 2: Kadane's vs Divide & Conquer (Maximum Subarray)
# ============================================================================

print("=" * 80)
print("COMPARISON 2: KADANE'S vs DIVIDE & CONQUER (Maximum Subarray)")
print("=" * 80)

"""
SCENARIO: Finding maximum sum of contiguous subarray

KEY DIFFERENCES:
- Kadane's: One-pass, greedy approach
- Divide & Conquer: Split, solve, merge

TIME COMPLEXITY: 
  - Kadane's: O(n)
  - Divide & Conquer: O(n log n)

SPACE COMPLEXITY:
  - Kadane's: O(1)
  - Divide & Conquer: O(log n) for recursion stack
"""

def kadane_algorithm(arr):
    """Kadane's Algorithm - O(n) time, O(1) space"""
    max_sum = arr[0]
    current_sum = arr[0]
    operations = 0
    
    for i in range(1, len(arr)):
        operations += 1
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum, operations

def divide_conquer_max_subarray(arr):
    """Divide & Conquer - O(n log n) time, O(log n) space"""
    operations = [0]
    
    def max_crossing_sum(arr, left, mid, right):
        operations[0] += 1
        # Left side
        left_sum = float('-inf')
        current_sum = 0
        for i in range(mid, left - 1, -1):
            current_sum += arr[i]
            left_sum = max(left_sum, current_sum)
        
        # Right side
        right_sum = float('-inf')
        current_sum = 0
        for i in range(mid + 1, right + 1):
            current_sum += arr[i]
            right_sum = max(right_sum, current_sum)
        
        return left_sum + right_sum
    
    def max_subarray(arr, left, right):
        if left == right:
            return arr[left]
        
        mid = (left + right) // 2
        
        left_max = max_subarray(arr, left, mid)
        right_max = max_subarray(arr, mid + 1, right)
        cross_max = max_crossing_sum(arr, left, mid, right)
        
        return max(left_max, right_max, cross_max)
    
    result = max_subarray(arr, 0, len(arr) - 1)
    return result, operations[0]

# Test with various array sizes
test_arrays = [
    [-2, 1, -3, 4, -1, 2, 1, -5, 4],
    [5, -3, 5],
    list(range(-50, 51))  # Larger array
]

for i, arr in enumerate(test_arrays, 1):
    print(f"\nTest {i}: Array of size {len(arr)}")
    
    start = time.perf_counter()
    kadane_result, kadane_ops = kadane_algorithm(arr)
    kadane_time = time.perf_counter() - start
    
    start = time.perf_counter()
    dc_result, dc_ops = divide_conquer_max_subarray(arr)
    dc_time = time.perf_counter() - start
    
    print(f"Kadane's:        Max = {kadane_result}, Operations = {kadane_ops}, Time = {kadane_time*1e6:.2f}μs")
    print(f"Divide&Conquer:  Max = {dc_result}, Operations = {dc_ops}, Time = {dc_time*1e6:.2f}μs")
    print(f"Speed ratio: Kadane's is {dc_time/kadane_time:.1f}x faster")

print("\n✓ WINNER: Kadane's Algorithm")
print("  - Linear time vs O(n log n)")
print("  - Constant space vs O(log n)")
print("  - Simpler implementation")
print("  - Divide & Conquer is mainly educational")
print()


# ============================================================================
# COMPARISON 3: Top-Down (Memoization) vs Bottom-Up DP
# ============================================================================

print("=" * 80)
print("COMPARISON 3: TOP-DOWN (Memoization) vs BOTTOM-UP DP")
print("=" * 80)

"""
SCENARIO: Fibonacci sequence, Coin Change, etc.

KEY DIFFERENCES:
- Top-Down: Recursive with caching (memoization)
- Bottom-Up: Iterative, builds table from base cases

TIME COMPLEXITY: Both O(n) for Fibonacci, O(n*amount) for Coin Change
SPACE COMPLEXITY: 
  - Top-Down: O(n) cache + O(n) recursion stack
  - Bottom-Up: O(n) table only
"""

# Example: Fibonacci
def fibonacci_topdown(n, memo=None):
    """Top-Down with Memoization"""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_topdown(n-1, memo) + fibonacci_topdown(n-2, memo)
    return memo[n]

def fibonacci_bottomup(n):
    """Bottom-Up Iterative"""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Test
test_n = 35

start = time.perf_counter()
result_topdown = fibonacci_topdown(test_n)
time_topdown = time.perf_counter() - start

start = time.perf_counter()
result_bottomup = fibonacci_bottomup(test_n)
time_bottomup = time.perf_counter() - start

print(f"\nFibonacci({test_n}):")
print(f"Top-Down:   Result = {result_topdown}, Time = {time_topdown*1e6:.2f}μs")
print(f"Bottom-Up:  Result = {result_bottomup}, Time = {time_bottomup*1e6:.2f}μs")
print(f"Speed ratio: Bottom-Up is {time_topdown/time_bottomup:.1f}x faster")

# Example: Coin Change
def coin_change_topdown(coins, amount):
    """Top-Down Memoization"""
    memo = {}
    
    def dp(remaining):
        if remaining in memo:
            return memo[remaining]
        if remaining == 0:
            return 0
        if remaining < 0:
            return float('inf')
        
        min_coins = float('inf')
        for coin in coins:
            min_coins = min(min_coins, 1 + dp(remaining - coin))
        
        memo[remaining] = min_coins
        return min_coins
    
    result = dp(amount)
    return result if result != float('inf') else -1

def coin_change_bottomup(coins, amount):
    """Bottom-Up Iterative"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

coins = [1, 2, 5]
amount = 11

start = time.perf_counter()
result_topdown = coin_change_topdown(coins, amount)
time_topdown = time.perf_counter() - start

start = time.perf_counter()
result_bottomup = coin_change_bottomup(coins, amount)
time_bottomup = time.perf_counter() - start

print(f"\nCoin Change (coins={coins}, amount={amount}):")
print(f"Top-Down:   Result = {result_topdown}, Time = {time_topdown*1e6:.2f}μs")
print(f"Bottom-Up:  Result = {result_bottomup}, Time = {time_bottomup*1e6:.2f}μs")

print("\n✓ WHEN TO USE:")
print("Top-Down (Memoization):")
print("  - More intuitive (think recursively)")
print("  - Only computes needed subproblems")
print("  - Good for sparse state spaces")
print("  - Easier to code initially")
print("\nBottom-Up:")
print("  - Slightly faster (no recursion overhead)")
print("  - No recursion limit issues")
print("  - Computes all subproblems")
print("  - Better for dense state spaces")
print()


# ============================================================================
# COMPARISON 4: Greedy vs Dynamic Programming
# ============================================================================

print("=" * 80)
print("COMPARISON 4: GREEDY vs DYNAMIC PROGRAMMING")
print("=" * 80)

"""
SCENARIO: Activity Selection vs Weighted Activity Selection

KEY DIFFERENCES:
- Greedy: Makes locally optimal choice at each step
- DP: Considers all possibilities, stores subproblem solutions

TIME COMPLEXITY:
  - Greedy: O(n log n) for sorting
  - DP: O(n²) typically

When Greedy Works: Activity selection, Huffman coding, Dijkstra
When DP Needed: Knapsack, weighted activities, subset problems
"""

# Activity Selection - Greedy WORKS
def activity_selection_greedy(activities):
    """Greedy: Sort by finish time, select non-overlapping"""
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish = activities[0][1]
    
    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish
    
    return selected

# Weighted Activity Selection - Need DP
def weighted_activity_selection_dp(activities, weights):
    """DP: Consider all possibilities with weights"""
    n = len(activities)
    # Sort by finish time
    activities = sorted(zip(activities, weights), key=lambda x: x[0][1])
    
    # dp[i] = max weight using activities 0...i
    dp = [0] * n
    dp[0] = activities[0][1]
    
    for i in range(1, n):
        # Option 1: Don't include current activity
        exclude = dp[i-1]
        
        # Option 2: Include current activity
        include = activities[i][1]
        
        # Find last non-overlapping activity
        for j in range(i-1, -1, -1):
            if activities[j][0][1] <= activities[i][0][0]:
                include += dp[j]
                break
        
        dp[i] = max(include, exclude)
    
    return dp[n-1]

# Test - Unweighted (Greedy works)
activities = [(1,3), (2,5), (4,7), (6,9), (8,10)]
greedy_result = activity_selection_greedy(activities)

print("\nActivity Selection (Unweighted):")
print(f"Activities: {activities}")
print(f"Greedy selects: {greedy_result}")
print(f"Count: {len(greedy_result)} activities")

# Test - Weighted (Need DP)
activities = [(1,3), (2,5), (4,7), (6,9), (8,10)]
weights = [5, 6, 5, 4, 11]  # Profits

start = time.perf_counter()
dp_result = weighted_activity_selection_dp(activities, weights)
dp_time = time.perf_counter() - start

print(f"\nWeighted Activity Selection:")
print(f"Activities: {activities}")
print(f"Weights: {weights}")
print(f"DP max weight: {dp_result}")
print(f"Time: {dp_time*1e6:.2f}μs")

# Example where Greedy FAILS - Coin Change with non-canonical system
print("\n\nCoin Change Example:")
print("Canonical system (Greedy works): coins = [1, 5, 10, 25]")
print("Amount = 30: Greedy takes 25+5 = 2 coins ✓")

print("\nNon-canonical system (Greedy fails): coins = [1, 3, 4]")
print("Amount = 6:")
print("  Greedy: Takes 4+1+1 = 3 coins ✗")
print("  DP:     Takes 3+3 = 2 coins ✓")

print("\n✓ WHEN TO USE:")
print("Greedy:")
print("  - Greedy choice property proven")
print("  - Optimal substructure holds")
print("  - Fast O(n log n) solution needed")
print("  - Examples: Activity selection, Huffman, MST")
print("\nDP:")
print("  - Greedy doesn't work")
print("  - Need to consider all options")
print("  - Overlapping subproblems")
print("  - Examples: Knapsack, weighted activities, subset sum")
print()


# ============================================================================
# COMPARISON 5: IDS vs BFS vs DFS (Search Algorithms)
# ============================================================================

print("=" * 80)
print("COMPARISON 5: IDS vs BFS vs DFS")
print("=" * 80)

"""
SCENARIO: Tree/Graph search

KEY DIFFERENCES:
- BFS: Queue, level-by-level, finds shortest path
- DFS: Stack/recursion, depth-first, memory efficient
- IDS: Combines BFS completeness with DFS space efficiency

TIME COMPLEXITY:
  - BFS: O(b^d)
  - DFS: O(b^m) where m = max depth
  - IDS: O(b^d) (revisits nodes but dominated by last level)

SPACE COMPLEXITY:
  - BFS: O(b^d) - stores entire level
  - DFS: O(bm) - path from root
  - IDS: O(bd) - like DFS
"""

class TreeNode:
    def __init__(self, val, children=None):
        self.val = val
        self.children = children or []

def bfs_search(root, target):
    """BFS - shortest path, high memory"""
    queue = deque([(root, 0)])
    nodes_visited = 0
    
    while queue:
        node, depth = queue.popleft()
        nodes_visited += 1
        
        if node.val == target:
            return depth, nodes_visited
        
        for child in node.children:
            queue.append((child, depth + 1))
    
    return -1, nodes_visited

def dfs_search(root, target):
    """DFS - low memory, may find longer path"""
    nodes_visited = [0]
    
    def dfs(node, depth):
        nodes_visited[0] += 1
        if node.val == target:
            return depth
        
        for child in node.children:
            result = dfs(child, depth + 1)
            if result != -1:
                return result
        
        return -1
    
    return dfs(root, 0), nodes_visited[0]

def ids_search(root, target):
    """IDS - combines BFS optimality with DFS space efficiency"""
    nodes_visited = [0]
    
    def dfs_limited(node, depth, limit):
        nodes_visited[0] += 1
        if node.val == target:
            return depth
        if depth == limit:
            return -1
        
        for child in node.children:
            result = dfs_limited(child, depth + 1, limit)
            if result != -1:
                return result
        
        return -1
    
    for limit in range(100):  # Arbitrary max depth
        result = dfs_limited(root, 0, limit)
        if result != -1:
            return result, nodes_visited[0]
    
    return -1, nodes_visited[0]

# Build a sample tree
root = TreeNode(1)
root.children = [TreeNode(2), TreeNode(3), TreeNode(4)]
root.children[0].children = [TreeNode(5), TreeNode(6)]
root.children[1].children = [TreeNode(7), TreeNode(8)]
root.children[2].children = [TreeNode(9), TreeNode(10)]

target = 8

print("\nTree structure (branching factor ~3):")
print("        1")
print("      / | \\")
print("     2  3  4")
print("    /| /| /|")
print("   5 6 7 8 9 10")
print()

bfs_depth, bfs_visited = bfs_search(root, target)
dfs_depth, dfs_visited = dfs_search(root, target)
ids_depth, ids_visited = ids_search(root, target)

print(f"Searching for node {target}:")
print(f"BFS: Depth = {bfs_depth}, Nodes visited = {bfs_visited}")
print(f"DFS: Depth = {dfs_depth}, Nodes visited = {dfs_visited}")
print(f"IDS: Depth = {ids_depth}, Nodes visited = {ids_visited}")

print("\n✓ COMPARISON:")
print("BFS:")
print("  + Finds shortest path")
print("  + Complete and optimal")
print("  - High memory O(b^d)")
print("\nDFS:")
print("  + Low memory O(bd)")
print("  + Good for deep solutions")
print("  - May find longer path")
print("  - Not complete (infinite paths)")
print("\nIDS:")
print("  + Finds shortest path (like BFS)")
print("  + Low memory (like DFS)")
print("  + Complete and optimal")
print("  - Revisits nodes (but acceptable overhead)")
print("  ★ Best of both worlds for large spaces!")
print()


# ============================================================================
# COMPARISON 6: Kruskal's vs Prim's (Minimum Spanning Tree)
# ============================================================================

print("=" * 80)
print("COMPARISON 6: KRUSKAL'S vs PRIM'S (MST)")
print("=" * 80)

"""
SCENARIO: Finding Minimum Spanning Tree

KEY DIFFERENCES:
- Kruskal's: Edge-based, sort all edges, use Union-Find
- Prim's: Vertex-based, grow tree from starting vertex, use priority queue

TIME COMPLEXITY:
  - Kruskal's: O(E log E) for sorting edges
  - Prim's: O(E log V) with binary heap

SPACE COMPLEXITY:
  - Both: O(V + E)

When to prefer:
- Kruskal's: Sparse graphs (fewer edges)
- Prim's: Dense graphs (many edges)
"""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        return True

def kruskal_mst(n, edges):
    """Kruskal's: Sort edges, use Union-Find"""
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0
    operations = 0
    
    for u, v, weight in edges:
        operations += 1
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == n - 1:
                break
    
    return total_weight, mst_edges, operations

def prim_mst(n, edges):
    """Prim's: Grow tree from starting vertex"""
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    
    visited = [False] * n
    pq = [(0, 0)]  # (weight, vertex)
    mst_edges = []
    total_weight = 0
    operations = 0
    
    while pq and len(mst_edges) < n - 1:
        weight, u = heapq.heappop(pq)
        operations += 1
        
        if visited[u]:
            continue
        
        visited[u] = True
        total_weight += weight
        if weight > 0:  # Skip first vertex (weight 0)
            mst_edges.append((u, weight))
        
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (w, v))
    
    return total_weight, mst_edges, operations

# Test graph
n = 6
edges = [
    (0, 1, 4), (0, 2, 4),
    (1, 2, 2), (1, 3, 5),
    (2, 3, 8), (2, 4, 10),
    (3, 4, 2), (3, 5, 6),
    (4, 5, 3)
]

print(f"\nGraph: {n} vertices, {len(edges)} edges")
print("Edges (u, v, weight):")
for edge in edges:
    print(f"  {edge}")

start = time.perf_counter()
kruskal_weight, kruskal_edges, kruskal_ops = kruskal_mst(n, edges[:])
kruskal_time = time.perf_counter() - start

start = time.perf_counter()
prim_weight, prim_edges, prim_ops = prim_mst(n, edges[:])
prim_time = time.perf_counter() - start

print(f"\nKruskal's:")
print(f"  MST weight: {kruskal_weight}")
print(f"  Operations: {kruskal_ops}")
print(f"  Time: {kruskal_time*1e6:.2f}μs")

print(f"\nPrim's:")
print(f"  MST weight: {prim_weight}")
print(f"  Operations: {prim_ops}")
print(f"  Time: {prim_time*1e6:.2f}μs")

print("\n✓ WHEN TO USE:")
print("Kruskal's:")
print("  + Better for sparse graphs (E << V²)")
print("  + Edges already sorted")
print("  + Don't need to start from specific vertex")
print("  + Uses Union-Find (elegant)")
print("\nPrim's:")
print("  + Better for dense graphs (E ≈ V²)")
print("  + Need MST from specific starting point")
print("  + Similar to Dijkstra's algorithm")
print("  + Can stop early if only partial MST needed")
print()


# ============================================================================
# COMPARISON 7: Backtracking vs Dynamic Programming
# ============================================================================

print("=" * 80)
print("COMPARISON 7: BACKTRACKING vs DYNAMIC PROGRAMMING")
print("=" * 80)

"""
SCENARIO: N-Queens vs Subset Sum

KEY DIFFERENCES:
- Backtracking: Explores all possibilities, backtracks on failure
- DP: Stores subproblem solutions, avoids recomputation

WHEN TO USE:
- Backtracking: Constraint satisfaction, generate all solutions
- DP: Optimization problems with overlapping subproblems
"""

# N-Queens (Backtracking is natural)
def n_queens_backtracking(n):
    """Count solutions using backtracking"""
    solutions = [0]
    
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or \
               board[i] - i == col - row or \
               board[i] + i == col + row:
                return False
        return True
    
    def solve(board, row):
        if row == n:
            solutions[0] += 1
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(board, row + 1)
    
    solve([-1] * n, 0)
    return solutions[0]

# Subset Sum (DP is better)
def subset_sum_dp(nums, target):
    """Check if subset exists with given sum using DP"""
    if target == 0:
        return True
    if not nums or target < 0:
        return False
    
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    return dp[target]

def subset_sum_backtracking(nums, target):
    """Check if subset exists using backtracking (slower)"""
    def backtrack(index, current_sum):
        if current_sum == target:
            return True
        if index == len(nums) or current_sum > target:
            return False
        
        # Include current number
        if backtrack(index + 1, current_sum + nums[index]):
            return True
        
        # Exclude current number
        return backtrack(index + 1, current_sum)
    
    return backtrack(0, 0)

# Test N-Queens
n = 8
start = time.perf_counter()
queens_solutions = n_queens_backtracking(n)
queens_time = time.perf_counter() - start

print(f"\nN-Queens (n={n}):")
print(f"Solutions: {queens_solutions}")
print(f"Time: {queens_time*1000:.2f}ms")
print("✓ Backtracking is natural choice (constraint satisfaction)")

# Test Subset Sum
nums = [3, 34, 4, 12, 5, 2]
target = 9

start = time.perf_counter()
result_dp = subset_sum_dp(nums, target)
dp_time = time.perf_counter() - start

start = time.perf_counter()
result_bt = subset_sum_backtracking(nums, target)
bt_time = time.perf_counter() - start

print(f"\nSubset Sum (nums={nums}, target={target}):")
print(f"DP:           Result = {result_dp}, Time = {dp_time*1e6:.2f}μs")
print(f"Backtracking: Result = {result_bt}, Time = {bt_time*1e6:.2f}μs")
print(f"Speed ratio: DP is {bt_time/dp_time:.1f}x faster")

print("\n✓ ALGORITHM SELECTION:")
print("Use Backtracking when:")
print("  - Constraint satisfaction problems")
print("  - Need ALL solutions (not just one)")
print("  - Pruning reduces search space significantly")
print("  - Examples: N-Queens, Sudoku, permutations")
print("\nUse DP when:")
print("  - Optimization problems (min/max)")
print("  - Overlapping subproblems exist")
print("  - Need one optimal solution")
print("  - Examples: Knapsack, coin change, LIS")
print()


# ============================================================================
# COMPARISON 8: Branch and Bound vs Backtracking
# ============================================================================

print("=" * 80)
print("COMPARISON 8: BRANCH AND BOUND vs BACKTRACKING")
print("=" * 80)

"""
SCENARIO: 0/1 Knapsack

KEY DIFFERENCES:
- Backtracking: Explores all possibilities with constraint checking
- Branch and Bound: Uses bounding function to prune more aggressively

TIME COMPLEXITY: Both O(2^n) worst case, B&B much better in practice
SPACE COMPLEXITY: Both O(n)
"""

def knapsack_backtracking(weights, values, capacity):
    """Backtracking - explores with basic pruning"""
    max_value = [0]
    pruned = [0]
    explored = [0]
    
    def backtrack(index, current_weight, current_value):
        explored[0] += 1
        
        if index == len(weights):
            max_value[0] = max(max_value[0], current_value)
            return
        
        # Prune if over capacity
        if current_weight > capacity:
            pruned[0] += 1
            return
        
        # Include current item
        if current_weight + weights[index] <= capacity:
            backtrack(index + 1, 
                     current_weight + weights[index],
                     current_value + values[index])
        
        # Exclude current item
        backtrack(index + 1, current_weight, current_value)
    
    backtrack(0, 0, 0)
    return max_value[0], explored[0], pruned[0]

def knapsack_branch_bound(weights, values, capacity):
    """Branch and Bound - uses fractional bound for aggressive pruning"""
    max_value = [0]
    pruned = [0]
    explored = [0]
    
    # Sort by value/weight ratio
    items = sorted(enumerate(zip(weights, values)), 
                   key=lambda x: x[1][1]/x[1][0] if x[1][0] > 0 else 0, 
                   reverse=True)
    
    def fractional_bound(index, current_weight, current_value):
        """Calculate upper bound using fractional knapsack"""
        if current_weight >= capacity:
            return 0
        
        bound = current_value
        total_weight = current_weight
        
        for i in range(index, len(items)):
            weight, value = items[i][1]
            if total_weight + weight <= capacity:
                total_weight += weight
                bound += value
            else:
                # Add fraction of remaining item
                remaining = capacity - total_weight
                bound += value * (remaining / weight)
                break
        
        return bound
    
    def branch_bound(index, current_weight, current_value):
        explored[0] += 1
        
        if current_weight <= capacity:
            max_value[0] = max(max_value[0], current_value)
        
        if index == len(items):
            return
        
        # Calculate bound
        bound = fractional_bound(index, current_weight, current_value)
        
        # Prune if bound can't improve
        if bound <= max_value[0]:
            pruned[0] += 1
            return
        
        weight, value = items[index][1]
        
        # Include current item
        if current_weight + weight <= capacity:
            branch_bound(index + 1, 
                        current_weight + weight,
                        current_value + value)
        
        # Exclude current item
        branch_bound(index + 1, current_weight, current_value)
    
    branch_bound(0, 0, 0)
    return max_value[0], explored[0], pruned[0]

# Test with various problem sizes
test_cases = [
    {
        'weights': [2, 3, 4, 5],
        'values': [3, 4, 5, 6],
        'capacity': 8
    },
    {
        'weights': [10, 20, 30, 40, 50],
        'values': [20, 30, 40, 50, 60],
        'capacity': 60
    }
]

for i, test in enumerate(test_cases, 1):
    weights = test['weights']
    values = test['values']
    capacity = test['capacity']
    
    print(f"\nTest {i}: Items = {len(weights)}, Capacity = {capacity}")
    
    start = time.perf_counter()
    bt_value, bt_explored, bt_pruned = knapsack_backtracking(weights, values, capacity)
    bt_time = time.perf_counter() - start
    
    start = time.perf_counter()
    bb_value, bb_explored, bb_pruned = knapsack_branch_bound(weights, values, capacity)
    bb_time = time.perf_counter() - start
    
    print(f"Backtracking:")
    print(f"  Value: {bt_value}, Explored: {bt_explored}, Pruned: {bt_pruned}, Time: {bt_time*1e6:.2f}μs")
    print(f"Branch & Bound:")
    print(f"  Value: {bb_value}, Explored: {bb_explored}, Pruned: {bb_pruned}, Time: {bb_time*1e6:.2f}μs")
    print(f"Pruning efficiency: B&B pruned {bb_pruned}/{bb_explored+bb_pruned} = {bb_pruned/(bb_explored+bb_pruned)*100:.1f}%")

print("\n✓ WINNER: Branch and Bound")
print("  - Uses bounding function for aggressive pruning")
print("  - Explores fewer nodes in practice")
print("  - Better for optimization problems")
print("  - Backtracking better for constraint satisfaction")
print()


# ============================================================================
# COMPARISON 9: Iterative vs Recursive
# ============================================================================

print("=" * 80)
print("COMPARISON 9: ITERATIVE vs RECURSIVE")
print("=" * 80)

"""
SCENARIO: Tree traversal, factorial, etc.

KEY DIFFERENCES:
- Iterative: Uses loops, explicit stack if needed
- Recursive: Function calls itself, implicit call stack

SPACE COMPLEXITY:
  - Iterative: O(1) or O(n) for explicit stack
  - Recursive: O(n) for call stack
"""

# Tree traversal
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_recursive(root):
    """Recursive inorder traversal"""
    result = []
    
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)
    
    traverse(root)
    return result

def inorder_iterative(root):
    """Iterative inorder traversal"""
    result = []
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result

# Build test tree
root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(6)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
root.right.left = TreeNode(5)
root.right.right = TreeNode(7)

print("\nTree structure:")
print("      4")
print("    /   \\")
print("   2     6")
print("  / \\   / \\")
print(" 1   3 5   7")
print()

start = time.perf_counter()
result_recursive = inorder_recursive(root)
time_recursive = time.perf_counter() - start

start = time.perf_counter()
result_iterative = inorder_iterative(root)
time_iterative = time.perf_counter() - start

print(f"Inorder Traversal:")
print(f"Recursive:  {result_recursive}, Time: {time_recursive*1e6:.2f}μs")
print(f"Iterative:  {result_iterative}, Time: {time_iterative*1e6:.2f}μs")

# Fibonacci with large n
def fibonacci_recursive_pure(n):
    """Pure recursive (exponential without memoization)"""
    if n <= 1:
        return n
    return fibonacci_recursive_pure(n-1) + fibonacci_recursive_pure(n-2)

def fibonacci_iterative_optimal(n):
    """Iterative with O(1) space"""
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    
    return prev1

n = 30
print(f"\nFibonacci({n}):")

start = time.perf_counter()
result = fibonacci_recursive_pure(n)
time_rec = time.perf_counter() - start

start = time.perf_counter()
result = fibonacci_iterative_optimal(n)
time_iter = time.perf_counter() - start

print(f"Recursive: Time: {time_rec*1000:.2f}ms")
print(f"Iterative: Time: {time_iter*1e6:.2f}μs")
print(f"Speed ratio: Iterative is {time_rec/time_iter:.0f}x faster")

print("\n✓ WHEN TO USE:")
print("Recursive:")
print("  + More intuitive and elegant")
print("  + Natural for tree/graph problems")
print("  + Easier to code initially")
print("  - Stack overflow risk")
print("  - Overhead from function calls")
print("\nIterative:")
print("  + No stack overflow")
print("  + Slightly faster (no call overhead)")
print("  + Better space efficiency")
print("  - Sometimes less intuitive")
print("  - May need explicit stack")
print()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("ALGORITHM COMPARISON SUMMARY")
print("=" * 80)

summary = """
KEY TAKEAWAYS:
==============

1. BFS vs DFS:
   • BFS: Shortest path in unweighted graphs, high memory
   • DFS: Memory efficient, explores deeply
   • IDS: Best of both worlds!

2. Kadane's vs D&C:
   • Kadane's: Clear winner (O(n) vs O(n log n))
   • D&C: Educational value only

3. Top-Down vs Bottom-Up DP:
   • Top-Down: More intuitive, sparse states
   • Bottom-Up: Slightly faster, no recursion limit

4. Greedy vs DP:
   • Greedy: Fast when it works (must prove!)
   • DP: General solution for optimization

5. IDS vs BFS vs DFS:
   • IDS: Combines BFS optimality with DFS space efficiency
   • Use when memory is constrained

6. Kruskal's vs Prim's:
   • Kruskal's: Better for sparse graphs
   • Prim's: Better for dense graphs

7. Backtracking vs DP:
   • Backtracking: Constraint satisfaction, all solutions
   • DP: Optimization, one optimal solution

8. Branch & Bound vs Backtracking:
   • B&B: More aggressive pruning with bounds
   • Better for optimization problems

9. Iterative vs Recursive:
   • Iterative: Faster, no stack overflow
   • Recursive: More elegant, natural for trees

SELECTION GUIDE:
===============
• Shortest path (unweighted)? → BFS
• Memory constrained? → DFS or IDS
• Max/min optimization? → DP
• Fast approximation? → Greedy (if applicable)
• All solutions needed? → Backtracking
• Exact solution to NP-hard? → Branch & Bound
• Tree/graph connectivity? → DFS/BFS
• Spanning tree? → Kruskal's or Prim's
"""

print(summary)
print("=" * 80)
print("Use this guide to choose the right algorithm for your problem!")
print("=" * 80)
