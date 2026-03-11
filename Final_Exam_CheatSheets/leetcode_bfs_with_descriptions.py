"""
10 LeetCode BFS Problems with Description, Input, and Output Examples
Each section contains:
- Problem description (short summary)
- Example input
- Example output
- Python BFS solution
"""

from typing import List, Optional
from collections import deque


# --------------------------------------------------
# 1. Binary Tree Level Order Traversal (LeetCode 102)
# --------------------------------------------------
"""
Description:
Given a binary tree, return the level order traversal of its nodes' values.

Example Input:
Tree:
    3
   / \
  9  20
    /  \
   15   7

Example Output:
[[3], [9, 20], [15, 7]]
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution102:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result


# --------------------------------------------------
# 2. Rotting Oranges (LeetCode 994)
# --------------------------------------------------
"""
Description:
Every minute, any fresh orange adjacent to a rotten orange becomes rotten.
Return the minimum minutes until no fresh oranges remain.

Example Input:
grid = [
[2,1,1],
[1,1,0],
[0,1,1]
]

Example Output:
4
"""

class Solution994:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        
        # Count fresh oranges and add rotten to queue
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c, 0))  # (row, col, time)
                elif grid[r][c] == 1:
                    fresh += 1
        
        if fresh == 0:
            return 0
        
        minutes = 0
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        
        while queue:
            r, c, time = queue.popleft()
            minutes = max(minutes, time)
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc, time + 1))
        
        return minutes if fresh == 0 else -1


# --------------------------------------------------
# 3. Shortest Path in Binary Matrix (LeetCode 1091)
# --------------------------------------------------
"""
Description:
Find shortest path from top-left to bottom-right in binary matrix.
Can move in 8 directions. 0 = empty, 1 = blocked.

Example Input:
grid = [
[0,0,0],
[1,1,0],
[1,1,0]
]

Example Output:
4
"""

class Solution1091:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] == 1 or grid[-1][-1] == 1:
            return -1
        
        n = len(grid)
        if n == 1:
            return 1
        
        queue = deque([(0, 0, 1)])  # (row, col, distance)
        visited = {(0, 0)}
        directions = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        
        while queue:
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if nr == n-1 and nc == n-1:
                    return dist + 1
                
                if (0 <= nr < n and 0 <= nc < n and 
                    grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        
        return -1


# --------------------------------------------------
# 4. Word Ladder (LeetCode 127)
# --------------------------------------------------
"""
Description:
Transform beginWord to endWord changing one letter at a time,
using only words from wordList.

Example Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]

Example Output:
5
Explanation: "hit" -> "hot" -> "dot" -> "dog" -> "cog"
"""

class Solution127:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, steps = queue.popleft()
            
            if word == endWord:
                return steps
            
            # Try changing each character
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    
                    if next_word in wordSet and next_word not in visited:
                        visited.add(next_word)
                        queue.append((next_word, steps + 1))
        
        return 0


# --------------------------------------------------
# 5. Walls and Gates (LeetCode 286)
# --------------------------------------------------
"""
Description:
Fill each empty room with distance to nearest gate.
-1 = wall, 0 = gate, INF = empty room

Example Input:
rooms = [
[INF, -1,  0,  INF],
[INF, INF, INF, -1],
[INF, -1,  INF, -1],
[0,   -1,  INF, INF]
]

Example Output:
[
[3, -1, 0, 1],
[2,  2, 1,-1],
[1, -1, 2,-1],
[0, -1, 3, 4]
]
"""

class Solution286:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        if not rooms:
            return
        
        rows, cols = len(rooms), len(rooms[0])
        queue = deque()
        INF = 2147483647
        
        # Add all gates to queue
        for r in range(rows):
            for c in range(cols):
                if rooms[r][c] == 0:
                    queue.append((r, c))
        
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    rooms[nr][nc] == INF):
                    rooms[nr][nc] = rooms[r][c] + 1
                    queue.append((nr, nc))


# --------------------------------------------------
# 6. Perfect Squares (LeetCode 279)
# --------------------------------------------------
"""
Description:
Find the least number of perfect square numbers that sum to n.

Example Input:
n = 12

Example Output:
3
Explanation: 12 = 4 + 4 + 4
"""

class Solution279:
    def numSquares(self, n: int) -> int:
        # Generate perfect squares up to n
        squares = []
        i = 1
        while i * i <= n:
            squares.append(i * i)
            i += 1
        
        queue = deque([(n, 0)])  # (remaining, steps)
        visited = {n}
        
        while queue:
            remaining, steps = queue.popleft()
            
            if remaining == 0:
                return steps
            
            for square in squares:
                next_remaining = remaining - square
                
                if next_remaining >= 0 and next_remaining not in visited:
                    visited.add(next_remaining)
                    queue.append((next_remaining, steps + 1))
        
        return -1


# --------------------------------------------------
# 7. Open the Lock (LeetCode 752)
# --------------------------------------------------
"""
Description:
Unlock a lock by rotating wheels. Start at "0000", target is given.
deadends cannot be visited.

Example Input:
deadends = ["0201","0101","0102","1212","2002"]
target = "0202"

Example Output:
6
"""

class Solution752:
    def openLock(self, deadends: List[str], target: str) -> int:
        dead = set(deadends)
        
        if "0000" in dead:
            return -1
        if target == "0000":
            return 0
        
        queue = deque([("0000", 0)])
        visited = {"0000"}
        
        def neighbors(code):
            result = []
            for i in range(4):
                digit = int(code[i])
                for move in [-1, 1]:
                    new_digit = (digit + move) % 10
                    result.append(code[:i] + str(new_digit) + code[i+1:])
            return result
        
        while queue:
            code, steps = queue.popleft()
            
            for neighbor in neighbors(code):
                if neighbor == target:
                    return steps + 1
                
                if neighbor not in visited and neighbor not in dead:
                    visited.add(neighbor)
                    queue.append((neighbor, steps + 1))
        
        return -1


# --------------------------------------------------
# 8. Minimum Depth of Binary Tree (LeetCode 111)
# --------------------------------------------------
"""
Description:
Find the minimum depth from root to any leaf node.

Example Input:
Tree:
    3
   / \
  9  20
    /  \
   15   7

Example Output:
2
"""

class Solution111:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        queue = deque([(root, 1)])
        
        while queue:
            node, depth = queue.popleft()
            
            # If leaf node, return depth
            if not node.left and not node.right:
                return depth
            
            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))
        
        return 0


# --------------------------------------------------
# 9. Shortest Bridge (LeetCode 934)
# --------------------------------------------------
"""
Description:
Given a binary matrix with exactly two islands, find the shortest
bridge (flips) to connect them.

Example Input:
grid = [
[0,1],
[1,0]
]

Example Output:
1
"""

class Solution934:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        n = len(grid)
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        
        # Find first island using DFS
        def dfs(r, c):
            if r < 0 or c < 0 or r >= n or c >= n or grid[r][c] != 1:
                return
            grid[r][c] = 2
            first_island.append((r, c))
            for dr, dc in directions:
                dfs(r + dr, c + dc)
        
        # Find first island
        first_island = []
        found = False
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    dfs(r, c)
                    found = True
                    break
            if found:
                break
        
        # BFS from first island to find second island
        queue = deque([(r, c, 0) for r, c in first_island])
        
        while queue:
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n:
                    if grid[nr][nc] == 1:
                        return dist
                    elif grid[nr][nc] == 0:
                        grid[nr][nc] = 2
                        queue.append((nr, nc, dist + 1))
        
        return -1


# --------------------------------------------------
# 10. As Far from Land as Possible (LeetCode 1162)
# --------------------------------------------------
"""
Description:
Find the water cell that is farthest from any land cell.
Return the distance.

Example Input:
grid = [
[1,0,1],
[0,0,0],
[1,0,1]
]

Example Output:
2
"""

class Solution1162:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n = len(grid)
        queue = deque()
        
        # Add all land cells to queue
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    queue.append((r, c, 0))
        
        # If all land or all water
        if len(queue) == 0 or len(queue) == n * n:
            return -1
        
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        max_dist = 0
        
        while queue:
            r, c, dist = queue.popleft()
            max_dist = max(max_dist, dist)
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1  # Mark as visited
                    queue.append((nr, nc, dist + 1))
        
        return max_dist


if __name__ == "__main__":
    print("BFS practice file with 10 LeetCode problems.")


# ============================================================================
# PROBLEM INDEX - QUICK REFERENCE
# ============================================================================
"""
================================================================================
                    LEETCODE BFS PROBLEMS - CONTENT INDEX
================================================================================

BREADTH-FIRST SEARCH (BFS) PROBLEMS
-------------------------------------------------
Line 17   | Problem 1:  Binary Tree Level Order Traversal (LeetCode 102)
Line 69   | Problem 2:  Rotting Oranges (LeetCode 994)
Line 129  | Problem 3:  Shortest Path in Binary Matrix (LeetCode 1091)
Line 187  | Problem 4:  Word Ladder (LeetCode 127)
Line 240  | Problem 5:  Walls and Gates (LeetCode 286)
Line 298  | Problem 6:  Perfect Squares (LeetCode 279)
Line 351  | Problem 7:  Open the Lock (LeetCode 752)
Line 408  | Problem 8:  Minimum Depth of Binary Tree (LeetCode 111)
Line 455  | Problem 9:  Shortest Bridge (LeetCode 934)
Line 515  | Problem 10: As Far from Land as Possible (LeetCode 1162)

================================================================================
                          TOTAL: 10 BFS PROBLEMS
================================================================================

PROBLEM CATEGORIES:
-------------------
TREE TRAVERSAL (Level-by-Level):
  • Problem 1: Binary Tree Level Order - Classic BFS tree traversal
  • Problem 8: Minimum Depth of Binary Tree - Shortest path to leaf

GRID/MATRIX TRAVERSAL (Multi-Source BFS):
  • Problem 2: Rotting Oranges - Multi-source, time-based spread
  • Problem 5: Walls and Gates - Multi-source distance calculation
  • Problem 10: As Far from Land - Multi-source maximum distance

SHORTEST PATH PROBLEMS:
  • Problem 3: Shortest Path in Binary Matrix - 8-directional movement
  • Problem 9: Shortest Bridge - Connect two components
  
TRANSFORMATION/STATE SPACE:
  • Problem 4: Word Ladder - Word transformation with constraints
  • Problem 6: Perfect Squares - Number theory with BFS
  • Problem 7: Open the Lock - Combination lock with deadends

KEY BFS PATTERNS:
-----------------
1. SINGLE-SOURCE BFS:
   - Start from one node/cell
   - Find shortest path to target
   - Use queue with (position, distance)

2. MULTI-SOURCE BFS:
   - Start from multiple nodes simultaneously
   - Process all sources in one BFS
   - Common in "spread" or "distance to nearest" problems

3. LEVEL-ORDER BFS:
   - Process nodes level by level
   - Track level size: for _ in range(len(queue))
   - Used in tree traversal

4. STATE-SPACE BFS:
   - States are not positions but configurations
   - Generate next states from current
   - Examples: word transformations, lock combinations

BFS VS DFS - WHEN TO USE BFS:
------------------------------
✓ Need SHORTEST PATH in unweighted graph
✓ Need to explore level-by-level
✓ Solution likely close to start
✓ Need all nodes at same distance
✓ Multi-source exploration

✗ Don't use if solution is very deep (use DFS)
✗ Don't use if need all paths (use DFS/Backtracking)
✗ Don't use if weighted graph (use Dijkstra)

TIME COMPLEXITY: O(V + E) for graphs, O(m×n) for grids
SPACE COMPLEXITY: O(V) for queue and visited set

COMMON BFS TEMPLATE:
--------------------
```python
from collections import deque

queue = deque([start])
visited = {start}

while queue:
    current = queue.popleft()
    
    if current == target:
        return result
    
    for neighbor in get_neighbors(current):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

MULTI-SOURCE BFS TEMPLATE:
---------------------------
```python
queue = deque()
for source in sources:
    queue.append((source, 0))

while queue:
    node, dist = queue.popleft()
    
    for neighbor in get_neighbors(node):
        if not visited[neighbor]:
            visited[neighbor] = True
            queue.append((neighbor, dist + 1))
```

KEY DIFFERENCES FROM DFS:
-------------------------
| Feature          | BFS              | DFS              |
|------------------|------------------|------------------|
| Data Structure   | Queue (deque)    | Stack/Recursion  |
| Order            | Level-by-level   | Depth-first      |
| Shortest Path    | YES (unweighted) | NO               |
| Space            | O(width)         | O(depth)         |
| Use Case         | Nearest/shortest | All paths/cycles |

DEBUGGING TIPS:
--------------
- Print queue contents at each step
- Visualize grid with distance values
- Check visited set to avoid cycles
- Verify boundary conditions (0 <= x < n)
- For multi-source: ensure all sources added initially

HOW TO USE THIS INDEX:
----------------------
1. Use Cmd/Ctrl+G to go to specific line number
2. Each problem includes description, examples, and solution
3. Practice identifying single-source vs multi-source BFS
4. Master the queue pattern: append right, popleft

COMMON MISTAKES TO AVOID:
-------------------------
✗ Forgetting to mark as visited BEFORE adding to queue
✗ Using stack instead of queue (makes it DFS!)
✗ Not handling edge cases (empty grid, no path)
✗ Confusing distance vs number of nodes
✗ Not using deque (list.pop(0) is O(n)!)

================================================================================
"""
