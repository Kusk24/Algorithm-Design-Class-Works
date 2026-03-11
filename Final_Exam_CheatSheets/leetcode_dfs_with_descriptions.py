"""
10 LeetCode DFS Problems with Description, Input, and Output Examples
Each section contains:
- Problem description (short summary)
- Example input
- Example output
- Python DFS solution
"""

from typing import List, Optional


# --------------------------------------------------
# 1. Number of Islands (LeetCode 200)
# --------------------------------------------------
"""
Description:
Given a 2D grid of '1's (land) and '0's (water), count the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands
(horizontally or vertically).

Example Input:
grid = [
["1","1","0","0","0"],
["1","1","0","0","0"],
["0","0","1","0","0"],
["0","0","0","1","1"]
]

Example Output:
3
"""

class Solution200:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])

        def dfs(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] != "1":
                return
            grid[r][c] = "0"
            dfs(r+1, c)
            dfs(r-1, c)
            dfs(r, c+1)
            dfs(r, c-1)

        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1":
                    count += 1
                    dfs(r, c)
        return count


# --------------------------------------------------
# 2. Max Area of Island (LeetCode 695)
# --------------------------------------------------
"""
Description:
Return the maximum area of an island in a grid.

Example Input:
grid = [
[0,0,1,0],
[1,1,1,0],
[0,1,0,0]
]

Example Output:
5
"""

class Solution695:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])

        def dfs(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == 0:
                return 0
            grid[r][c] = 0
            return 1 + dfs(r+1,c) + dfs(r-1,c) + dfs(r,c+1) + dfs(r,c-1)

        ans = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    ans = max(ans, dfs(r,c))
        return ans


# --------------------------------------------------
# 3. Flood Fill (LeetCode 733)
# --------------------------------------------------
"""
Description:
Given an image represented by a grid, change the color of the starting pixel
and all connected pixels with the same color.

Example Input:
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1
sc = 1
color = 2

Example Output:
[[2,2,2],[2,2,0],[2,0,1]]
"""

class Solution733:
    def floodFill(self, image, sr, sc, color):
        start = image[sr][sc]
        rows, cols = len(image), len(image[0])

        def dfs(r,c):
            if r<0 or c<0 or r>=rows or c>=cols or image[r][c]!=start:
                return
            image[r][c] = color
            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        if start != color:
            dfs(sr,sc)

        return image


# --------------------------------------------------
# 4. Clone Graph (LeetCode 133)
# --------------------------------------------------
"""
Description:
Clone an undirected graph.

Example Input:
1 -- 2
|    |
4 -- 3

Example Output:
Cloned graph with same connections
"""

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []


class Solution133:
    def cloneGraph(self, node: 'Node') -> 'Node':
        visited = {}

        def dfs(n):
            if n in visited:
                return visited[n]

            copy = Node(n.val)
            visited[n] = copy

            for nei in n.neighbors:
                copy.neighbors.append(dfs(nei))

            return copy

        if not node:
            return None

        return dfs(node)


# --------------------------------------------------
# 5. Course Schedule (LeetCode 207)
# --------------------------------------------------
"""
Description:
Determine if it is possible to finish all courses given prerequisites.

Example Input:
numCourses = 2
prerequisites = [[1,0]]

Example Output:
True
"""

class Solution207:
    def canFinish(self, numCourses, prerequisites):

        graph = {i:[] for i in range(numCourses)}

        for a,b in prerequisites:
            graph[a].append(b)

        visiting = set()
        visited = set()

        def dfs(course):
            if course in visiting:
                return False
            if course in visited:
                return True

            visiting.add(course)

            for pre in graph[course]:
                if not dfs(pre):
                    return False

            visiting.remove(course)
            visited.add(course)

            return True

        for c in range(numCourses):
            if not dfs(c):
                return False

        return True


# --------------------------------------------------
# 6. Pacific Atlantic Water Flow (LeetCode 417)
# --------------------------------------------------
"""
Description:
Find all cells where water can flow to both Pacific and Atlantic oceans.

Example Input:
heights = [
[1,2,2,3],
[3,2,3,4],
[2,4,5,3]
]

Example Output:
Cells reachable by both oceans
"""

class Solution417:

    def pacificAtlantic(self, heights):

        rows, cols = len(heights), len(heights[0])

        pac, atl = set(), set()

        def dfs(r,c,visited,prev):

            if (r<0 or c<0 or r>=rows or c>=cols or
                (r,c) in visited or heights[r][c] < prev):
                return

            visited.add((r,c))

            dfs(r+1,c,visited,heights[r][c])
            dfs(r-1,c,visited,heights[r][c])
            dfs(r,c+1,visited,heights[r][c])
            dfs(r,c-1,visited,heights[r][c])

        for c in range(cols):
            dfs(0,c,pac,heights[0][c])
            dfs(rows-1,c,atl,heights[rows-1][c])

        for r in range(rows):
            dfs(r,0,pac,heights[r][0])
            dfs(r,cols-1,atl,heights[r][cols-1])

        return list(pac & atl)


# --------------------------------------------------
# 7. Surrounded Regions (LeetCode 130)
# --------------------------------------------------
"""
Description:
Capture surrounded regions on the board.

Example Input:
X X X X
X O O X
X X O X
X O X X

Example Output:
X X X X
X X X X
X X X X
X O X X
"""

class Solution130:

    def solve(self, board):

        rows, cols = len(board), len(board[0])

        def dfs(r,c):
            if r<0 or c<0 or r>=rows or c>=cols or board[r][c]!="O":
                return
            board[r][c] = "T"

            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        for r in range(rows):
            dfs(r,0)
            dfs(r,cols-1)

        for c in range(cols):
            dfs(0,c)
            dfs(rows-1,c)

        for r in range(rows):
            for c in range(cols):
                if board[r][c]=="O":
                    board[r][c]="X"
                elif board[r][c]=="T":
                    board[r][c]="O"


# --------------------------------------------------
# 8. Path Sum (LeetCode 112)
# --------------------------------------------------
"""
Description:
Determine if the tree has a root‑to‑leaf path equal to target sum.

Example Input:
Tree:
    5
   / \
  4   8
targetSum = 9

Example Output:
True
"""

class TreeNode:

    def __init__(self,val=0,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution112:

    def hasPathSum(self, root, targetSum):

        if not root:
            return False

        if not root.left and not root.right:
            return root.val == targetSum

        return (self.hasPathSum(root.left,targetSum-root.val) or
                self.hasPathSum(root.right,targetSum-root.val))


# --------------------------------------------------
# 9. Binary Tree Paths (LeetCode 257)
# --------------------------------------------------
"""
Description:
Return all root‑to‑leaf paths in a binary tree.

Example Input:
Tree:
   1
  / \
 2   3
  \
   5

Example Output:
["1->2->5", "1->3"]
"""

class Solution257:

    def binaryTreePaths(self, root):

        paths = []

        def dfs(node, path):

            if not node:
                return

            path += str(node.val)

            if not node.left and not node.right:
                paths.append(path)
                return

            dfs(node.left, path+"->")
            dfs(node.right, path+"->")

        dfs(root,"")

        return paths


# --------------------------------------------------
# 10. Letter Combinations of Phone Number (LeetCode 17)
# --------------------------------------------------
"""
Description:
Return all possible letter combinations that the number could represent.

Example Input:
digits = "23"

Example Output:
["ad","ae","af","bd","be","bf","cd","ce","cf"]
"""

class Solution17:

    def letterCombinations(self, digits):

        if not digits:
            return []

        phone = {
            "2":"abc","3":"def","4":"ghi","5":"jkl",
            "6":"mno","7":"pqrs","8":"tuv","9":"wxyz"
        }

        res = []

        def dfs(i, path):

            if i == len(digits):
                res.append(path)
                return

            for c in phone[digits[i]]:
                dfs(i+1, path+c)

        dfs(0,"")

        return res


if __name__ == "__main__":
    print("DFS practice file with 10 LeetCode problems.")


# ============================================================================
# PROBLEM INDEX - QUICK REFERENCE
# ============================================================================
"""
================================================================================
                    LEETCODE DFS PROBLEMS - CONTENT INDEX
================================================================================

DEPTH-FIRST SEARCH (DFS) PROBLEMS
-------------------------------------------------
Line 13   | Problem 1:  Number of Islands (LeetCode 200)
Line 56   | Problem 2:  Max Area of Island (LeetCode 695)
Line 92   | Problem 3:  Flood Fill (LeetCode 733)
Line 130  | Problem 4:  Clone Graph (LeetCode 133)
Line 174  | Problem 5:  Course Schedule (LeetCode 207)
Line 224  | Problem 6:  Pacific Atlantic Water Flow (LeetCode 417)
Line 274  | Problem 7:  Surrounded Regions (LeetCode 130)
Line 326  | Problem 8:  Path Sum (LeetCode 112)
Line 366  | Problem 9:  Binary Tree Paths (LeetCode 257)
Line 410  | Problem 10: Letter Combinations of Phone Number (LeetCode 17)

================================================================================
                          TOTAL: 10 DFS PROBLEMS
================================================================================

PROBLEM CATEGORIES:
-------------------
GRID TRAVERSAL (DFS on 2D Arrays):
  • Problem 1: Number of Islands - Count connected components
  • Problem 2: Max Area of Island - Find largest connected area
  • Problem 3: Flood Fill - Color fill algorithm
  • Problem 6: Pacific Atlantic Water Flow - Multi-source DFS
  • Problem 7: Surrounded Regions - Boundary-based DFS

GRAPH TRAVERSAL:
  • Problem 4: Clone Graph - Deep copy with DFS
  • Problem 5: Course Schedule - Cycle detection in directed graph

TREE TRAVERSAL:
  • Problem 8: Path Sum - Root-to-leaf path finding
  • Problem 9: Binary Tree Paths - Collect all root-to-leaf paths

BACKTRACKING VARIANT:
  • Problem 10: Letter Combinations - Generate all combinations with DFS

KEY DFS PATTERNS:
-----------------
1. GRID DFS: Mark visited cells, explore 4 directions
2. GRAPH DFS: Use visited set, detect cycles
3. TREE DFS: Base case at leaves, track path from root
4. BACKTRACKING DFS: Build solution incrementally, explore all branches

TIME COMPLEXITY: O(V + E) for graphs, O(m×n) for grids
SPACE COMPLEXITY: O(depth) for recursion stack

HOW TO USE THIS INDEX:
----------------------
1. Use Cmd/Ctrl+G to go to specific line number
2. Each problem includes description, examples, and solution
3. Practice recognizing when to use DFS vs BFS

================================================================================
"""
