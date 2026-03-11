"""
================================================================================
CHESS & BOARD GAME ALGORITHM PROBLEMS
================================================================================
Author: Algorithm Design Course
Topics: Backtracking, BFS, DFS, Dynamic Programming, State Space Search
================================================================================

This file contains 9 classic algorithm problems inspired by chess and board games.
Each problem demonstrates a fundamental algorithm technique used in your course.

TABLE OF CONTENTS:
1. N-Queens (Backtracking)
2. N-Queens II - Count Solutions (Backtracking)
3. Knight Minimum Moves (BFS)
4. Knight Probability in Chessboard (Dynamic Programming)
5. Queens That Can Attack the King (Simulation)
6. Rook Captures (Directional Search)
7. Battleships in a Board (DFS/Connected Components)
8. Snakes and Ladders (BFS - State Space)
9. Sliding Puzzle (BFS/A*/IDA*)

================================================================================
"""

from collections import deque
from typing import List, Set, Tuple
import heapq

# ============================================================================
# PROBLEM 1: N-QUEENS
# ============================================================================
"""
LEETCODE 51 — N-QUEENS

PROBLEM:
--------
Place N queens on an N×N chessboard so that no two queens attack each other.
A queen can attack horizontally, vertically, and diagonally.
Return all distinct solutions.

EXAMPLE:
--------
Input: n = 4
Output: 
[
 [".Q..",
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",
  "Q...",
  "...Q",
  ".Q.."]
]

ALGORITHM USED:
--------------
★ BACKTRACKING (Constraint Satisfaction with Pruning)

HOW THE ALGORITHM WORKS:
-----------------------
1. Start with empty board, place queens row by row
2. For each row, try placing queen in each column (0 to N-1)
3. Before placing queen, check if position is safe:
   - No queen in same column
   - No queen on diagonal (row-col) direction
   - No queen on anti-diagonal (row+col) direction
4. If safe, place queen and recursively solve next row
5. If recursive call succeeds, solution found
6. If no column works in current row, BACKTRACK to previous row
7. Continue until all N queens placed or all possibilities exhausted

WHY BACKTRACKING:
----------------
- Need to explore all possible placements
- Can prune invalid branches early (queens attacking each other)
- Constraint satisfaction problem
- Build solution incrementally
- Avoids generating 2^(N²) invalid boards

KEY INSIGHT:
-----------
- Instead of trying all 2^(N²) possible placements
- We place one queen per row (reduces to N^N)
- With constraint checking, prunes to ~N! actual checks

TIME COMPLEXITY: O(N!) - but with pruning, much faster in practice
SPACE COMPLEXITY: O(N²) for board + O(N) for recursion stack

RELATED TO YOUR COURSE:
----------------------
This is exactly like your Week 8 - Worksheet 8: n_queens.py
"""

def solveNQueens(n: int) -> List[List[str]]:
    """
    Solve N-Queens problem using backtracking.
    
    Args:
        n: Size of the board (n x n)
    
    Returns:
        List of all valid board configurations
    """
    def is_safe(row, col):
        """Check if placing queen at (row, col) is safe"""
        # Check column
        for r in range(row):
            if board[r][col] == 'Q':
                return False
        
        # Check upper-left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r][c] == 'Q':
                return False
            r -= 1
            c -= 1
        
        # Check upper-right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c < n:
            if board[r][c] == 'Q':
                return False
            r -= 1
            c += 1
        
        return True
    
    def backtrack(row):
        """Place queens row by row"""
        if row == n:
            # Found a valid solution
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'  # Place queen
                backtrack(row + 1)     # Recurse to next row
                board[row][col] = '.'  # Backtrack
    
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0)
    return result


def solveNQueens_optimized(n: int) -> List[List[str]]:
    """
    Optimized N-Queens using sets to track attacked positions.
    This is faster than checking the board each time.
    """
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            # Check if current position is under attack
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            # Recurse
            backtrack(row + 1)
            
            # Backtrack
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    cols = set()      # Columns under attack
    diag1 = set()     # Diagonals (row - col)
    diag2 = set()     # Anti-diagonals (row + col)
    backtrack(0)
    return result


# ============================================================================
# PROBLEM 2: N-QUEENS II (COUNT ONLY)
# ============================================================================
"""
LEETCODE 52 — N-QUEENS II

PROBLEM:
--------
Same as N-Queens, but return only the number of distinct solutions.

EXAMPLE:
--------
Input: n = 4
Output: 2

Input: n = 1
Output: 1

ALGORITHM USED:
--------------
★ BACKTRACKING (Optimized - Count Only, No Board Storage)

HOW THE ALGORITHM WORKS:
-----------------------
1. Use same backtracking approach as N-Queens I
2. But instead of storing board configurations:
   - Just increment counter when N queens placed
   - Use sets to track attacked columns/diagonals
3. Three sets track constraints:
   - cols: blocked columns
   - diag1: blocked diagonals (row - col)
   - diag2: blocked anti-diagonals (row + col)
4. When placing queen at (row, col):
   - Add col to cols set
   - Add (row - col) to diag1 set
   - Add (row + col) to diag2 set
5. When backtracking, remove from all three sets
6. Return total count at the end

WHY OPTIMIZED:
-------------
- No need to build/store actual board strings
- Set operations are O(1) for constraint checking
- Reduced space complexity (no board storage)
- Faster than building strings repeatedly

TIME COMPLEXITY: O(N!) - same recursion tree
SPACE COMPLEXITY: O(N) - only sets, no board storage
"""

def totalNQueens(n: int) -> int:
    """
    Count total number of N-Queens solutions.
    
    Args:
        n: Size of the board
    
    Returns:
        Number of distinct solutions
    """
    def backtrack(row):
        if row == n:
            return 1  # Found one solution
        
        count = 0
        for col in range(n):
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                # Place queen
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                
                # Count solutions from this state
                count += backtrack(row + 1)
                
                # Backtrack
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)
        
        return count
    
    cols = set()
    diag1 = set()
    diag2 = set()
    return backtrack(0)


# ============================================================================
# PROBLEM 3: KNIGHT MINIMUM MOVES
# ============================================================================
"""
LEETCODE 1197 — MINIMUM KNIGHT MOVES

PROBLEM:
--------
A knight starts at (0, 0) on an infinite chessboard.
The knight can move in 8 directions: (±2, ±1) or (±1, ±2)
Find the minimum number of moves to reach (x, y).

EXAMPLE:
--------
Input: x = 5, y = 5
Output: 4
Explanation: (0,0) → (2,1) → (4,2) → (3,4) → (5,5)

Input: x = 2, y = 1
Output: 1

ALGORITHM USED:
--------------
★ BFS (Breadth-First Search) - Level-by-Level Exploration

HOW THE ALGORITHM WORKS:
-----------------------
1. Start from position (0, 0) with 0 moves
2. Use a queue to store states: (row, col, moves)
3. Use a set to track visited positions (avoid revisiting)
4. For each position, try all 8 knight moves:
   - (±2, ±1) horizontal first
   - (±1, ±2) vertical first
5. Add valid unvisited positions to queue
6. Process queue level by level (BFS guarantees shortest path)
7. When target (x, y) reached, return move count
8. Optimization: Use symmetry - work in first quadrant only
   - Knight moves are symmetric in all quadrants
   - Convert (x, y) to (|x|, |y|)

WHY BFS (NOT DFS or A*):
-----------------------
- BFS explores all moves at distance k before distance k+1
- First time we reach target = shortest path
- DFS might find longer path first
- A* possible but BFS is simpler for unweighted graphs

VISUALIZATION:
-------------
Level 0: (0,0)
Level 1: 8 positions reachable in 1 move
Level 2: Positions reachable in 2 moves
... continue until target found

TIME COMPLEXITY: O(|x| * |y|) - may visit all positions in search space
SPACE COMPLEXITY: O(|x| * |y|) - queue and visited set

RELATED TO YOUR COURSE:
----------------------
This is exactly like Week 8 - Maze Running (BFS for shortest path)
"""

def minKnightMoves(x: int, y: int) -> int:
    """
    Find minimum moves for knight to reach (x, y) from (0, 0).
    
    Args:
        x, y: Target coordinates
    
    Returns:
        Minimum number of moves
    """
    # Knight moves: 8 possible directions
    directions = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    
    # Use symmetry: work in first quadrant only
    x, y = abs(x), abs(y)
    
    # BFS
    queue = deque([(0, 0, 0)])  # (current_x, current_y, moves)
    visited = {(0, 0)}
    
    while queue:
        cx, cy, moves = queue.popleft()
        
        if cx == x and cy == y:
            return moves
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Optimization: stay near target (don't go too far in wrong direction)
            if (nx, ny) not in visited and -2 <= nx <= x + 2 and -2 <= ny <= y + 2:
                visited.add((nx, ny))
                queue.append((nx, ny, moves + 1))
    
    return -1  # Should never reach here


# ============================================================================
# PROBLEM 4: KNIGHT PROBABILITY IN CHESSBOARD
# ============================================================================
"""
LEETCODE 688 — KNIGHT PROBABILITY IN CHESSBOARD

PROBLEM:
--------
A knight is placed on an n×n chessboard at (row, col).
The knight makes k moves randomly (each of 8 directions equally likely).
Find the probability that the knight remains on the board after k moves.

EXAMPLE:
--------
Input: n = 3, k = 2, row = 0, col = 0
Output: 0.0625
Explanation: 
- 8 valid moves from (0,0)
- After 2 random moves, only 1/16 paths keep knight on board

ALGORITHM USED:
--------------
★ DYNAMIC PROGRAMMING (3D DP) - Probability Accumulation

HOW THE ALGORITHM WORKS:
-----------------------
1. Define state: dp[r][c][moves] = probability at position (r,c) after 'moves'
2. Base case: dp[row][col][0] = 1.0 (start here with 100% probability)
3. For each move from 1 to k:
   a. Create new DP table for current move
   b. For each position (r, c) on board:
      - Try all 8 knight moves
      - If move lands on board at (nr, nc):
        * Add probability: dp[nr][nc][m] += dp[r][c][m-1] / 8.0
        * (Divide by 8 because each move equally likely)
4. After k moves, sum all probabilities on board
5. Return total probability

WHY DP:
-------
- Overlapping subproblems: same position reached via different paths
- Optimal substructure: P(pos, k moves) = sum of P(previous, k-1 moves) / 8
- Need to track all possible states and accumulate probabilities
- Memoization alternative: could use dict with (r, c, moves) as key

KEY INSIGHT:
-----------
- Probability spreads out like a wave from starting position
- Each move splits probability equally among 8 directions
- Positions off board have 0 probability
- Final answer = sum of all on-board probabilities after k moves

SPACE OPTIMIZATION:
------------------
- Can use only two 2D arrays (previous and current move)
- Reduces space from O(n² * k) to O(n²)

TIME COMPLEXITY: O(n² * k) - for each of k moves, check all n² positions
SPACE COMPLEXITY: O(n² * k) or O(n²) if optimized

RELATED TO YOUR COURSE:
----------------------
Similar to your DP problems (Weeks 3-7), but with probabilities instead of counts
"""

def knightProbability(n: int, k: int, row: int, column: int) -> float:
    """
    Calculate probability knight stays on board after k moves.
    
    Args:
        n: Board size (n x n)
        k: Number of moves
        row, column: Starting position
    
    Returns:
        Probability (0.0 to 1.0)
    """
    directions = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    
    # dp[r][c] = probability of being at (r,c)
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0
    
    # For each move
    for move in range(k):
        new_dp = [[0.0] * n for _ in range(n)]
        
        for r in range(n):
            for c in range(n):
                if dp[r][c] > 0:
                    # Try all 8 knight moves
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            # Each move has 1/8 probability
                            new_dp[nr][nc] += dp[r][c] / 8.0
        
        dp = new_dp
    
    # Sum all probabilities (knight is somewhere on board)
    return sum(sum(row) for row in dp)


# ============================================================================
# PROBLEM 5: QUEENS THAT CAN ATTACK THE KING
# ============================================================================
"""
LEETCODE 1222 — QUEENS THAT CAN ATTACK THE KING

PROBLEM:
--------
Given positions of queens and a king on 8×8 board, return coordinates of
queens that can attack the king.

A queen attacks in 8 directions: horizontal, vertical, and diagonal.
Return the closest queen in each direction.

EXAMPLE:
--------
Input: 
queens = [[0,1],[1,0],[4,0],[0,4],[3,3],[2,4]]
king = [0,0]

Output: [[0,1],[1,0],[3,3]]

ALGORITHM USED:
--------------
★ SIMULATION / 8-DIRECTION RAY CASTING

HOW THE ALGORITHM WORKS:
-----------------------
1. Store all queen positions in a set for O(1) lookup
2. Define 8 direction vectors from king:
   - Horizontal: (-1,0), (1,0)
   - Vertical: (0,-1), (0,1)
   - Diagonal: (-1,-1), (-1,1), (1,-1), (1,1)
3. For each direction:
   a. Start from king's position
   b. Move step by step in that direction
   c. Check if position contains a queen
   d. If queen found, add to result and stop in this direction
   e. If out of bounds (0-7 range), stop in this direction
4. Return all queens found (max 8, one per direction)

WHY THIS APPROACH:
-----------------
- Queen attacks in straight lines (rays)
- Can't attack "through" another piece
- So only closest queen in each direction matters
- Ray casting naturally finds closest piece

VISUALIZATION:
-------------
```
  Q . . . . . . .
  . \ . . . . . .
  . . K → → → → Q
  . . ↓ . . . . .
  . . Q . . . . .
```
King at (2,2):
- Northwest: Q at (0,0) ✓
- East: Q at (2,6) ✓  
- South: Q at (4,2) ✓

TIME COMPLEXITY: O(n) where n = board size (8 in chess)
                 Each direction scans at most n cells
SPACE COMPLEXITY: O(1) - only storing result list (max 8 queens)

RELATED TO YOUR COURSE:
----------------------
Similar to directional search in maze problems
"""

def queensAttacktheKing(queens: List[List[int]], king: List[int]) -> List[List[int]]:
    """
    Find all queens that can attack the king.
    
    Args:
        queens: List of queen positions [[r1,c1], [r2,c2], ...]
        king: King position [r, c]
    
    Returns:
        List of queens that can attack king
    """
    # 8 directions: up, down, left, right, 4 diagonals
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # vertical, horizontal
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
    ]
    
    queen_set = set(map(tuple, queens))  # For O(1) lookup
    result = []
    
    kr, kc = king
    
    # Search in each direction
    for dr, dc in directions:
        r, c = kr + dr, kc + dc
        
        # Move in this direction until we find a queen or go off board
        while 0 <= r < 8 and 0 <= c < 8:
            if (r, c) in queen_set:
                result.append([r, c])
                break  # Found closest queen in this direction
            r += dr
            c += dc
    
    return result


# ============================================================================
# PROBLEM 6: AVAILABLE CAPTURES FOR ROOK
# ============================================================================
"""
LEETCODE 999 — AVAILABLE CAPTURES FOR ROOK

PROBLEM:
--------
Given an 8×8 chessboard with one rook 'R', bishops 'B', and pawns 'p'.
A rook can move horizontally or vertically.
Count how many pawns the rook can capture.
(Bishops block the rook's path)

EXAMPLE:
--------
Input: [
  [".",".",".",".",".",".",".","."],
  [".","p","p","p","p","p",".","."],
  [".","p","p","B","p","p",".","."],
  [".","p","B","R","B","p",".","."],
  ...
]
Output: 3

ALGORITHM USED:
--------------
★ DIRECTIONAL SEARCH (4-Direction Ray Casting)

HOW THE ALGORITHM WORKS:
-----------------------
1. Find rook's position (marked 'R') on 8x8 board
2. Define 4 direction vectors (rook moves only horizontally/vertically):
   - Up: (-1, 0)
   - Down: (1, 0)
   - Left: (0, -1)
   - Right: (0, 1)
3. For each direction:
   a. Start from rook's position
   b. Move step by step in that direction
   c. Check each cell:
      - If 'p' (pawn): increment count, stop this direction
      - If 'B' (bishop): stop this direction (blocked)
      - If '.': continue searching
      - If out of bounds: stop this direction
4. Return total count of capturable pawns

WHY THIS APPROACH:
-----------------
- Rook captures first piece in each direction
- Bishops block the rook's path (cannot capture through them)
- Must check all 4 directions independently
- Stop at first piece (pawn or bishop)

VISUALIZATION:
-------------
```
. . . p . . . .  ← cannot capture (no path)
. p p B p p . .  ← can capture 2 pawns (left + right of B)
. p B R B p . .  ← blocked by bishops
. p . . . p . .  ← can capture 1 pawn (down)
```
Rook can capture: left p, right p, down p = 3 pawns

TIME COMPLEXITY: O(1) - board always 8x8, max 4 * 8 = 32 cells checked
SPACE COMPLEXITY: O(1) - only storing count and position
"""

def numRookCaptures(board: List[List[str]]) -> int:
    """
    Count pawns rook can capture.
    
    Args:
        board: 8x8 chess board
    
    Returns:
        Number of capturable pawns
    """
    # Find rook position
    rook_r, rook_c = 0, 0
    for r in range(8):
        for c in range(8):
            if board[r][c] == 'R':
                rook_r, rook_c = r, c
                break
    
    # 4 directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    captures = 0
    
    for dr, dc in directions:
        r, c = rook_r + dr, rook_c + dc
        
        # Move in this direction
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r][c] == 'B':
                break  # Bishop blocks
            if board[r][c] == 'p':
                captures += 1  # Found pawn
                break
            r += dr
            c += dc
    
    return captures


# ============================================================================
# PROBLEM 7: BATTLESHIPS IN A BOARD
# ============================================================================
"""
LEETCODE 419 — BATTLESHIPS IN A BOARD

PROBLEM:
--------
Given an m×n board with battleships 'X' and empty cells '.',
count the number of battleships.
Battleships are 1×k or k×1 rectangles with no adjacent battleships.

EXAMPLE:
--------
Input: [
  ["X",".",".","X"],
  [".",".",".","X"],
  [".",".",".","X"]
]
Output: 2

ALGORITHM USED:
--------------
★ DFS / CONNECTED COMPONENTS (Graph Traversal)

HOW THE ALGORITHM WORKS (DFS VERSION):
--------------------------------------
1. Iterate through all cells in the grid
2. When 'X' (ship) found and not visited:
   a. Increment ship counter
   b. Start DFS from this cell
   c. Mark current cell as visited
   d. Recursively visit all adjacent 'X' cells (up, down, left, right)
   e. Mark entire ship as visited
3. Continue until all cells checked
4. Return total ship count

HOW THE ALGORITHM WORKS (OPTIMIZED VERSION - O(1) SPACE):
--------------------------------------------------------
1. Count only "top-left" corners of ships
2. A cell is top-left corner if:
   - It's 'X' (part of ship)
   - No 'X' above it (not continuation of vertical ship)
   - No 'X' to its left (not continuation of horizontal ship)
3. Iterate through grid:
   - If cell is 'X' and has no 'X' above or left
   - Increment counter (found new ship)
4. Return count

WHY DFS:
--------
- Each battleship is a connected component of 'X' cells
- DFS naturally explores entire connected region
- Marks all cells of one ship before finding next ship
- Same technique as "number of islands" problem

VISUALIZATION:
-------------
```
X . . X    Ship 1★ (vertical)
. . . X    Ship 1 continues
. . . X    Ship 1 continues
X X . .    Ship 2★ (horizontal)
```
DFS: Visits entire first ship, marks it, then finds second ship
Optimized: Counts 2 top-left corners marked with ★

TIME COMPLEXITY: O(m * n) - visit each cell once
SPACE COMPLEXITY: O(m * n) for DFS recursion/visited, O(1) for optimized

RELATED TO YOUR COURSE:
----------------------
This is exactly like Week 13 - Connected Components (largest_cloud problem)
"""

def countBattleships(board: List[List[str]]) -> int:
    """
    Count battleships using DFS.
    
    Args:
        board: m x n board
    
    Returns:
        Number of battleships
    """
    if not board or not board[0]:
        return 0
    
    m, n = len(board), len(board[0])
    count = 0
    
    def dfs(r, c):
        """Mark entire battleship as visited"""
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != 'X':
            return
        
        board[r][c] = '.'  # Mark as visited
        
        # Check 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(m):
        for c in range(n):
            if board[r][c] == 'X':
                count += 1
                dfs(r, c)  # Mark entire battleship
    
    return count


def countBattleships_optimized(board: List[List[str]]) -> int:
    """
    O(1) space solution: count only top-left corners.
    
    A cell is a battleship's start if:
    - It's 'X'
    - No 'X' to its left
    - No 'X' above it
    """
    if not board or not board[0]:
        return 0
    
    m, n = len(board), len(board[0])
    count = 0
    
    for r in range(m):
        for c in range(n):
            if board[r][c] == 'X':
                # Check if it's the start of a battleship
                if (r == 0 or board[r-1][c] != 'X') and \
                   (c == 0 or board[r][c-1] != 'X'):
                    count += 1
    
    return count


# ============================================================================
# PROBLEM 8: SNAKES AND LADDERS
# ============================================================================
"""
LEETCODE 909 — SNAKES AND LADDERS

PROBLEM:
--------
Given an n×n board representing a snakes and ladders game.
- Board squares labeled 1 to n²
- Some squares have snakes/ladders (board[r][c] != -1 means teleport to that square)
- Roll dice (1-6) each turn
- Find minimum moves to reach square n²

EXAMPLE:
--------
Input: board = [
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,35,-1,-1,13,-1],
[-1,-1,-1,-1,-1,-1],
[-1,15,-1,-1,-1,-1]]
Output: 4

ALGORITHM USED:
--------------
★ BFS (State Space Search with Teleportation)

HOW THE ALGORITHM WORKS:
-----------------------
1. Convert board to 1D array mapping square → destination
   - If board[r][c] = -1: square leads to itself (normal)
   - Otherwise: snake/ladder teleports to board[r][c]
2. Handle zigzag numbering:
   - Odd rows: left to right (1,2,3...)
   - Even rows: right to left (12,11,10...)
3. BFS from square 1:
   a. Use queue with (square, moves) states
   b. Mark visited squares to avoid cycles
   c. For each square, try dice rolls 1-6:
      - Calculate next square (curr + roll)
      - If snake/ladder exists, teleport
      - If not visited, add to queue with moves+1
4. When square n² reached, return move count
5. If queue empty before reaching n², return -1 (impossible)

WHY BFS (NOT DFS):
-----------------
- Need MINIMUM moves (shortest path)
- BFS explores moves level by level
- First time we reach goal = shortest path
- DFS might find longer path first
- Each dice roll has equal cost (unweighted)

KEY INSIGHTS:
------------
- Snakes/ladders are "teleport edges" in the graph
- Graph nodes = board squares (1 to n²)
- Each node connects to up to 6 next nodes (dice 1-6)
- Snakes/ladders modify the destination
- Visited set prevents infinite loops (e.g., snake then ladder back)

VISUALIZATION:
-------------
```
Square:  1 → 2 → 3 → 4 ...
         ↓   ↓   Ladder↑
Roll 1-6 +   +   to 10
```
BFS explores all 1-move positions, then 2-move, etc.

TIME COMPLEXITY: O(n²) - visit each square at most once
SPACE COMPLEXITY: O(n²) - queue and visited set

RELATED TO YOUR COURSE:
----------------------
Week 13 - Flappy Bird (state space search with BFS)
"""

def snakesAndLadders(board: List[List[int]]) -> int:
    """
    Find minimum moves to reach end of board.
    
    Args:
        board: n x n board
    
    Returns:
        Minimum moves, or -1 if impossible
    """
    n = len(board)
    
    def get_position(square):
        """Convert square number (1-indexed) to board coordinates"""
        square -= 1  # Convert to 0-indexed
        row = n - 1 - square // n
        col = square % n
        
        # Alternate rows go right-to-left
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        
        return row, col
    
    # BFS
    queue = deque([(1, 0)])  # (current_square, moves)
    visited = {1}
    target = n * n
    
    while queue:
        square, moves = queue.popleft()
        
        if square == target:
            return moves
        
        # Try all dice rolls (1-6)
        for dice in range(1, 7):
            next_square = square + dice
            
            if next_square > target:
                continue
            
            # Check for snake or ladder
            r, c = get_position(next_square)
            if board[r][c] != -1:
                next_square = board[r][c]
            
            if next_square not in visited:
                visited.add(next_square)
                queue.append((next_square, moves + 1))
    
    return -1  # Cannot reach end


# ============================================================================
# PROBLEM 9: SLIDING PUZZLE
# ============================================================================
"""
LEETCODE 773 — SLIDING PUZZLE

PROBLEM:
--------
Given a 2×3 board with tiles 1-5 and an empty space 0.
Slide tiles into empty space to reach target configuration.

EXAMPLE:
--------
Input: [[1,2,3],[4,0,5]]
Output: 1
Explanation: Swap 0 and 5

Input: [[4,1,2],[5,0,3]]
Output: 5

ALGORITHM USED:
--------------
★ BFS (State Space with String Representation)
★ A* (Heuristic Search with Manhattan Distance)
★ IDA* (Memory-Efficient Iterative Deepening)

HOW THE ALGORITHM WORKS (BFS VERSION):
-------------------------------------
1. Represent board state as string (e.g., "123450")
2. Find position of empty space (0)
3. Define valid moves based on 0's position:
   - Position 0: can swap with 1,3
   - Position 1: can swap with 0,2,4
   - Position 2: can swap with 1,5
   - Position 3: can swap with 0,4
   - Position 4: can swap with 1,3,5
   - Position 5: can swap with 2,4
4. BFS exploration:
   a. Start with initial state, 0 moves
   b. For each state, generate all valid next states
   c. If target "123450" reached, return moves
   d. Add unvisited states to queue
   e. Mark visited to avoid cycles
5. Continue until target found or queue empty

HOW A* WORKS (OPTIMIZED WITH HEURISTIC):
----------------------------------------
1. Same state representation as BFS
2. Use priority queue ordered by: f(n) = g(n) + h(n)
   - g(n) = moves so far (cost)
   - h(n) = Manhattan distance heuristic
3. Manhattan distance: sum of distances each tile is from goal
4. A* explores most promising states first
5. Guarantees shortest path if h(n) is admissible (never overestimates)

WHY BFS/A*:
----------
- Need shortest move sequence
- State space search: each board configuration is a node
- Edges = valid tile swaps
- BFS: guarantees shortest but explores many states
- A*: much faster with good heuristic
- IDA*: same as A* but memory-efficient (like IDS)

KEY INSIGHTS:
------------
- Total possible states: (m*n)! permutations
- But only need to explore reachable states
- Half of 6! = 360 states unreachable in 2x3 puzzle
- String representation allows easy visited tracking
- Good heuristic (Manhattan) reduces explored states dramatically

VISUALIZATION:
-------------
```
Initial:  1 2 3     Target:  1 2 3
          4 0 5              4 5 0
                             
Moves: swap 0 with 5 → solved in 1 move
```

TIME COMPLEXITY: O((m*n)! * m*n) worst - visit all states
                 A* much better in practice with heuristic
SPACE COMPLEXITY: O((m*n)!) - store visited states in set

RELATED TO YOUR COURSE:
----------------------
Week 9 - 8puzzle_IDS_example.py and 8puzzle_IDAstar_example.py
This is EXACTLY THE SAME as your 8-puzzle assignment!
"""

def slidingPuzzle(board: List[List[int]]) -> int:
    """
    Solve sliding puzzle using BFS.
    
    Args:
        board: 2x3 board
    
    Returns:
        Minimum moves, or -1 if unsolvable
    """
    # Convert board to string for easy hashing
    start = ''.join(str(board[i][j]) for i in range(2) for j in range(3))
    target = "123450"
    
    if start == target:
        return 0
    
    # Possible moves for each position of 0
    # Index represents position in string, values are adjacent positions
    neighbors = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [1, 3, 5],
        5: [2, 4]
    }
    
    # BFS
    queue = deque([(start, 0)])  # (state, moves)
    visited = {start}
    
    while queue:
        state, moves = queue.popleft()
        
        # Find position of 0
        zero_pos = state.index('0')
        
        # Try all possible swaps
        for next_pos in neighbors[zero_pos]:
            # Swap
            state_list = list(state)
            state_list[zero_pos], state_list[next_pos] = state_list[next_pos], state_list[zero_pos]
            new_state = ''.join(state_list)
            
            if new_state == target:
                return moves + 1
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + 1))
    
    return -1  # Unsolvable


def slidingPuzzle_Astar(board: List[List[int]]) -> int:
    """
    Solve sliding puzzle using A* algorithm with Manhattan distance heuristic.
    More efficient than plain BFS.
    """
    start = tuple(board[i][j] for i in range(2) for j in range(3))
    target = (1, 2, 3, 4, 5, 0)
    
    if start == target:
        return 0
    
    def manhattan_distance(state):
        """Calculate Manhattan distance heuristic"""
        distance = 0
        for i in range(6):
            if state[i] != 0:
                # Current position
                curr_row, curr_col = i // 3, i % 3
                # Target position for this number
                target_val = state[i] - 1
                target_row, target_col = target_val // 3, target_val % 3
                distance += abs(curr_row - target_row) + abs(curr_col - target_col)
        return distance
    
    neighbors = {
        0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
        3: [0, 4], 4: [1, 3, 5], 5: [2, 4]
    }
    
    # Priority queue: (f_cost, moves, state)
    # f_cost = moves + heuristic
    heap = [(manhattan_distance(start), 0, start)]
    visited = {start: 0}
    
    while heap:
        f_cost, moves, state = heapq.heappop(heap)
        
        if state == target:
            return moves
        
        # If we've found a better path to this state, skip
        if moves > visited.get(state, float('inf')):
            continue
        
        zero_pos = state.index(0)
        
        for next_pos in neighbors[zero_pos]:
            # Swap
            state_list = list(state)
            state_list[zero_pos], state_list[next_pos] = state_list[next_pos], state_list[zero_pos]
            new_state = tuple(state_list)
            new_moves = moves + 1
            
            # Only explore if we haven't seen this state or found a better path
            if new_state not in visited or new_moves < visited[new_state]:
                visited[new_state] = new_moves
                new_f_cost = new_moves + manhattan_distance(new_state)
                heapq.heappush(heap, (new_f_cost, new_moves, new_state))
    
    return -1


# ============================================================================
# TEST CASES AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CHESS & BOARD GAME ALGORITHM PROBLEMS - TEST CASES")
    print("=" * 80)
    
    # Test 1: N-Queens
    print("\n1️⃣  N-QUEENS (Backtracking)")
    print("-" * 80)
    solutions = solveNQueens(4)
    print(f"N = 4, Number of solutions: {len(solutions)}")
    print("First solution:")
    for row in solutions[0]:
        print(row)
    
    # Test 2: N-Queens II
    print("\n2️⃣  N-QUEENS II - COUNT (Backtracking)")
    print("-" * 80)
    print(f"N = 4, Total solutions: {totalNQueens(4)}")
    print(f"N = 8, Total solutions: {totalNQueens(8)}")
    
    # Test 3: Knight Moves
    print("\n3️⃣  KNIGHT MINIMUM MOVES (BFS)")
    print("-" * 80)
    print(f"From (0,0) to (5,5): {minKnightMoves(5, 5)} moves")
    print(f"From (0,0) to (2,1): {minKnightMoves(2, 1)} moves")
    
    # Test 4: Knight Probability
    print("\n4️⃣  KNIGHT PROBABILITY (Dynamic Programming)")
    print("-" * 80)
    prob = knightProbability(3, 2, 0, 0)
    print(f"n=3, k=2, start=(0,0): Probability = {prob:.4f}")
    
    # Test 5: Queens Attack King
    print("\n5️⃣  QUEENS ATTACK KING (Simulation)")
    print("-" * 80)
    queens = [[0,1],[1,0],[4,0],[0,4],[3,3],[2,4]]
    king = [0,0]
    print(f"Queens: {queens}")
    print(f"King: {king}")
    print(f"Attacking queens: {queensAttacktheKing(queens, king)}")
    
    # Test 6: Rook Captures
    print("\n6️⃣  ROOK CAPTURES (Directional Search)")
    print("-" * 80)
    board = [
        [".",".",".",".",".",".",".","."],
        [".",".",".","p",".",".",".","."],
        [".",".",".","R",".",".",".","p"],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".","p",".",".",".","."],
        [".",".",".",".",".",".",".","."],
        [".",".",".",".",".",".",".","."]
    ]
    print(f"Rook can capture: {numRookCaptures(board)} pawns")
    
    # Test 7: Battleships
    print("\n7️⃣  BATTLESHIPS (DFS/Connected Components)")
    print("-" * 80)
    board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]
    print("Board:")
    for row in board:
        print(row)
    # Use optimized version to preserve board
    print(f"Number of battleships: {countBattleships_optimized([row[:] for row in board])}")
    
    # Test 8: Snakes and Ladders
    print("\n8️⃣  SNAKES AND LADDERS (BFS - State Space)")
    print("-" * 80)
    board = [
        [-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1],
        [-1,35,-1,-1,13,-1],
        [-1,-1,-1,-1,-1,-1],
        [-1,15,-1,-1,-1,-1]
    ]
    print(f"Minimum moves: {snakesAndLadders(board)}")
    
    # Test 9: Sliding Puzzle
    print("\n9️⃣  SLIDING PUZZLE (BFS/A*)")
    print("-" * 80)
    board1 = [[1,2,3],[4,0,5]]
    board2 = [[4,1,2],[5,0,3]]
    print(f"Board {board1}: {slidingPuzzle(board1)} moves (BFS)")
    print(f"Board {board2}: {slidingPuzzle_Astar(board2)} moves (A*)")
    
    print("\n" + "=" * 80)
    print("ALGORITHM SUMMARY")
    print("=" * 80)
    print("""
Algorithm           | Problems                  | When to Use
--------------------|---------------------------|----------------------------
Backtracking        | N-Queens, N-Queens II     | Constraint satisfaction
BFS                 | Knight Moves, Snakes      | Shortest path (unweighted)
DFS                 | Battleships               | Connected components
Dynamic Programming | Knight Probability        | Overlapping subproblems
Simulation          | Queens Attack, Rook       | Geometric/directional
A*/IDA*             | Sliding Puzzle (8-puzzle) | Shortest path with heuristic
    """)
    
    print("\n🎓 These problems cover all major topics from your Algorithm Design course!")
    print("📚 Practice these to master: Backtracking, BFS, DFS, DP, and State Space Search")
