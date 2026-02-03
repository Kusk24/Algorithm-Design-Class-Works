# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import copy
from collections import deque

class state():
    def __init__(self, n):
        self.queens = [-1] * n
        self.col = 0

def printqueens(Q):
    n = len(Q)
    board = [['.']*n for i in range(n)]
    for j in range(n):
        board[Q[j]][j] = 'Q'
    for i in range(n):
        for j in range(n):
            print(board[i][j], end='')
        print()

def conflict(Q, i, j):
    if Q[i] == Q[j] or abs(Q[i] - Q[j]) == abs(i - j):
        return True
    else:
        return False

def is_safe(Q, col):
    for i in range(col):
        if conflict(Q, i, col):
            return False
    return True

def bfs_N_Queen(N):
    q = deque()
    solutions = []
    
    start = state(N)
    q.append(start)
    
    while q:
        current = q.popleft()
        
        if current.col == N:
            solutions.append(current.queens[:])
            continue
        
        col = current.col
        
        for i in range(N):
            child = copy.deepcopy(current)
            child.queens[col] = i
            if is_safe(child.queens, col):
                child.col += 1
                q.append(child)
    
    return solutions

if __name__ == "__main__":
    N = int(input("Enter N: "))
    
    solutions = bfs_N_Queen(N)
    print()
    if solutions:
        print(f"Found {len(solutions)} solution(s):")
        for i in range(len(solutions)):
            print(f"Solution {i+1}")
            printqueens(solutions[i])
            print()
    else:
        print("No solution exists.")