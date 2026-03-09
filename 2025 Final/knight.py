from collections import deque

# Name : Win Yu Maung 
# ID : 6612054 
# Sec : 541


class State:
    def __init__(self, r, c, dist):
        self.r = r
        self.c = c
        self.dist = dist # Records how far from initial state [cite: 214]

def knight_quest(start_pos, target_pos, N):
    # Possible knight moves
    moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
    queue = deque([State(start_pos[0], start_pos[1], 0)])
    visited = set([(start_pos[0], start_pos[1])]) # Prevent repeat searches [cite: 226]

    while queue:
        curr = queue.popleft()
        if (curr.r, curr.c) == target_pos:
            return curr.dist
        
        for dr, dc in moves:
            nr, nc = curr.r + dr, curr.c + dc
            # Valid function logic: check if within board [cite: 210]
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(State(nr, nc, curr.dist + 1))
    return -1