# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

MOVES = {
    0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
}

found = False

def heuristic(state):
    """Manhattan distance heuristic"""
    dist = 0
    for i in range(9):
        if state[i] != 0:
            goal_pos = state[i] - 1
            dist += abs(i // 3 - goal_pos // 3) + abs(i % 3 - goal_pos % 3)
    return dist


def dfs(state, g, bound, visited):
    global found

    f = g + heuristic(state)
    if f > bound:
        return f

    if state == GOAL:
        found = True
        return g

    min_exceed = float('inf')
    visited.add(state)
    zero = state.index(0)

    for nxt in MOVES[zero]:
        new_state = list(state)
        new_state[zero], new_state[nxt] = new_state[nxt], new_state[zero]
        new_state = tuple(new_state)

        if new_state not in visited:
            t = dfs(new_state, g + 1, bound, visited)
            if found:
                return t
            min_exceed = min(min_exceed, t)

    visited.remove(state)
    return min_exceed


def IDAstar(start):
    global found
    bound = heuristic(start)

    while True:
        visited = set()
        found = False
        t = dfs(start, 0, bound, visited)
        if found:
            return t
        bound = t


start_state = tuple(map(int, input().split()))
print(IDAstar(start_state))
