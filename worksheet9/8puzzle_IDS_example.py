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

def dfs(state, depth, limit, visited):
    if state == GOAL:
        return depth

    if depth == limit:
        return -1

    visited.add(state)
    zero = state.index(0)

    for nxt in MOVES[zero]:
        new_state = list(state)
        new_state[zero], new_state[nxt] = new_state[nxt], new_state[zero]
        new_state = tuple(new_state)

        if new_state not in visited:
            result = dfs(new_state, depth + 1, limit, visited)
            if result != -1:
                return result

    visited.remove(state)
    return -1


def IDS(start):
    depth = 0
    while True:
        visited = set()
        result = dfs(start, 0, depth, visited)
        if result != -1:
            return result
        depth += 1


start_state = tuple(map(int, input().split()))
print(IDS(start_state))
