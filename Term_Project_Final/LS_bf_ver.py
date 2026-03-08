H, W = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(H)]

max_side = 0

for i in range(H):
    for j in range(W):
        # Only start if current cell is 0
        if matrix[i][j] == 0:
            size = 1

            # Try increasing square size
            while i + size <= H and j + size <= W:
                valid = True

                # Check all cells inside the square
                for x in range(i, i + size):
                    for y in range(j, j + size):
                        if matrix[x][y] == 1:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    max_side = max(max_side, size)
                    size += 1
                else:
                    break

print(max_side * max_side)