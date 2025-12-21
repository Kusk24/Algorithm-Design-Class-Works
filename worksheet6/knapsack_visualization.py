# 0/1 KNAPSACK PROBLEM - VISUALIZATION & EXPLANATION
# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

"""
===================================================================================
0/1 KNAPSACK PROBLEM - DYNAMIC PROGRAMMING SOLUTION
===================================================================================

PROBLEM:
--------
Given:
- N items, each with weight w[i] and value v[i]
- A knapsack with maximum capacity M
Find: Maximum value you can carry without exceeding capacity M
Constraint: Each item can be taken at most once (0 or 1)

EXAMPLE:
--------
N = 4 items
M = 5 (max capacity)
Weights: [2, 1, 3, 2]
Values:  [12, 10, 20, 15]

Items:
  Item 0: weight=2, value=12
  Item 1: weight=1, value=10
  Item 2: weight=3, value=20
  Item 3: weight=2, value=15

DYNAMIC PROGRAMMING APPROACH:
------------------------------
dp[i][C] = maximum value using first i items with capacity C

Recurrence relation:
  dp[i][C] = max(
      dp[i-1][C],              // Skip item i-1
      v[i-1] + dp[i-1][C-w[i-1]]  // Take item i-1 (if fits)
  )

Base case: dp[0][C] = 0 for all C (no items = 0 value)
"""

print("=" * 80)
print("0/1 KNAPSACK - STEP BY STEP VISUALIZATION")
print("=" * 80)

# Example input
N = 4  # number of items
M = 5  # max capacity
w = [2, 1, 3, 2]  # weights
v = [12, 10, 20, 15]  # values

print(f"\nInput:")
print(f"  Number of items (N): {N}")
print(f"  Max capacity (M): {M}")
print(f"  Weights: {w}")
print(f"  Values:  {v}")
print()

# Print items
print("Items:")
for i in range(N):
    print(f"  Item {i}: weight={w[i]}, value={v[i]}")

# Create DP table
dp = [[0]*(M+1) for i in range(N+1)]

print("\n" + "=" * 80)
print("DP TABLE CONSTRUCTION")
print("=" * 80)

# Fill DP table with detailed steps
for i in range(1, N+1):
    print(f"\n--- Processing Item {i-1} (weight={w[i-1]}, value={v[i-1]}) ---")
    
    for C in range(0, M+1):
        # Option 1: Skip
        skip = dp[i-1][C]
        
        # Option 2: Take (if fits)
        take = 0
        if w[i-1] <= C:
            take = v[i-1] + dp[i-1][C - w[i-1]]
        
        # Choose max
        dp[i][C] = max(skip, take)
        
        # Print decision for this cell
        if C <= M:
            decision = ""
            if w[i-1] > C:
                decision = f"Can't take (weight {w[i-1]} > capacity {C})"
            elif take >= skip:
                decision = f"TAKE (value={v[i-1]} + dp[{i-1}][{C-w[i-1]}]={dp[i-1][C-w[i-1]]} = {take})"
            else:
                decision = f"SKIP (keep {skip})"
            
            print(f"  dp[{i}][{C}] = {dp[i][C]:2d}  |  {decision}")

# Print final DP table
print("\n" + "=" * 80)
print("FINAL DP TABLE")
print("=" * 80)
print("\nRows = items (0 to N), Columns = capacity (0 to M)")
print("\n    ", end="")
for C in range(M+1):
    print(f"C={C:2d} ", end="")
print()
print("    " + "-" * (5 * (M+1)))

for i in range(N+1):
    if i == 0:
        print(f"i=0 ", end="")
    else:
        print(f"i={i} ", end="")
    for C in range(M+1):
        print(f"{dp[i][C]:4d} ", end="")
    print()

print("\n" + "=" * 80)
print(f"ANSWER: Maximum value = {dp[N][M]}")
print("=" * 80)

# Trace back to find which items were selected
print("\nFinding which items were selected:")
selected_items = []
i = N
C = M

while i > 0 and C > 0:
    # If value came from taking the item
    if dp[i][C] != dp[i-1][C]:
        selected_items.append(i-1)
        C -= w[i-1]
    i -= 1

selected_items.reverse()
print(f"Selected items: {selected_items}")
print("\nDetails:")
total_weight = 0
total_value = 0
for idx in selected_items:
    print(f"  Item {idx}: weight={w[idx]}, value={v[idx]}")
    total_weight += w[idx]
    total_value += v[idx]
print(f"\nTotal weight: {total_weight} (capacity: {M})")
print(f"Total value: {total_value}")

print("\n" + "=" * 80)
print("VISUAL REPRESENTATION")
print("=" * 80)
print("\nKnapsack state:")
print("┌" + "─" * 50 + "┐")
for idx in selected_items:
    print(f"│ Item {idx}: weight={w[idx]}, value={v[idx]}" + " " * (50 - 25) + "│")
print("├" + "─" * 50 + "┤")
print(f"│ Total: weight={total_weight}/{M}, value={total_value}" + " " * (50 - 30) + "│")
print("└" + "─" * 50 + "┘")

print("\n" + "=" * 80)
print("TIME & SPACE COMPLEXITY")
print("=" * 80)
print("Time Complexity:  O(N × M)")
print("Space Complexity: O(N × M)")
print("\nOptimization: Can reduce space to O(M) using 1D array")
