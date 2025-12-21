# Visualization of the Balanced Split Problem
# Shows how the recursion explores all partitions to find minimal difference

import sys
sys.setrecursionlimit(10000)

values = [1, 2, 3, 4]  # Example: small array for visualization
n = len(values)
x = [0] * n
indent_level = 0
call_count = 0

def comb(i):
    global indent_level, call_count
    call_count += 1
    
    indent = "│ " * indent_level
    
    # Show current state
    print(f"{indent}comb({i}) called, x = {x}")
    
    total1 = 0
    total2 = 0
    
    if i == n:
        # Calculate partition sums
        group0 = []
        group1 = []
        for j in range(n):
            if x[j] == 0:
                total1 += values[j]
                group0.append(values[j])
            else:
                total2 += values[j]
                group1.append(values[j])
        
        diff = abs(total1 - total2)
        print(f"{indent}  ✓ BASE CASE:")
        print(f"{indent}    Group 0: {group0} → sum = {total1}")
        print(f"{indent}    Group 1: {group1} → sum = {total2}")
        print(f"{indent}    Difference = |{total1} - {total2}| = {diff}")
        return diff
    
    else:
        # Try assigning to group 0
        print(f"{indent}  ├─ Assign values[{i}]={values[i]} to Group 0 (x[{i}]=0)")
        x[i] = 0
        indent_level += 1
        diff0 = comb(i + 1)
        indent_level -= 1
        
        # Try assigning to group 1
        print(f"{indent}  └─ Assign values[{i}]={values[i]} to Group 1 (x[{i}]=1)")
        x[i] = 1
        indent_level += 1
        diff1 = comb(i + 1)
        indent_level -= 1
        
        min_diff = min(diff0, diff1)
        print(f"{indent}  → min({diff0}, {diff1}) = {min_diff}")
        return min_diff

print("=" * 70)
print(f"BALANCED SPLIT VISUALIZATION")
print(f"Input array: {values}")
print(f"Goal: Split into 2 groups to minimize |sum(group0) - sum(group1)|")
print("=" * 70)
print()

ans = comb(0)

print()
print("=" * 70)
print(f"RESULT: Minimal Difference = {ans}")
print(f"Total recursive calls: {call_count}")
print("=" * 70)

print("\n" + "=" * 70)
print("EXPLANATION:")
print("=" * 70)
print("""
The algorithm explores ALL possible partitions using binary choices:
- x[i] = 0 means values[i] goes to Group 0
- x[i] = 1 means values[i] goes to Group 1

For each element, it tries BOTH groups and returns the minimum difference.

Example trace for values = [1, 2, 3, 4]:
                    
                      comb(0)
                     /        \\
              x[0]=0            x[0]=1
             (1→G0)             (1→G1)
               /                    \\
          comb(1)                 comb(1)
           /    \\                  /    \\
      x[1]=0  x[1]=1          x[1]=0  x[1]=1
      (2→G0)  (2→G1)          (2→G0)  (2→G1)
        /        \\              /        \\
      ...       ...           ...       ...

At each leaf (i==n), calculate:
  |sum(Group0) - sum(Group1)|

Then backtrack, taking minimum at each level.

OPTIMAL PARTITIONS for [1, 2, 3, 4]:
  - Group 0: {1, 4} → sum = 5
  - Group 1: {2, 3} → sum = 5
  - Difference = 0 ← BEST!

Time Complexity: O(2^n) - tries all 2^n possible partitions
Space Complexity: O(n) - recursion depth
""")

print("\n" + "=" * 70)
print("ALL POSSIBLE PARTITIONS:")
print("=" * 70)

def show_all_partitions():
    x = [0] * n
    results = []
    
    def comb_collect(i):
        if i == n:
            group0 = [values[j] for j in range(n) if x[j] == 0]
            group1 = [values[j] for j in range(n) if x[j] == 1]
            sum0 = sum(group0)
            sum1 = sum(group1)
            diff = abs(sum0 - sum1)
            results.append((group0, sum0, group1, sum1, diff))
            return diff
        
        x[i] = 0
        comb_collect(i + 1)
        x[i] = 1
        comb_collect(i + 1)
    
    comb_collect(0)
    
    # Sort by difference
    results.sort(key=lambda r: r[4])
    
    for idx, (g0, s0, g1, s1, diff) in enumerate(results, 1):
        marker = " ← OPTIMAL" if diff == results[0][4] else ""
        print(f"{idx:2}. Group0: {str(g0):15} sum={s0:2}  |  "
              f"Group1: {str(g1):15} sum={s1:2}  |  diff={diff}{marker}")

show_all_partitions()
