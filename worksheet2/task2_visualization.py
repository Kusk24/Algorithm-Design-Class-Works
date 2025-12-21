# Visualization of k-combinations recursion tree
# Shows how task2.py generates combinations with exactly k ones

import sys
sys.setrecursionlimit(10000)

n = 4
k = 2
x = [0] * n
indent_level = 0

def comb(i, count):
    global indent_level
    
    indent = "│ " * indent_level
    print(f"{indent}comb(i={i}, count={count}) → x={x}")
    
    if i == n:
        if count == k:
            print(f"{indent}  ✓ BASE CASE: count={count} == k={k}")
            print(f"{indent}  ✓ OUTPUT: {' '.join(map(str, x))}")
            return 1
        else:
            print(f"{indent}  ✗ BASE CASE: count={count} ≠ k={k} (rejected)")
            return 0
    
    # Branch 1: x[i] = 0
    print(f"{indent}  ├─ Setting x[{i}] = 0 (count stays {count})")
    x[i] = 0
    indent_level += 1
    c1 = comb(i + 1, count)
    indent_level -= 1
    
    # Branch 2: x[i] = 1
    print(f"{indent}  └─ Setting x[{i}] = 1 (count becomes {count + 1})")
    x[i] = 1
    indent_level += 1
    c2 = comb(i + 1, count + 1)
    indent_level -= 1
    
    total = c1 + c2
    print(f"{indent}  → Returning {c1} + {c2} = {total}")
    return total

print("=" * 70)
print(f"K-COMBINATIONS VISUALIZATION (n={n}, k={k})")
print(f"Finding all sequences with exactly {k} ones out of {n} positions")
print("=" * 70)
total = comb(0, 0)
print("=" * 70)
print(f"\nTotal valid combinations = {total}")
print(f"This matches C({n},{k}) = {n}!/({k}!×{n-k}!) = {total}")

print("\n" + "=" * 70)
print("TREE STRUCTURE DIAGRAM:")
print("=" * 70)
print("""
                      comb(0, count=0)
                     /              \\
               x[0]=0                x[0]=1
              count=0                count=1
                 /                      \\
          comb(1,0)                  comb(1,1)
           /      \\                    /      \\
      x[1]=0    x[1]=1            x[1]=0    x[1]=1
      cnt=0     cnt=1             cnt=1     cnt=2
        /         \\                 /         \\
   comb(2,0)   comb(2,1)      comb(2,1)   comb(2,2)
     / \\         / \\            / \\         / \\
    ... ...     ... ...        ... ...     ... ...

Legend:
  ✓ = Valid (count == k, gets printed)
  ✗ = Invalid (count ≠ k, rejected)

Only paths that reach count=k at the end are printed.
Pruning happens naturally: if count > k before reaching the end,
that branch will never produce valid combinations.
""")

print("\n" + "=" * 70)
print("VALID COMBINATIONS ONLY:")
print("=" * 70)
x = [0] * n

def comb_print_only(i, count):
    if i == n:
        if count == k:
            print(*x)
            return 1
        return 0
    
    x[i] = 0
    c1 = comb_print_only(i + 1, count)
    
    x[i] = 1
    c2 = comb_print_only(i + 1, count + 1)
    
    return c1 + c2

total = comb_print_only(0, 0)
print(f"\nTotal: {total} combinations")
