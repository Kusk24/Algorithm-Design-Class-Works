# Visualization of the recursion tree for task1.py
# This shows how the binary combinations are generated

import sys
sys.setrecursionlimit(10000)

n = 3  # Small example for visualization
x = [0] * n
indent_level = 0

def comb(i):
    global indent_level
    
    # Print current state entering this call
    indent = "  " * indent_level
    print(f"{indent}comb({i}) called, x = {x}")
    
    if i == n:
        print(f"{indent}  → BASE CASE: Print {x}")
        print(f"{indent}  → OUTPUT: {' '.join(map(str, x))}")
        return
    
    # First branch: set x[i] = 0
    print(f"{indent}  Setting x[{i}] = 0")
    x[i] = 0
    indent_level += 1
    comb(i + 1)
    indent_level -= 1
    
    # Second branch: set x[i] = 1
    print(f"{indent}  Setting x[{i}] = 1")
    x[i] = 1
    indent_level += 1
    comb(i + 1)
    indent_level -= 1

print("=" * 60)
print("RECURSION TREE VISUALIZATION (n=3)")
print("=" * 60)
comb(0)
print("=" * 60)
print("\nTree Structure Diagram:")
print("""
                    comb(0)
                   /        \\
              x[0]=0        x[0]=1
                /              \\
           comb(1)           comb(1)
           /    \\            /    \\
      x[1]=0  x[1]=1    x[1]=0  x[1]=1
        /        \\        /        \\
    comb(2)   comb(2)  comb(2)   comb(2)
      /\\        /\\        /\\        /\\
  x[2]=0,1  x[2]=0,1  x[2]=0,1  x[2]=0,1
     |  |     |  |     |  |     |  |
   [0,0,0] [0,0,1] [0,1,0] [0,1,1] [1,0,0] [1,0,1] [1,1,0] [1,1,1]
   
Each path from root to leaf generates one binary sequence.
Total: 2^3 = 8 combinations
""")
