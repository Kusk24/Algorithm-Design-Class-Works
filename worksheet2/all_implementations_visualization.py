# Comprehensive Visualization of All Three Implementations
# No.3, No.4, and No.5 from Worksheet 2

import sys
sys.setrecursionlimit(10000)

print("=" * 80)
print("IMPLEMENTATION #3: BINARY COMBINATIONS WITH COUNT")
print("=" * 80)
print("Generates all 2^n binary combinations and counts them\n")

n = 3
x = [0] * n

def comb_no3(i):
    if i == n:
        print(*x)
        return 1
    x[i] = 0
    c1 = comb_no3(i + 1)
    x[i] = 1
    c2 = comb_no3(i + 1)
    return c1 + c2

total = comb_no3(0)
print(f"Total combinations = {total}")
print(f"This equals 2^{n} = {2**n}\n")

print("=" * 80)
print("TREE STRUCTURE (No.3):")
print("=" * 80)
print("""
                comb(0)
               /      \\
          x[0]=0      x[0]=1
            /            \\
        comb(1)        comb(1)
         /    \\         /    \\
    x[1]=0  x[1]=1  x[1]=0  x[1]=1
      /       \\       /       \\
    ...      ...     ...      ...

Each branch returns 1 for each combination
Total returned = sum of all 1's = 2^n
""")

print("\n" + "=" * 80)
print("IMPLEMENTATION #4: K-COMBINATIONS")
print("=" * 80)
print("Generates only combinations with exactly k ones\n")

n = 4
k = 2
x = [0] * n

def comb_no4(i, count):
    if i == n:
        if count == k:
            print(*x)
            return 1
        return 0
    x[i] = 0
    c1 = comb_no4(i + 1, count)
    x[i] = 1
    c2 = comb_no4(i + 1, count + 1)
    return c1 + c2

total = comb_no4(0, 0)
print(f"Total combinations = {total}")
print(f"This equals C({n},{k}) = {n}!/({k}!×{n-k}!) = {total}\n")

print("=" * 80)
print("TREE STRUCTURE (No.4):")
print("=" * 80)
print(f"""
                 comb(0, count=0)
                /              \\
          x[0]=0                x[0]=1
         count=0               count=1
            /                      \\
      comb(1,0)                 comb(1,1)
       /      \\                   /      \\
   x[1]=0   x[1]=1           x[1]=0   x[1]=1
   cnt=0    cnt=1            cnt=1    cnt=2
     ...      ...              ...      ...

Only paths that reach count={k} at the end return 1
Invalid paths (count≠{k}) return 0
""")

print("\n" + "=" * 80)
print("IMPLEMENTATION #5: TERNARY COMBINATIONS")
print("=" * 80)
print("Generates all 3^n ternary combinations (values: 0, 1, 2)\n")

n = 3
x = [0] * n

def comb_no5(i):
    if i == n:
        print(*x)
        return 1
    x[i] = 0
    c1 = comb_no5(i + 1)
    x[i] = 1
    c2 = comb_no5(i + 1)
    x[i] = 2
    c3 = comb_no5(i + 1)
    return c1 + c2 + c3

total = comb_no5(0)
print(f"Total combinations = {total}")
print(f"This equals 3^{n} = {3**n}\n")

print("=" * 80)
print("TREE STRUCTURE (No.5):")
print("=" * 80)
print("""
                    comb(0)
                /      |      \\
           x[0]=0   x[0]=1   x[0]=2
             /        |         \\
         comb(1)   comb(1)   comb(1)
         / | \\     / | \\     / | \\
       0  1  2    0  1  2   0  1  2
        ...       ...       ...

Each position has 3 choices: 0, 1, or 2
Total = 3 × 3 × 3 = 3^n combinations
""")

print("\n" + "=" * 80)
print("COMPARISON TABLE:")
print("=" * 80)
print(f"{'Implementation':<20} {'Choices/Position':<20} {'Total (n=3)':<15} {'Formula':<15}")
print("-" * 80)
print(f"{'No.3: Binary':<20} {'2 (0, 1)':<20} {2**3:<15} {'2^n':<15}")
print(f"{'No.4: K-Comb (k=2)':<20} {'2 (0, 1)':<20} {'6 (for n=4)':<15} {'C(n,k)':<15}")
print(f"{'No.5: Ternary':<20} {'3 (0, 1, 2)':<20} {3**3:<15} {'3^n':<15}")
print("=" * 80)

print("\n" + "=" * 80)
print("KEY DIFFERENCES:")
print("=" * 80)
print("""
No.3: Returns count of ALL binary combinations
      - No filtering, prints everything
      - Count = 2^n

No.4: Returns count of VALID k-combinations only
      - Filters by count parameter
      - Only prints when count == k at end
      - Count = C(n,k) = n!/(k!(n-k)!)

No.5: Returns count of ALL ternary combinations
      - THREE recursive calls instead of two
      - Each position can be 0, 1, OR 2
      - Count = 3^n
""")
