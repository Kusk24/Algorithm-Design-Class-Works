"""
===============================================================================
OPTIMIZATION TECHNIQUES: BRUTE FORCE → MEMOIZATION → DYNAMIC PROGRAMMING
===============================================================================
A comprehensive guide to algorithm optimization strategies
===============================================================================
"""

# =============================================================================
# PART 1: WHAT IS BRUTE FORCE?
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                            BRUTE FORCE                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

DEFINITION:
━━━━━━━━━━━
Brute Force is an algorithmic approach that tries ALL possible solutions
and picks the best one. It's exhaustive search without any optimization.

CHARACTERISTICS:
━━━━━━━━━━━━━━━━
✓ Tries every possible combination/solution
✓ Guaranteed to find the correct answer (if implemented correctly)
✓ No clever optimizations
✓ Often recursive or uses nested loops
✓ Typically exponential time complexity (O(2^n), O(n!), etc.)

WHEN TO USE:
━━━━━━━━━━━━
✓ Small input sizes (n < 20)
✓ No better algorithm exists
✓ Understanding the problem first
✓ Correctness verification
✓ Teaching/learning purposes
✓ Quick prototyping

PROS:
━━━━━
✓ Simple to implement
✓ Always correct (if logic is right)
✓ Easy to understand and debug
✓ No need to identify patterns

CONS:
━━━━━
✗ Extremely slow for large inputs
✗ Often impractical/unusable
✗ Wastes computation on redundant work
✗ Exponential time complexity
✗ May not finish in reasonable time

EXAMPLES FROM CODEBASE:
━━━━━━━━━━━━━━━━━━━━━━━
- Generate all binary combinations (2^n combinations)
- Balance Split (try all partitions)
- Knapsack V1 (try all item combinations)
- Rod Cutting naive (try all cuts)
- Coin Change naive (try all coin combinations)
"""

# Example 1: Fibonacci - Brute Force
def fibonacci_brute_force(n):
    """
    TIME: O(2^n) - exponential!
    PROBLEM: Recalculates same values many times
    
    Tree for fib(5):
                    fib(5)
                   /      \
              fib(4)      fib(3)
             /     \      /     \
        fib(3)  fib(2) fib(2) fib(1)
        /   \    /  \   /  \
    fib(2) fib(1) ...  ...
    
    Notice: fib(3) calculated twice, fib(2) calculated 3 times!
    """
    if n <= 1:
        return n
    return fibonacci_brute_force(n-1) + fibonacci_brute_force(n-2)


# Example 2: Coin Change - Brute Force
def coin_change_brute_force(coins, amount):
    """
    TIME: O(C^amount) where C = number of coin types
    PROBLEM: Recalculates minCoins(amount-c) repeatedly
    
    For coins=[1,2,5], amount=11:
    - minCoins(11) calls minCoins(10), minCoins(9), minCoins(6)
    - minCoins(10) calls minCoins(9), minCoins(8), minCoins(5)
    - Notice: minCoins(9) called MULTIPLE times!
    """
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')
    
    min_coins = float('inf')
    for coin in coins:
        result = 1 + coin_change_brute_force(coins, amount - coin)
        min_coins = min(min_coins, result)
    
    return min_coins


# =============================================================================
# PART 2: WHAT IS MEMOIZATION?
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                           MEMOIZATION                                     ║
║                    (Top-Down Dynamic Programming)                         ║
╚══════════════════════════════════════════════════════════════════════════╝

DEFINITION:
━━━━━━━━━━━
Memoization is an optimization technique that STORES the results of expensive
function calls and REUSES them when the same inputs occur again.

KEY CONCEPT: "Calculate once, remember forever"

HOW IT WORKS:
━━━━━━━━━━━━━
1. Before computing, CHECK if result already exists in cache (memo)
2. If YES → return cached result immediately (no recursion!)
3. If NO → compute result, STORE in cache, then return

CHARACTERISTICS:
━━━━━━━━━━━━━━━━
✓ Top-Down approach (starts from problem, recurses to base cases)
✓ Uses recursion + caching
✓ Stores results in dictionary/array (memo table)
✓ Only computes each unique state ONCE
✓ Transforms exponential → polynomial time

WHEN TO USE:
━━━━━━━━━━━━
✓ Recursive solution with overlapping subproblems
✓ Can identify unique "states" for caching
✓ Want to keep recursive structure (natural)
✓ Not all states may be visited
✓ Easier to implement from recursive solution

PROS:
━━━━━
✓ Massive speedup (exponential → polynomial)
✓ Easy to add to existing recursive code
✓ Natural recursive formulation
✓ Only computes needed states
✓ Great for problems where not all states visited

CONS:
━━━━━
✗ Recursion overhead (function call stack)
✗ Extra memory for memo table
✗ Stack overflow risk for deep recursion
✗ Slightly harder to optimize space

MEMOIZATION PATTERN:
━━━━━━━━━━━━━━━━━━━━
def solve(state):
    # Step 1: Check cache
    if state in memo:
        return memo[state]
    
    # Step 2: Base case
    if base_condition:
        return base_value
    
    # Step 3: Recursive computation
    result = compute_using_recursion(...)
    
    # Step 4: Store in cache
    memo[state] = result
    return result
"""

# Example 1: Fibonacci - Memoization
def fibonacci_memoization(n, memo=None):
    """
    TIME: O(n) - each value calculated once!
    SPACE: O(n) for memo + O(n) for recursion stack
    
    Execution for fib(5):
    1. fib(5) → not in memo, calculate
    2. fib(4) → not in memo, calculate
    3. fib(3) → not in memo, calculate
    4. fib(2) → not in memo, calculate
    5. fib(1) → base case, return 1
    6. fib(0) → base case, return 0
    7. fib(2) AGAIN → IN MEMO! Return immediately
    8. fib(3) AGAIN → IN MEMO! Return immediately
    
    Result: Only 6 function calls instead of 15!
    """
    if memo is None:
        memo = {}
    
    # Check cache first
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and cache
    result = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    memo[n] = result
    return result


# Example 2: Coin Change - Memoization
def coin_change_memoization(coins, amount, memo=None):
    """
    TIME: O(amount * len(coins))
    SPACE: O(amount)
    
    STATE: amount (the remaining amount to make change for)
    Each unique amount is computed exactly once!
    """
    if memo is None:
        memo = {}
    
    # Check cache
    if amount in memo:
        return memo[amount]
    
    # Base cases
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')
    
    # Compute
    min_coins = float('inf')
    for coin in coins:
        result = 1 + coin_change_memoization(coins, amount - coin, memo)
        min_coins = min(min_coins, result)
    
    # Cache and return
    memo[amount] = min_coins
    return min_coins


# Example 3: Knapsack - Memoization
def knapsack_memoization(weights, values, capacity):
    """
    TIME: O(n * capacity)
    SPACE: O(n * capacity)
    
    STATE: (item_index, remaining_capacity)
    Each unique state computed exactly once!
    """
    memo = {}
    n = len(weights)
    
    def solve(i, cap):
        # Check cache
        if (i, cap) in memo:
            return memo[(i, cap)]
        
        # Base case
        if i == n or cap == 0:
            return 0
        
        # Skip item i
        skip = solve(i + 1, cap)
        
        # Take item i (if fits)
        take = 0
        if weights[i] <= cap:
            take = values[i] + solve(i + 1, cap - weights[i])
        
        # Cache and return
        result = max(skip, take)
        memo[(i, cap)] = result
        return result
    
    return solve(0, capacity)


# =============================================================================
# PART 3: WHAT IS DYNAMIC PROGRAMMING (TABULATION)?
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    DYNAMIC PROGRAMMING (TABULATION)                       ║
║                         (Bottom-Up Approach)                              ║
╚══════════════════════════════════════════════════════════════════════════╝

DEFINITION:
━━━━━━━━━━━
Dynamic Programming (Tabulation) is an optimization technique that solves
problems by breaking them into subproblems, solving them ITERATIVELY from
smallest to largest, and storing results in a TABLE.

KEY CONCEPT: "Start from base cases, build up to the answer"

HOW IT WORKS:
━━━━━━━━━━━━━
1. Define a DP table (array/matrix) to store subproblem solutions
2. Initialize base cases
3. Fill table iteratively in correct order (small → large)
4. Each cell computed using previously computed cells
5. Final answer is in dp[n] or dp[n][m]

CHARACTERISTICS:
━━━━━━━━━━━━━━━━
✓ Bottom-Up approach (starts from base cases, builds to answer)
✓ Iterative (uses loops, not recursion)
✓ Stores results in array/matrix (DP table)
✓ Computes ALL states (even if not needed)
✓ No recursion overhead

WHEN TO USE:
━━━━━━━━━━━━
✓ All states need to be computed
✓ Want to avoid recursion overhead
✓ Need maximum performance
✓ Want to optimize space easily
✓ Production code

PROS:
━━━━━
✓ No recursion overhead → faster
✓ No stack overflow risk
✓ Easier to optimize space (rolling array)
✓ Clear iteration order
✓ Better cache locality
✓ Preferred in competitive programming

CONS:
━━━━━
✗ Less intuitive than memoization
✗ Must compute ALL states (even unnecessary ones)
✗ Harder to derive from recursive solution
✗ Must carefully determine fill order

TABULATION PATTERN:
━━━━━━━━━━━━━━━━━━━
# Step 1: Create DP table
dp = [[base_value] * (cols) for _ in range(rows)]

# Step 2: Initialize base cases
dp[0][0] = base_case

# Step 3: Fill table iteratively
for i in range(start, end):
    for j in range(start, end):
        dp[i][j] = compute_from(dp[i-1][j], dp[i][j-1], etc.)

# Step 4: Return answer
return dp[n][m]
"""

# Example 1: Fibonacci - Tabulation
def fibonacci_tabulation(n):
    """
    TIME: O(n)
    SPACE: O(n) [can optimize to O(1)!]
    
    DP Table: dp[i] = fibonacci(i)
    
    i    | 0 | 1 | 2 | 3 | 4 | 5 |
    -----+---+---+---+---+---+---+
    dp[i]| 0 | 1 | 1 | 2 | 3 | 5 |
    
    Each cell computed from two previous cells!
    """
    if n <= 1:
        return n
    
    # Create DP table
    dp = [0] * (n + 1)
    
    # Base cases
    dp[0] = 0
    dp[1] = 1
    
    # Fill table
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]


# Space-Optimized Fibonacci
def fibonacci_tabulation_optimized(n):
    """
    TIME: O(n)
    SPACE: O(1) - only need last two values!
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1


# Example 2: Coin Change - Tabulation
def coin_change_tabulation(coins, amount):
    """
    TIME: O(amount * len(coins))
    SPACE: O(amount)
    
    DP Table: dp[i] = minimum coins to make amount i
    
    For coins=[1,2,5], amount=11:
    i    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10 |11 |
    -----+---+---+---+---+---+---+---+---+---+---+---+---+
    dp[i]| 0 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 3 | 3 | 2 | 3 |
    """
    # Create DP table
    dp = [float('inf')] * (amount + 1)
    
    # Base case
    dp[0] = 0
    
    # Fill table
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], 1 + dp[i - coin])
    
    return dp[amount] if dp[amount] != float('inf') else -1


# Example 3: Knapsack - Tabulation
def knapsack_tabulation(weights, values, capacity):
    """
    TIME: O(n * capacity)
    SPACE: O(n * capacity) [can optimize to O(capacity)!]
    
    DP Table: dp[i][c] = max value using items 0..i-1 with capacity c
    
    Recurrence:
    dp[i][c] = max(
        dp[i-1][c],                    # Skip item i-1
        values[i-1] + dp[i-1][c-w[i-1]]  # Take item i-1
    )
    """
    n = len(weights)
    
    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill table (base cases already 0)
    for i in range(1, n + 1):
        for c in range(capacity + 1):
            # Skip item i-1
            skip = dp[i-1][c]
            
            # Take item i-1 (if fits)
            take = 0
            if weights[i-1] <= c:
                take = values[i-1] + dp[i-1][c - weights[i-1]]
            
            dp[i][c] = max(skip, take)
    
    return dp[n][capacity]


# =============================================================================
# PART 4: COMPARISON TABLE
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         COMPREHENSIVE COMPARISON                          ║
╚══════════════════════════════════════════════════════════════════════════╝

┌──────────────────┬───────────────────┬───────────────────┬─────────────────┐
│   ASPECT         │   BRUTE FORCE     │   MEMOIZATION     │   TABULATION    │
├──────────────────┼───────────────────┼───────────────────┼─────────────────┤
│ Approach         │ Try everything    │ Top-Down          │ Bottom-Up       │
│ Implementation   │ Pure recursion    │ Recursion + cache │ Iteration+table │
│ Time Complexity  │ Exponential       │ Polynomial        │ Polynomial      │
│                  │ O(2^n), O(n!)     │ O(n*m)            │ O(n*m)          │
│ Space Complexity │ O(n) stack        │ O(n*m) memo+stack │ O(n*m) table    │
│ Recursion        │ Yes               │ Yes               │ No              │
│ Overhead         │ High              │ High              │ None            │
│ Stack Overflow   │ Possible          │ Possible          │ Never           │
│ Cache Strategy   │ None              │ On-demand         │ Pre-compute all │
│ States Computed  │ Many (redundant)  │ Only needed       │ All states      │
│ Ease of Coding   │ Very Easy         │ Easy              │ Moderate        │
│ Performance      │ Very Slow         │ Fast              │ Fastest         │
│ Space Optimize   │ N/A               │ Harder            │ Easier          │
│ Debugging        │ Easy              │ Moderate          │ Moderate        │
│ When to Use      │ Learning, n<20    │ Natural recursion │ Production code │
└──────────────────┴───────────────────┴───────────────────┴─────────────────┘

FIBONACCI COMPARISON (n=40):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Brute Force:   ~2 seconds, 331,160,281 calls
Memoization:   <0.001 seconds, 79 calls
Tabulation:    <0.001 seconds, 40 iterations

KNAPSACK COMPARISON (n=20, capacity=100):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Brute Force:   1,048,576 combinations (2^20)
Memoization:   ≤2,000 unique states (may not visit all)
Tabulation:    2,000 states computed (20 * 100)
"""


# =============================================================================
# PART 5: EVOLUTION EXAMPLE - COIN CHANGE PROBLEM
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         EVOLUTION: BRUTE FORCE → MEMOIZATION → TABULATION                ║
║                    Problem: Coin Change (coins=[1,2,5], amount=11)       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

print("=" * 80)
print("DEMONSTRATION: COIN CHANGE EVOLUTION")
print("=" * 80)
print("Problem: Minimum coins to make 11 with coins [1, 2, 5]")
print("Expected answer: 3 (5+5+1)")
print()

coins = [1, 2, 5]
amount = 11

# Method 1: Brute Force
call_count_bf = 0
def coin_change_bf_counting(coins, amt):
    global call_count_bf
    call_count_bf += 1
    if amt == 0:
        return 0
    if amt < 0:
        return float('inf')
    min_coins = float('inf')
    for coin in coins:
        min_coins = min(min_coins, 1 + coin_change_bf_counting(coins, amt - coin))
    return min_coins

result_bf = coin_change_bf_counting(coins, amount)
print(f"1. BRUTE FORCE:")
print(f"   Answer: {result_bf}")
print(f"   Function calls: {call_count_bf}")
print(f"   Analysis: MANY redundant calculations!")
print()

# Method 2: Memoization
call_count_memo = 0
def coin_change_memo_counting(coins, amt, memo=None):
    global call_count_memo
    call_count_memo += 1
    if memo is None:
        memo = {}
    if amt in memo:
        return memo[amt]
    if amt == 0:
        return 0
    if amt < 0:
        return float('inf')
    min_coins = float('inf')
    for coin in coins:
        min_coins = min(min_coins, 1 + coin_change_memo_counting(coins, amt - coin, memo))
    memo[amt] = min_coins
    return min_coins

result_memo = coin_change_memo_counting(coins, amount)
print(f"2. MEMOIZATION:")
print(f"   Answer: {result_memo}")
print(f"   Function calls: {call_count_memo}")
print(f"   Improvement: {call_count_bf // call_count_memo}x fewer calls!")
print()

# Method 3: Tabulation
iteration_count = 0
def coin_change_tab_counting(coins, amount):
    global iteration_count
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            iteration_count += 1
            if coin <= i:
                dp[i] = min(dp[i], 1 + dp[i - coin])
    return dp[amount]

result_tab = coin_change_tab_counting(coins, amount)
print(f"3. TABULATION:")
print(f"   Answer: {result_tab}")
print(f"   Iterations: {iteration_count}")
print(f"   No recursion overhead!")
print()


# =============================================================================
# PART 6: DECISION TREE - WHEN TO USE WHAT?
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     DECISION TREE: WHICH APPROACH?                        ║
╚══════════════════════════════════════════════════════════════════════════╝

START: Do you need to solve a problem?
│
├─ Is input size very small (n < 15)?
│  └─ YES → Use BRUTE FORCE (simplicity > efficiency)
│
├─ NO → Does problem have overlapping subproblems?
│  │
│  ├─ NO → Use GREEDY or DIVIDE & CONQUER
│  │
│  └─ YES → Does problem have optimal substructure?
│     │
│     ├─ NO → Not a DP problem
│     │
│     └─ YES → It's a DP problem! Choose implementation:
│        │
│        ├─ Natural recursive formulation? → MEMOIZATION
│        │  - Easy to code from recursion
│        │  - Not all states may be needed
│        │  - Acceptable recursion depth
│        │
│        └─ Need maximum performance? → TABULATION
│           - Production code
│           - Want to optimize space
│           - Very large state space
│           - No stack overflow risk

SPECIFIC USE CASES:
━━━━━━━━━━━━━━━━━━━

BRUTE FORCE:
✓ Understanding problem first
✓ Verifying correctness
✓ Teaching/learning
✓ Small competitions with tiny inputs
✓ Prototyping before optimization

MEMOIZATION:
✓ Naturally recursive problems (Fibonacci, factorials)
✓ Tree/graph traversal with repeated states
✓ Problems where not all states visited
✓ When you already have recursive solution
✓ Complex state transitions easier to express recursively

TABULATION:
✓ Production systems
✓ Competitive programming (performance matters)
✓ All states need computation anyway
✓ Want space optimization (rolling arrays)
✓ Very deep recursion would cause stack overflow
✓ Sequential/grid-based problems (edit distance, LCS, knapsack)
"""


# =============================================================================
# PART 7: COMMON PATTERNS & TEMPLATES
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         COMMON DP PATTERNS                                ║
╚══════════════════════════════════════════════════════════════════════════╝

PATTERN 1: LINEAR DP (1D)
━━━━━━━━━━━━━━━━━━━━━━━━━
Examples: Fibonacci, Stair Climbing, House Robber
State: dp[i] = optimal solution for first i elements
Transition: dp[i] depends on dp[i-1], dp[i-2], etc.

PATTERN 2: KNAPSACK (2D)
━━━━━━━━━━━━━━━━━━━━━━━━
Examples: 0/1 Knapsack, Subset Sum, Partition Equal Subset
State: dp[i][c] = optimal solution for items 0..i with capacity c
Transition: dp[i][c] = max(skip item i, take item i)

PATTERN 3: STRING DP (2D)
━━━━━━━━━━━━━━━━━━━━━━━━
Examples: Edit Distance, LCS, Longest Palindromic Subsequence
State: dp[i][j] = solution for string A[0..i] and B[0..j]
Transition: Depends on whether A[i] == B[j]

PATTERN 4: GRID/PATH DP (2D)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Examples: Unique Paths, Minimum Path Sum, Dungeon Game
State: dp[i][j] = optimal solution to reach cell (i, j)
Transition: dp[i][j] depends on dp[i-1][j] and dp[i][j-1]

PATTERN 5: STATE MACHINE DP
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Examples: M3 Tile, Stock Buy/Sell with cooldown
State: dp[i][state] = solution at position i in given state
Transition: Based on state machine transitions

PATTERN 6: INTERVAL DP
━━━━━━━━━━━━━━━━━━━━━━
Examples: Matrix Chain Multiplication, Burst Balloons
State: dp[i][j] = optimal solution for interval [i, j]
Transition: Try all split points k in [i, j]
"""


# =============================================================================
# PART 8: KEY TAKEAWAYS
# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║                           KEY TAKEAWAYS                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

1. PROBLEM IDENTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━
   Ask yourself:
   ✓ Are there overlapping subproblems?
   ✓ Does optimal solution use optimal subsolutions?
   ✓ Can I define states clearly?
   → If YES to all three → Use Dynamic Programming!

2. OPTIMIZATION PROGRESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Brute Force → Identify overlapping subproblems
              → Add memoization (cache results)
              → Convert to tabulation (if needed)
              → Optimize space (rolling arrays)

3. COMPLEXITY IMPROVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Brute Force:   O(2^n) or O(n!) - EXPONENTIAL
   Memoization:   O(n*m) - POLYNOMIAL
   Tabulation:    O(n*m) - POLYNOMIAL
   Space Optimized: O(m) - LINEAR SPACE

4. WHEN EACH APPROACH IS BEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Small inputs (n<15):         BRUTE FORCE
   Learning/Understanding:       BRUTE FORCE → MEMOIZATION
   Recursive problems:           MEMOIZATION
   Production/Performance:       TABULATION
   Space-constrained:           TABULATION + space optimization

5. COMMON MISTAKES
━━━━━━━━━━━━━━━━━━
   ✗ Using brute force for large inputs
   ✗ Not checking cache in memoization
   ✗ Wrong table initialization in tabulation
   ✗ Incorrect fill order in tabulation
   ✗ Not optimizing space when possible

6. PRACTICE STRATEGY
━━━━━━━━━━━━━━━━━━━━
   1. Start with brute force (understand problem)
   2. Identify overlapping subproblems
   3. Add memoization (prove it works)
   4. Convert to tabulation (better performance)
   5. Optimize space (if needed)

7. REAL-WORLD APPLICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ Route optimization (GPS, logistics)
   ✓ Text processing (spell check, diff)
   ✓ Bioinformatics (DNA sequence alignment)
   ✓ Resource allocation (knapsack variants)
   ✓ Game AI (minimax with DP)
   ✓ Finance (stock trading strategies)
   ✓ Compiler optimization
"""

print("=" * 80)
print("SUMMARY: THREE OPTIMIZATION APPROACHES")
print("=" * 80)
print("""
1. BRUTE FORCE: Try everything (slow but simple)
   → Use for: Small inputs, learning, verification

2. MEMOIZATION: Cache recursive results (Top-Down DP)
   → Use for: Natural recursion, not all states needed

3. TABULATION: Build solution iteratively (Bottom-Up DP)
   → Use for: Production code, maximum performance

KEY INSIGHT: All three can solve the same problems!
             The difference is in efficiency and implementation style.
""")
print("=" * 80)
