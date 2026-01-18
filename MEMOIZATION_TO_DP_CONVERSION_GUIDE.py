"""
===============================================================================
🔁 HOW TO CONVERT MEMOIZATION → DYNAMIC PROGRAMMING (6-STEP METHOD)
===============================================================================
A mechanical process you can always follow
===============================================================================

✅ THE GENERAL 6-STEP METHOD (MEMORIZE THIS!)

STEP 1️⃣  Identify the STATE
         → Look at recursive function parameters: f(i), f(i,j), etc.

STEP 2️⃣  Identify the DP MEANING
         → Ask: "What does this function return?"

STEP 3️⃣  Identify BASE CASES
         → Look at recursion stopping conditions

STEP 4️⃣  Identify the TRANSITION
         → Look at recursive calls

STEP 5️⃣  Decide the COMPUTATION ORDER
         → Ask: "What must be computed before dp[i]?"

STEP 6️⃣  Extract the FINAL ANSWER
         → Look at what the original call was

🧠 KEY INSIGHT:
   Memoization = "What do I need?" (ask, then compute)
   DP = "What should I compute first?" (compute, then use)
   
   Same logic, opposite direction!

===============================================================================
"""

# =============================================================================
# 🔁 FULL EXAMPLE: CLIMBING STAIRS WITH COSTS
# =============================================================================
"""
Problem: You can climb 1 or 2 steps at a time. Each step has a cost.
         What's the minimum cost to reach the top?

Example: cost = [10, 15, 20]
         Answer: 15 (pay 15 at index 1, climb 2 steps to reach top)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧠 MEMOIZATION VERSION (Top-Down)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cost = [10, 15, 20]  # Example input
memo = {}

def f(i):
    """Memoization: minimum cost to reach step i"""
    # Base cases
    if i == 0:
        return cost[0]
    if i == 1:
        return cost[1]
    
    # Check memo
    if i in memo:
        return memo[i]
    
    # Recursive calls
    memo[i] = cost[i] + min(f(i-1), f(i-2))
    return memo[i]

# Original call
result_memo = min(f(len(cost)-1), f(len(cost)-2))
print(f"Memoization result: {result_memo}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔄 CONVERT STEP-BY-STEP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
1️⃣ State
   → i

2️⃣ DP meaning
   → f(i) = minimum cost to reach step i
   → So: dp[i] = minimum cost to reach step i

3️⃣ Base cases
   → if i == 0: return cost[0]
   → if i == 1: return cost[1]
   
   Becomes:
   → dp[0] = cost[0]
   → dp[1] = cost[1]

4️⃣ Transition
   → cost[i] + min(f(i-1), f(i-2))
   
   Becomes:
   → dp[i] = cost[i] + min(dp[i-1], dp[i-2])

5️⃣ Loop order
   → dp[i] depends on dp[i-1], dp[i-2]
   → Must compute smaller indices first
   
   Becomes:
   → for i in range(2, N):

6️⃣ Final answer
   → Original call: min(f(N-1), f(N-2))
   
   Becomes:
   → answer = min(dp[N-1], dp[N-2])
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ FINAL DP CODE (Bottom-Up)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def min_cost_climbing_stairs_dp(cost):
    """Dynamic Programming: minimum cost to reach top"""
    N = len(cost)
    
    # Create DP table
    dp = [0] * N
    
    # Initialize base cases (Step 3)
    dp[0] = cost[0]
    dp[1] = cost[1]
    
    # Fill table in order (Step 5)
    for i in range(2, N):
        dp[i] = cost[i] + min(dp[i-1], dp[i-2])  # Step 4: Transition
    
    # Extract final answer (Step 6)
    answer = min(dp[N-1], dp[N-2])
    return answer

# Test
result_dp = min_cost_climbing_stairs_dp(cost)
print(f"DP result: {result_dp}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Memoization (Top-Down):  {result_memo}")
print(f"DP (Bottom-Up):          {result_dp}")
print("\nBoth give same answer! ✓")
print("=" * 60)


# =============================================================================
# ⚠️ WHEN CONVERSION IS EASY vs HARD
# =============================================================================
"""
✅ EASY to convert (Use DP Tabulation):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• State is small (index, capacity, position)
• Clear dependency order
• All states needed anyway

Examples: Fibonacci, Stair Climbing, Coin Change, Knapsack


❌ HARD to convert (Keep Memoization):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• State contains large values (like product of numbers)
• No clear computation order
• Huge/sparse state space (not all states visited)

Example: Assignment 1 (product_v, sum_d) → memoization is better
"""


# =============================================================================
# 📝 MENTAL CHECKLIST (Use Every Time!)
# =============================================================================
"""
1. Function parameters    → state
2. Return value          → dp meaning
3. Base case             → dp initialization
4. Recursive calls       → dp transition
5. Call order            → loop order
6. Initial call          → final answer

🧩 One-Line Exam Answer:
━━━━━━━━━━━━━━━━━━━━━━━━━
To convert memoization to dynamic programming, identify the state and 
transition from the recursive function, initialize base cases, and compute 
states iteratively in dependency order.
"""


# =============================================================================
# 🎯 PRACTICE TEMPLATE
# =============================================================================
"""
Use this template to convert ANY memoization to DP:

# STEP 1: Identify state
# State: _______

# STEP 2: DP meaning
# dp[...] = _______

# STEP 3: Base cases
# dp[...] = _______

# STEP 4: Transition
# dp[...] = _______

# STEP 5: Loop order
# for ... in range(...):

# STEP 6: Final answer
# return _______
"""


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("✅ CONVERSION SUCCESSFUL!")
    print("=" * 60)
    print("\nNow you have a mechanical process to convert")
    print("ANY memoization code to dynamic programming!")
    print("\nJust follow the 6 steps every time! 🎯")
    print("=" * 60)
