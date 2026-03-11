"""
10 LeetCode Divide and Conquer Problems with:
- Short problem description
- Example input
- Example output
- Python answer code

Note:
These are short summaries for study use, not full official problem statements.
"""

from typing import List, Optional


# --------------------------------------------------
# 1. Convert Sorted Array to Binary Search Tree
# LeetCode 108
# --------------------------------------------------
"""
Description:
Given a sorted array, build a height-balanced binary search tree.

Example Input:
nums = [-10, -3, 0, 5, 9]

Example Output:
A height-balanced BST, for example:
[0, -10, 5, null, -3, null, 9]
"""

class TreeNode:
    def __init__(self, val=0, left: Optional["TreeNode"] = None, right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right


class Solution108:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(left: int, right: int) -> Optional[TreeNode]:
            if left > right:
                return None

            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = build(left, mid - 1)
            root.right = build(mid + 1, right)
            return root

        return build(0, len(nums) - 1)


# --------------------------------------------------
# 2. Merge k Sorted Lists
# LeetCode 23
# --------------------------------------------------
"""
Description:
Merge k sorted linked lists into one sorted linked list.

Example Input:
lists = [[1,4,5], [1,3,4], [2,6]]

Example Output:
[1,1,2,3,4,4,5,6]
"""

class ListNode:
    def __init__(self, val=0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


class Solution23:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None

        def merge_two(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode()
            tail = dummy

            while l1 and l2:
                if l1.val < l2.val:
                    tail.next = l1
                    l1 = l1.next
                else:
                    tail.next = l2
                    l2 = l2.next
                tail = tail.next

            tail.next = l1 or l2
            return dummy.next

        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged.append(merge_two(l1, l2))
            lists = merged

        return lists[0]


# --------------------------------------------------
# 3. Sort an Array
# LeetCode 912
# --------------------------------------------------
"""
Description:
Sort an array of integers in ascending order.

Example Input:
nums = [5,2,3,1]

Example Output:
[1,2,3,5]
"""

class Solution912:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge_sort(arr: List[int]) -> List[int]:
            if len(arr) <= 1:
                return arr

            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])

            result = []
            i = j = 0

            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1

            result.extend(left[i:])
            result.extend(right[j:])
            return result

        return merge_sort(nums)


# --------------------------------------------------
# 4. Maximum Subarray
# LeetCode 53
# --------------------------------------------------
"""
Description:
Find the contiguous subarray with the largest sum.

Example Input:
nums = [-2,1,-3,4,-1,2,1,-5,4]

Example Output:
6
Explanation:
The subarray [4,-1,2,1] has the largest sum = 6.
"""

class Solution53:
    def maxSubArray(self, nums: List[int]) -> int:
        def solve(left: int, right: int) -> int:
            if left == right:
                return nums[left]

            mid = (left + right) // 2
            left_best = solve(left, mid)
            right_best = solve(mid + 1, right)

            left_sum = float("-inf")
            curr = 0
            for i in range(mid, left - 1, -1):
                curr += nums[i]
                left_sum = max(left_sum, curr)

            right_sum = float("-inf")
            curr = 0
            for i in range(mid + 1, right + 1):
                curr += nums[i]
                right_sum = max(right_sum, curr)

            cross = left_sum + right_sum
            return max(left_best, right_best, cross)

        return solve(0, len(nums) - 1)


# --------------------------------------------------
# 5. Search a 2D Matrix II
# LeetCode 240
# --------------------------------------------------
"""
Description:
Given a matrix where each row and each column is sorted, determine whether a target exists.

Example Input:
matrix = [
  [1,4,7,11,15],
  [2,5,8,12,19],
  [3,6,9,16,22],
  [10,13,14,17,24],
  [18,21,23,26,30]
]
target = 5

Example Output:
True
"""

class Solution240:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False

        rows, cols = len(matrix), len(matrix[0])

        def search(r1: int, r2: int, c1: int, c2: int) -> bool:
            if r1 > r2 or c1 > c2:
                return False
            if target < matrix[r1][c1] or target > matrix[r2][c2]:
                return False

            mid_col = (c1 + c2) // 2
            row = r1

            while row <= r2 and matrix[row][mid_col] <= target:
                if matrix[row][mid_col] == target:
                    return True
                row += 1

            return search(row, r2, c1, mid_col - 1) or search(r1, row - 1, mid_col + 1, c2)

        return search(0, rows - 1, 0, cols - 1)


# --------------------------------------------------
# 6. Different Ways to Add Parentheses
# LeetCode 241
# --------------------------------------------------
"""
Description:
Given an expression string, return all possible results from computing
all different ways to group numbers and operators.

Example Input:
expression = "2-1-1"

Example Output:
[0, 2]
"""

class Solution241:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        memo = {}

        def solve(expr: str) -> List[int]:
            if expr in memo:
                return memo[expr]

            result = []
            for i, ch in enumerate(expr):
                if ch in "+-*":
                    left = solve(expr[:i])
                    right = solve(expr[i + 1:])

                    for a in left:
                        for b in right:
                            if ch == "+":
                                result.append(a + b)
                            elif ch == "-":
                                result.append(a - b)
                            else:
                                result.append(a * b)

            if not result:
                result.append(int(expr))

            memo[expr] = result
            return result

        return solve(expression)


# --------------------------------------------------
# 7. Kth Largest Element in an Array
# LeetCode 215
# --------------------------------------------------
"""
Description:
Find the k-th largest element in an unsorted array.

Example Input:
nums = [3,2,1,5,6,4]
k = 2

Example Output:
5
"""

class Solution215:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target = len(nums) - k

        def quickselect(left: int, right: int) -> int:
            pivot = nums[right]
            p = left

            for i in range(left, right):
                if nums[i] <= pivot:
                    nums[i], nums[p] = nums[p], nums[i]
                    p += 1

            nums[p], nums[right] = nums[right], nums[p]

            if p == target:
                return nums[p]
            if p < target:
                return quickselect(p + 1, right)
            return quickselect(left, p - 1)

        return quickselect(0, len(nums) - 1)


# --------------------------------------------------
# 8. Construct Quad Tree
# LeetCode 427
# --------------------------------------------------
"""
Description:
Construct a quad tree from a binary grid.

Example Input:
grid = [
[1,1],
[1,1]
]

Example Output:
A quad tree node where isLeaf = True and val = True
"""

class QuadNode:
    def __init__(self, val: bool, isLeaf: bool, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution427:
    def construct(self, grid: List[List[int]]) -> 'QuadNode':
        def build(r: int, c: int, size: int) -> QuadNode:
            first = grid[r][c]
            same = True

            for i in range(r, r + size):
                for j in range(c, c + size):
                    if grid[i][j] != first:
                        same = False
                        break
                if not same:
                    break

            if same:
                return QuadNode(bool(first), True)

            half = size // 2
            return QuadNode(
                True,
                False,
                build(r, c, half),
                build(r, c + half, half),
                build(r + half, c, half),
                build(r + half, c + half, half),
            )

        return build(0, 0, len(grid))


# --------------------------------------------------
# 9. Beautiful Array
# LeetCode 932
# --------------------------------------------------
"""
Description:
Build an array of numbers from 1 to n such that for any i < k < j,
nums[k] * 2 != nums[i] + nums[j].

Example Input:
n = 4

Example Output:
[2,1,4,3]
Another valid answer is also acceptable.
"""

class Solution932:
    def beautifulArray(self, n: int) -> List[int]:
        memo = {1: [1]}

        def build(size: int) -> List[int]:
            if size in memo:
                return memo[size]

            odds = build((size + 1) // 2)
            evens = build(size // 2)

            result = [2 * x - 1 for x in odds] + [2 * x for x in evens]
            memo[size] = result
            return result

        return build(n)


# --------------------------------------------------
# 10. Reverse Pairs
# LeetCode 493
# --------------------------------------------------
"""
Description:
Count important reverse pairs in an array where i < j and nums[i] > 2 * nums[j].

Example Input:
nums = [1,3,2,3,1]

Example Output:
2
"""

class Solution493:
    def reversePairs(self, nums: List[int]) -> int:
        def merge_sort(arr: List[int]) -> int:
            nonlocal nums
            if len(arr) <= 1:
                return 0

            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]

            count = merge_sort(left) + merge_sort(right)

            j = 0
            for i in range(len(left)):
                while j < len(right) and left[i] > 2 * right[j]:
                    j += 1
                count += j

            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1

            return count

        return merge_sort(nums)


PROBLEM_LIST = [
    "LeetCode 108 - Convert Sorted Array to Binary Search Tree",
    "LeetCode 23 - Merge k Sorted Lists",
    "LeetCode 912 - Sort an Array",
    "LeetCode 53 - Maximum Subarray",
    "LeetCode 240 - Search a 2D Matrix II",
    "LeetCode 241 - Different Ways to Add Parentheses",
    "LeetCode 215 - Kth Largest Element in an Array",
    "LeetCode 427 - Construct Quad Tree",
    "LeetCode 932 - Beautiful Array",
    "LeetCode 493 - Reverse Pairs",
]


if __name__ == "__main__":
    print("10 LeetCode Divide and Conquer problems in this file:")
    for i, item in enumerate(PROBLEM_LIST, 1):
        print(f"{i}. {item}")


# ============================================================================
# PROBLEM INDEX - QUICK REFERENCE
# ============================================================================
"""
================================================================================
              LEETCODE DIVIDE & CONQUER PROBLEMS - CONTENT INDEX
================================================================================

DIVIDE AND CONQUER PROBLEMS
-------------------------------------------------
Line 15   | Problem 1:  Convert Sorted Array to Binary Search Tree (LeetCode 108)
Line 53   | Problem 2:  Merge k Sorted Lists (LeetCode 23)
Line 106  | Problem 3:  Sort an Array (LeetCode 912)
Line 149  | Problem 4:  Maximum Subarray (LeetCode 53)
Line 194  | Problem 5:  Search a 2D Matrix II (LeetCode 240)
Line 242  | Problem 6:  Different Ways to Add Parentheses (LeetCode 241)
Line 290  | Problem 7:  Kth Largest Element in an Array (LeetCode 215)
Line 330  | Problem 8:  Construct Quad Tree (LeetCode 427)
Line 388  | Problem 9:  Beautiful Array (LeetCode 932)
Line 423  | Problem 10: Reverse Pairs (LeetCode 493)

================================================================================
                    TOTAL: 10 DIVIDE & CONQUER PROBLEMS
================================================================================

PROBLEM CATEGORIES:
-------------------
TREE CONSTRUCTION:
  • Problem 1: Convert Sorted Array to BST - Recursive tree building
  • Problem 8: Construct Quad Tree - 2D space partitioning

SORTING & MERGING:
  • Problem 2: Merge k Sorted Lists - Multi-way merge
  • Problem 3: Sort an Array - Merge sort implementation
  • Problem 10: Reverse Pairs - Modified merge sort with counting

ARRAY OPTIMIZATION:
  • Problem 4: Maximum Subarray - Classic D&C approach
  • Problem 7: Kth Largest Element - Quickselect algorithm

SEARCH PROBLEMS:
  • Problem 5: Search a 2D Matrix II - 2D binary search variant

MATHEMATICAL/COMBINATORIAL:
  • Problem 6: Different Ways to Add Parentheses - Expression evaluation
  • Problem 9: Beautiful Array - Recursive construction

KEY DIVIDE & CONQUER PATTERNS:
-------------------------------
1. SPLIT: Divide problem into smaller subproblems
2. CONQUER: Solve subproblems recursively
3. COMBINE: Merge solutions together

COMMON RECURRENCE:
- T(n) = 2T(n/2) + O(n) → O(n log n) - Binary split with linear merge
- T(n) = T(n/2) + O(1) → O(log n) - Binary search
- T(n) = 8T(n/2) + O(n²) → O(n³) - Octree/Quadtree

TIME COMPLEXITIES:
------------------
• Binary Search Tree Construction: O(n)
• Merge k Lists: O(n log k)
• Merge Sort: O(n log n)
• Maximum Subarray (D&C): O(n log n)
• Search 2D Matrix: O(m + n)
• Quickselect: O(n) average, O(n²) worst
• Quad Tree: O(n²)

SPACE COMPLEXITIES:
-------------------
• Recursion depth: O(log n) for balanced divisions
• Merge operations: O(n) for temporary arrays
• Tree structures: O(n) for storing nodes

HOW TO USE THIS INDEX:
----------------------
1. Use Cmd/Ctrl+G to go to specific line number
2. Each problem includes description, examples, and solution
3. Compare with DP: D&C for independent subproblems, DP for overlapping

WHEN TO USE DIVIDE & CONQUER:
------------------------------
✓ Problem can be split into independent subproblems
✓ Subproblem structure is same as original problem
✓ Solutions can be combined efficiently
✓ Examples: Sorting, searching in sorted data, tree construction

NOT SUITABLE WHEN:
✗ Subproblems overlap (use DP instead)
✗ Can't divide problem naturally
✗ Combining solutions is too expensive

================================================================================
"""
