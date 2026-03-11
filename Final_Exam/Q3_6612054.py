# Name - Win Yu Maung
# ID - 6612054  
# Sec - 541

first = list(input().split())
second = list(input().split())

text = ""

def longestCommonSubsequence(text1, text2):
    """2D DP for LCS."""
    global text
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                text += text1[i-1] + " "
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

print(longestCommonSubsequence(first, second))
print(text)

#I used Dp because it is to find longest common subsequences
#which can be divided into sub problems
