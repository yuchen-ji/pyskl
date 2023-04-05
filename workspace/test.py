s = input() # 输入字符串
n = len(s)
s = list(s)

# 检查当前字符串是否已经是回文串
def is_palindrome(s):
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            return False
    return True

# 将字符修改为字典序最小的'a' - 'z'
def change_char(s, i, j):
    s[i] = s[j]
    return
    for c in range(ord('a'), ord('z') + 1):
        if chr(c) != s[i] and chr(c) != s[j]:
            s[i] = s[j] = chr(c)
            return

sum = 0
for i in range(n // 2):
    j = n - 1 - i
    if s[i] != s[j]:
        if sum == 2:
            for ii in range(i, n//2, 1):
                if s[ii] != s[n-1-ii]:
                    print("False")
                    break
        s[i] = s[j]
        sum += 1

print(''.join(s))
