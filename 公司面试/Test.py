def PerfectNumbers(N):
    s = set()
    for a in range(2, 30):
        for b in range(2, 30):
            flag = True
            for i in range(1, 1001):
                if flag == False:
                    break
                for j in range(i, 1001):
                    num = i ** b + j ** a
                    if num <= N:
                        s.add(num)
                    num = i ** a + j ** b
                    if num > N:
                        flag = False
                        break
                    s.add(num)
    ans = 0
    for i in range(1, N + 1):
        if i in s:
            ans += 1
    return ans


# Driver program
print(PerfectNumbers(2))
