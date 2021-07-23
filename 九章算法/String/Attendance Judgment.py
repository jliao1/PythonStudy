'''
Description
Given A string representing a student's attendance, 'A' stands for attendance, 'D' for default, and 'L' for lateness. If the student is default for two and more times or he is late for three and more consecutive times, he should be punished. Please judge whether the student should be punished or not and return a Boolean type.
'''

class Solution:
    """
    @param record: Attendance record.
    @return: If the student should be punished return true, else return false.
    """
    # 我自己做的
    def judge(self, record):
        # Write your code here.
        if record == '':
            return False

        #  max_D:0  max_L:1
        max_D = 0
        max_L = 0
        D = 0
        L = 0
        if record[0] == 'D':
            D += 1
        if record[0] == 'L':
            L += 1

        for i in range(1, len(record)):
            if record[i] == 'D':
                D += 1
                max_D = max(max_D,D)


            if record[i] == 'L':
                if record[i] == record[i - 1]:
                    L += 1
                    max_L = max(max_L,L)
                else:
                    L = 1

        return max_D >= 2 or max_L >= 3

if __name__ == '__main__':
    sol = Solution()
    res = sol.judge("LLL")
    while 0:
        print('我唉你')
