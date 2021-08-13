'''
Description
Given an array of strings, return all groups of strings that are anagrams.If a string is Anagram,there must be another string with the same letter set but different order in S.

'''

import collections

'''
【知识点】

sorted() 可以用于一切iterable的对象
时间复杂度是 NlogN，N是sort对象的长度
返回的是一个已经sort好的 list 

join() for strings is O(n) where n is the length of the string to be concatenated
所以比 + 好 The time complexity of string concatenation using + is O(n²)

'''

class Solution:
    """
    @param strs: A list of strings
    @return: A list of strings
    """
    # 应该是 时间 O(NK·logK)
    def anagrams1(self, strs):
        # write your code here
        dict = {}
        # N^2·logN
        for word in strs:
            sortedword = ''.join(sorted(word)) # KlogK
            dict[sortedword] = [word] if sortedword not in dict else dict[sortedword] + [word]
        res = []
        for key in dict:
            if len(dict[key]) >= 2:
                res.extend( dict[key] )
        return res

    # 写法比1更简短
    # 时间 O(NK·logK)
    # 空间 O(NK)
    def anagrams2(self, strs):
        # write your code here
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()


    '''
    Time Complexity: O(NK), where N is the length of strs, and K is the maximum length of a string in strs. Counting each string is linear in the size of the string, and we count every string.
    Space Complexity: O(NK), the total information content stored in ans.
    '''
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()

    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t)

if __name__ == '__main__':
    sol = Solution()
    res = sol.anagrams(["lint", "intl", "inlt", "code"])
    print(res)