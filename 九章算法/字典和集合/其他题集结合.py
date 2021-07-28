class Solution:

    # Lintcode(力扣734) easy 856 · Sentence Similarity 是 hash map 和 set 的结合体 双向记录，单向查找
    def isSentenceSimilarity1(self, words1, words2, pairs):
        """
        思路：双向记录，单向查找（更省一些空间的看版本2：单向记录，双向查找）
        value要用list来装，为啥？因为有可能一个单词 => 对多个同义词的
        但真的要用list吗？不是的。有2个原因
        (1) 由于一对多，“多”有可能重复，但重复在这题是没必要的，所以用 set() 来存
        (2) 查找主要用set，非常快！ list会慢的
        不要一上去就无脑地想着它一定是一对一的
        """
        if not words1 and not words2:
            return True
        if not words1 or not words2 or len(words1) != len(words2):
            return False

        dic = {}  # 也可以写成 dict()

        # 构建map
        for each in pairs:
            first_word = each[0]
            second_word = each[1]

            # first_word => second_word
            # 如果 first_word 不存在 dic 的 keys 里
            # 那就把 first_word 存进 dic，且其对应 value 设置为一个空集合 set()，有返回值也是这个空集合
            dic.setdefault(first_word, set())  # 如果 first_word 存在 dic 里，返回的是它对应的 value
            dic[first_word].add(second_word)

            # 因为是 symetric 的，所以除了 first_word => second_word
            # 也要 second_word => first_word （双向记录）
            dic.setdefault(second_word, set())
            dic[second_word].add(first_word)

        for i in range(len(words1)):
            # 如果俩词儿 其实是同一个词儿，那就没有必要用 配对的map判断啦，直接continue就好啦
            if words1[i] == words2[i]:
                continue

            # 如果 key 压根儿就不在 map 中
            # 或者 不存在 words1[i] => words2[i], 返回 False
            if not dic[words1[i]] or words2[i] not in dic[words1[i]]:  # （单向查找）
                return False

        # 能走到最后就返回 True 就好
        return True

    # Lintcode(力扣734) easy 856 · Sentence Similarity 是 hash map 和 set 的结合体 单向记录，双向查找，就可以更省一半map空间了
    def isSentenceSimilarity2(self, words1, words2, pairs):
        """单向记录，双向查找，就可以更省一半map空间了"""
        if not words1 and not words2:
            return True
        if not words1 or not words2 or len(words1) != len(words2):
            return False

        dic = {}  # 也可以写成 dict()

        # 构建map
        for each in pairs:
            first_word = each[0]
            second_word = each[1]

            # 单向记录
            dic.setdefault(first_word, set())
            dic[first_word].add(second_word)

        for i in range(len(words1)):
            if words1[i] == words2[i]:
                continue

            # 双向查找
            if ( not dic[words1[i]] or words2[i] not in dic[words1[i]] ) \
            and ( not dic[words2[i]] or words1[i] not in dic[words2[i]] ):
                return False

        # 能走到最后就返回 True 就好
        return True

    # Lintcode(力扣737) Medium 855 · Sentence Similarity II
    def areSentencesSimilarTwo1_wrong_answer(self, words1, words2, pairs):
        if not words1 and not words2:
            return True
        if not words1 or not words2 or len(words1) != len(words2):
            return False

        # construct map
        dic = {}
        size = len(words1)
        for each in pairs:
            # get specific words
            w1 = each[0]
            w2 = each[1]

            # words1 => words2 就是words1映射到words2
            dic.setdefault(w1, set())
            dic[w1].add(w2)

            # words2 => words1 就是words2映射到words1
            dic.setdefault(w2, set())
            dic[w2].add(w1)
        # 这个会stack overflow，因为会产生无限调用（无限循环），因此要对每个访问过的点，作标记才行
        def find(w1, w2):
            key = w1
            s = dic.get(key, None)
            # 如果value存在，但value不是w2，那就继续找
            if s and w2 not in s:
                for each in s:
                    if find(each, w2):
                        return True

            # 如果value存在，而且value是w2，那就返回true
            elif s and w2 in s:
                return True
            # 剩下情况返回False
            else:
                return False

        # begin search
        for i in range(size):
            if words1[i] == words2[i]:
                continue

            is_find = find(words1[i], words2[i])
            if is_find:
                continue
            if not is_find:
                return False

        return True

    # Lintcode(力扣737) Medium 855 · Sentence Similarity II
    def areSentencesSimilarTwo2(self, words1, words2, pairs):
        """
        这题可以用union find可以做，但现在我不会，以后再看

        我这种时间复杂度感觉是O(np) n是words1/words2的长度，p是看一个word最多可能有几个同义词
        """
        if not words1 and not words2:
            return True
        if not words1 or not words2 or len(words1) != len(words2):
            return False

        # construct map
        dic = {}
        for each in pairs:
            # get specific words
            w1 = each[0]
            w2 = each[1]

            # words1 => words2 就是words1映射到words2
            dic.setdefault(w1, set())
            dic[w1].add(w2)

            # words2 => words1 就是words2映射到words1
            dic.setdefault(w2, set())
            dic[w2].add(w1)

        # 这个递归函数用s去重查找，就不会无限循环导致stack overflow了
        def find_all_sub_sets(w1):
            sub_s = dic.get(w1, None)
            if sub_s:
                for each in sub_s:
                    if each in Sets:
                        continue
                    else:
                        Sets.add(each)
                        find_all_sub_sets(each)
            else:
                return

        # begin search
        size = len(words1)
        for i in range(size):
            w1 = words1[i]
            w2 = words2[i]

            if w1 == w2:
                continue

            Sets = set()
            find_all_sub_sets(w1)

            if Sets and w2 in Sets:
                continue
            else:
                return False

        return True

if __name__ == '__main__':
    sol = Solution()
    l = sol.areSentencesSimilarTwo2(["great","acting","skills"],["fine","talent","talentdrama"],[["great","good"],["fine","good"],["drama","acting"],["skills","talent"]])
    print(l)