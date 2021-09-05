"""
input 就是一堆line 每一个line都有一个string或者空的， input大小写其实不用在意。
output就是把这些line合成一个string， 如果line是空的话就加空格。
空格后面的首个字母都是小写 之后每一个line只要前面不是空格首字母就要大写，
这个line剩下的字母都小写。 最后output的第一个字母永远小写
"""

# Self Descriptive Number
#

def Q1(arr):
    res = []
    is_last_empty = True
    for each_str in arr:
        if not each_str:
            # each_line 是条空的
            res.append(' ')
            is_last_empty = True
        else:
            # each_line 非空空的
            if is_last_empty:
                res.append(each_str.lower())
            else:
                res.append(each_str[0].upper() + each_str[1:].lower())
            is_last_empty = False
    return ''.join(res)

def Q2(s):

    import collections
    # time complexity O(n)
    # space complexity is O(1) since 都contains constant number of elements
    # count frequency of occurrence of each letter in input parameter
    count = collections.Counter(s)
    '''
    要用上面这种写法，用下面这种不行。如果用下面这种
    在取count["z"]时会出错
    count = {}
    for each in s:
        if each not in count:
            count[each] = 1
        else:
            count[each] += 1
    '''
    """
    that all even digits have at least one character not present in any other strings, while all odd digits don’t
    """
    # building hashmap digit -> its frequency

    '''
    That is actually the key how to count 3s, 5s and 7s since some letters are present only 
    in one odd and one even number (and all even numbers has already been counted) :
    Letter "h" is present only in "three" and "eight".
    Letter "f" is present only in "five" and "four".
    Letter "s" is present only in "seven" and "six".
    Now one needs to count 9s and 1s only, and the logic is basically the same :
    
    Letter "i" is present in "nine", "five", "six", and "eight".
    Letter "n" is present in "one", "seven", and "nine".
    '''
    num_frequency = {}
    # "zero" only has one digit "z"
    num_frequency["0"] = count["z"]
    # "two" only has one digit "w"
    num_frequency["2"] = count["w"]
    # "four" only has one digit "u"
    num_frequency["4"] = count["u"]
    # "six"  only has one digit "x"
    num_frequency["6"] = count["x"]
    # "eight" only has one digit "g"
    num_frequency["8"] = count["g"]
    # letter "h" is present only in "three" and "eight"
    num_frequency["3"] = count["h"] - num_frequency["8"]
    # letter "f" is present only in "five" and "four"
    num_frequency["5"] = count["f"] - num_frequency["4"]
    # letter "s" is present only in "seven" and "six"
    num_frequency["7"] = count["s"] - num_frequency["6"]
    # letter "i" is present in "nine", "five", "six", and "eight"
    num_frequency["9"] = count["i"] - num_frequency["5"] - num_frequency["6"] - num_frequency["8"]
    # letter "n" is present in "one", "nine", and "seven"
    num_frequency["1"] = count["n"] - num_frequency["7"] - 2 * num_frequency["9"]

    # building output string
    output = [key * num_frequency[key] for key in sorted(num_frequency.keys())]
    return "".join(output)

def Q3(line):
    string = str(line)
    is_self_des = True

    import collections
    count = collections.Counter(string)

    for i in range(len(string)):
        num = int(string[i])
        if count[str(i)] != num:
            is_self_des = False

    if is_self_des:
        print(1)
    else:
        print(0)



if __name__ == '__main__':
    arr = ['One','tWO','thrEe','','Six','five','', 'eight']
    res = Q1(arr)
    ans = Q3(2020)
    print(ans)