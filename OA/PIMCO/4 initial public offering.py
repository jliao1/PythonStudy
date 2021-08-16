#!/bin/python3

import math
import os
import random
import re
import sys

# 题目：https://www.chegg.com/homework-help/questions-and-answers/1-initial-public-offering-company-registers-ipo-website-sellsharescom-shares-website-avail-q53504295
# 答案1：15个test case过了14个：https://leetcode.com/discuss/interview-question/750495/citadel-and-citadel-securities-campus-software-engineering-challenge-2020-2021
# 我自己写的有点复杂，但是test case全过了

# Complete the 'getUnallottedUsers' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. 2D_INTEGER_ARRAY bids
#  2. INTEGER totalShares
#

# 我自己写的
def getUnallottedUsers1(bids, totalShares):
    # Write your code here
    price_list = []
    id_share_dic ={}
    id_time = {}
    price_dic = {}
    for bid in bids:
        id_share_dic[bid[0]] = bid[1]
        id_time[bid[0]] = bid[3]
        if bid[2] not in price_list:
            price_list.append(bid[2])
        if bid[2] not in price_dic:
            price_dic[bid[2]] = [bid[0]]
        else:
            price_dic[bid[2]].append(bid[0])
    price_list.sort(reverse=True)

    for each in price_list:
        if totalShares <= 0:
            break

        ids = price_dic[each]
        if len(ids) > 1:
            totalShares = interative(ids, id_share_dic, totalShares, id_time)
        else:
            totalShares = totalShares - id_share_dic[ids[0]]
            id_share_dic[ids[0]] = 0

    id_share_dic2 = {}
    for bid in bids:
        id_share_dic2[bid[0]] = bid[1]

    res = []
    for key in id_share_dic.keys():
        a =  id_share_dic[key]
        b = id_share_dic2[key]
        if a == b:
            res.append(key)

    return res

def interative(ids, id_share_dic, totalShares, id_time):
    id_t = []
    for each in ids:
        t = id_time[each]
        id_t.append((each, t))
    id_t.sort(key=lambda x: x[1])
    ids = [i[0] for i in id_t ]

    while totalShares > 0 and returnSum(id_share_dic,ids) > 0:
        for i in range(len(ids)):
            name = ids[i]
            if id_share_dic[name] > 0:
                id_share_dic[name] -= 1
                totalShares -= 1
                if totalShares <= 0:
                    break
    return  totalShares


def returnSum(myDict,dis):
    sum = 0
    for i in dis:
        sum = sum + myDict[i]

    return sum



if __name__ == '__main__':
    case1_bids = [[3, 5, 4, 6], [1, 2, 5, 0], [2, 1, 4, 2]]  # total = 3   答案是 3
    case2_bids = [[1, 3, 1, 9866], [2, 1, 2, 5258], [3, 2, 4, 5788], [4, 2, 4, 6536]] # total = 2
    case3_bids = [[2, 7, 8, 1], [3, 7, 5, 1], [4, 10, 3, 3], [1, 5, 5, 0]]  # total = 18
    total = 2
    l = getUnallottedUsers1(case2_bids, 2)
    print(l)
    pass