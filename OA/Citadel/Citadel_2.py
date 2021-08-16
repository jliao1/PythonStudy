#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'lightBulbs' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER_ARRAY states
#  2. INTEGER_ARRAY numbers
#
def prime_factors(number):
    prime_factors = set()
    while number % 2 == 0:
        prime_factors.add(2)
        number = number / 2
    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.add(i)
            number = number / i
    if number > 2:
        prime_factors.add(int(number))
    return prime_factors

def flip_bulb(states, index):
    init = index - 1
    while init < len(states):
        if states[init] == 1:
            states[init] = 0
        else:
            states[init] = 1
        init += index

def lightBulbs(states, numbers):
    # main
    total_set = set()
    for number in numbers:
        p_fs = prime_factors(number)
        for p_f in p_fs:
            # 这里 add 和 remove 的操作，就是去掉一些重复的没必要的翻转
            if p_f in total_set:
                total_set.remove(p_f)
            else:
                total_set.add(p_f)
    for prime_factor in total_set:
            flip_bulb(states, prime_factor)
    return states

if __name__ == '__main__':
    states = [0,0,0,0,0,0]
    numbers = [2,4,2,6]
    res = lightBulbs(states, numbers)
    print(res)