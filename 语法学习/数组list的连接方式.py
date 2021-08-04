if __name__ == '__main__':
    l1 = [1,2,3]
    l2 = [6,7]
    l3 = l1 + l2
    l4 = l1.append(l2[:])

    print(l1)  # 打印出来是 [1, 2, 3, [6, 7]]
    print(l2)  # 打印出来是 [6, 7]
    print(l3)  # [1, 2, 3, 6, 7]
    print(l4)  # 打印出来是 None

    l5 = [1,2,3,4]
    del l5[-1]
    print(l5)  # 打印出来是 [1, 2, 3]