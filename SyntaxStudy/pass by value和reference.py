# if __name__ == '__main__':
#
#     # student1 = {'Jim': 12, 'Anna': 14, 'Preet': 10}
#     #
#     # def test(student):
#     #     new = {'Sam': 20, 'Steve': 21}
#     #     student.update(new)
#     #     print("Inside the function", student)
#     #     return
#     #
#     # test(student1)
#     # print("Outside the function:", student1)
#

def test(stu):
    new = {'Sam': 4, 'Steve': 5}
    stu.update(new)
    print("Inside:", stu)
    return

if __name__ == '__main__':
    stu = {'Jim': 1, 'Anna': 2, 'Preet': 3}
    test(dict(stu))
    print("Outside:", stu)

    '''
    python 有deepcopy 库, 用法是
    import copy
    new_stu = copy.deepcopy(stu)
    '''






