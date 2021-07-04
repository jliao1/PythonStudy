class interface(object):  # 假设这就是一个接口，接口名可以随意定义，所有的子类不需要实现在这个类中的函数
    def Lee(self):
        pass

    def Marlon(self):
        pass


class Realaize_interface(interface):
    def __init__(self):
        pass

    def Lee(self):
        print("1.1")

    def Marlon(self):
        print("1.2")

class Realaize_interface2(interface):
    def __init__(self):
        pass

    def Lee(self):
        print("2.1")

    def Marlon(self):
        print("2.2")


obj1 = Realaize_interface()
obj1.Lee()

obj2 = Realaize_interface2()
obj2.Lee()

a = [obj1,obj2]
for each in a:
    each.Lee()
