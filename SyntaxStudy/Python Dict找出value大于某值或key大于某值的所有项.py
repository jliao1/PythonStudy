

test_dict = {1:5, 2:4, 3:3, 4:2, 5:1}

for v in test_dict.values():
    if v >=3:
        print('找到了')

print({k:v for k, v in test_dict.items() if k>=3})
print({k:v for k, v in test_dict.items() if v>=3})
print([k for k, v in test_dict.items() if k>=3])
print([k for k, v in test_dict.items() if v>=3])
print([v for k, v in test_dict.items() if k>=3])
print([v for k, v in test_dict.items() if v>=3])


# 想要求value值大于等于3的所有项：
print({k:v for k, v in test_dict.items() if v>=3})
{1: 5, 2: 4, 3: 3}

# 想要求key值大于等于3的所有项：
print({k:v for k, v in test_dict.items() if k>=3})


def keyInList(k, l):
        return bool([True for i in l if k in i.values()])