# Implement a class Class with the following attributes and methods:
#
# A public attribute students which is a array of Student instances.
# A constructor with a parameter n, which is the total number of students in this class. The constructor should create n Student instances and initialized with student id from 0 ~ n-1
# 样例
# Example 1:
#
# Input: 3
# Output: [0, 1, 2]
# Explanation: For 3 students, your cls.students[0] should equal to 0, cls.students[1] should equal to 1 and the cls.students[2] should equal to 2.


# Solution 1
class Student:
    def __init__(self, id):
        self.id = id;

class Class:

    '''
     * Declare a constructor with a parameter n which is the total number of
     * students in the *class*. The constructor should create n Student
     * instances and initialized with student id from 0 ~ n-1
    '''
    # write your code here
    def __init__(self, n):
        self.students = []
        for i in range(n):
            self.students.append(Student(i))

# Solution 2
class Student:
    def __init__(self, id):
        self.id = id;

class Class:

    '''
     * Declare a constructor with a parameter n which is the total number of
     * students in the *class*. The constructor should create n Student
     * instances and initialized with student id from 0 ~ n-1
    '''
    # write your code here
    def __init__(self, n):
        self.students = [Student(i) for i in range(n)]