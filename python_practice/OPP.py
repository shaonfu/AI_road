#%%
"""
把一组数据结构和处理它们的方法组成对象（object）,
把相同行为的对象归纳为类（class）,
通过类的封装（encapsulation）隐藏内部细节，
通过继承（inheritance）实现类的特化（specialization）和泛化（generalization），
通过多态（polymorphism）实现基于对象类型的动态分派。"
"""

# %%
class Student(object):
    # __init__是一个特殊方法用于在创建对象时进行初始化操作
    # 通过这个方法我们可以为学生对象绑定name和age两个属性
    def __init__(self,name,age):
        self.name = name
        self.age = age
    
    def study(self,course_name):
        print('%s正在学习%s' % (self.name,course_name))

    #PEP 8要求标识符的名字用全小写多个单词用下划线连接
    #但是部分程序员和公司更倾向于使用驼峰命名法

    def watch_movie(self):
        if self.age < 18:
            print('%s只能观看熊出没'%(self.name))
        else:
            print('%s正在观看岛国爱情大电影。'%(self.name))

#说明:
#写在类中的函数,我们通常称之为(对象的)方法,这些方法就是对象可以接收的消息
# %%
def main():
    #创建学生对象并指定姓名和年龄
    stu1 = Student('楚旸',23)
    #给对象study消息
    stu1.study('Python程序设计')
    #给对象watch_av消息
    stu1.watch_movie()
    stu2 = Student('柔柔',17)
    stu2.study('思想品德')
    stu2.watch_movie()
# %%
if __name__ == '__main__':
    main()

# %%
#在Python中,属性和方法的访问权限只有公开和私有的
#如果希望属性是私有的,在给属性命名时可以用两个下划线左右开头
class Test:

    def __init__(self, foo):
        self.__foo = foo
    
    def __bar(self):
        print(self.__foo)
        print('__bar')
    
def main():
    test = Test('hello')
    # AttributeError: 'Test' object has no attribute '__bar'
    test.__bar()
    # AttributeError: 'Test' object has no attribute '__foo'
    print(test.__foo)

if __name__ == '__main__':
    main()


# %%
#Python并没有从语法上严格保证私有属性或方法的私密性，
#它只是给私有的属性和方法换了一个名字来妨碍对它们的访问
def main():
    test = Test('hello')
    test._Test__bar()
    print(test._Test__foo)


if __name__ == "__main__":
    main()

# %%
from time import sleep

class Clock(object):
    """数字时钟"""

    def __init__(self,hour=0,minute=0,second=0):
        """
        
        param hour: 时 
        param minute: 分 
        param second: 秒 
        """
        self._hour = hour
        self._minute = minute
        self._second = second

    def run(self):
        """走字"""
        self._second +=1
        if self._second == 60:
            self._second = 0
            self._minute += 1
            if self._minute == 60:
                self._minute = 0
                self._hour += 1
                if self._hour == 24:
                    self._hour = 0

    def show(self):
        """显示时间"""
        return '%02d:%02d:%02d' % \
            (self._hour,self._minute,self._second)
    
def main():
    clock = Clock(23,59,58)
    while True:
        print(clock.show())
        sleep(1)
        clock.run()
    
if __name__ == "__main__":
    main()


# %%
"""
我们之前的建议是将属性命名以单下划线开头，通过这种方式来暗示属性是受保护的，
不建议外界直接访问，那么如果想访问属性可以通过属性的getter（访问器）
和setter（修改器）方法进行对应的操作。
如果要做到这点，就可以考虑使用@property装饰器器来包装getter和setter方法
使得对属性的访问既安全又方便
"""
class Person(object):

    def __init__(self,name,age):
        self._name = name
        self._age = age
    
    #访问器 - getter方法
    @property
    def name(self):
        return self._name
    
    #访问器 - getter方法
    @property
    def age(self):
        return self._age
    
    #修改器 - setter方法
    @age.setter
    def age(self,age):
        self._age = age
    
    def play(self):
        if self._age <=16:
            print('%s正在玩飞行棋.'%self._name)
        else:
            print('%s正在玩斗地主.'%self._name)
    
def main():
    person = Person('王大锤',12)
    person.play()
    person.age = 22
    person.play()

if __name__ == '__main__':
    main()


#%%
#如果我们需要限定自定义类型的对象只能绑定某些属性，
#可以通过在类中定义__slots__变量来进行限定。
#需要注意的是__slots__的限定只对当前类的对象生效,
#对子类并不起任何作用。
#__slots__魔法

class Person(object):
    