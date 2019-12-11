#%%
"""
从列表中找出最大的或最小的N个元素
堆结构(大根堆/小根堆)
"""
import heapq

list1 = [34, 25, 12, 99, 87, 63, 58, 78, 88, 92]
list2 = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]

print(heapq.nlargest(3,list1))
print(heapq.nsmallest(3,list1))
print(heapq.nlargest(2,list2,key=lambda n: n['price']))
print(heapq.nlargest(2,list2,key=lambda n: n['shares']))

# %%
"""
找出序列中出现次数最多的元素
"""
from collections import Counter

words = [
    'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
    'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around',
    'the', 'eyes', "don't", 'look', 'around', 'the', 'eyes',
    'look', 'into', 'my', 'eyes', "you're", 'under'
]

counter = Counter(words)
print(counter.most_common(3))

# %%
#常用算法
#穷举法
"""
A、B、C、D、E五人在某天夜里合伙捕鱼 最后疲惫不堪各自睡觉
第二天A第一个醒来 他将鱼分为5份 扔掉多余的1条 拿走自己的一份
B第二个醒来 也将鱼分为5份 扔掉多余的1条 拿走自己的一份
然后C、D、E依次醒来也按同样的方式分鱼 问他们至少捕了多少条鱼
"""

fish = 6
while True:
    total = fish
    enough = True
    for _ in range(5):
        if (total - 1) % 5 == 0:
            total = (total - 1) // 5 * 4
        else:
            enough = False
            break
    if enough:
        print(fish)
        break
    fish += 5

# %%
#贪心法
"""
贪婪法：在对问题求解时，总是做出在当前看来是最好的选择，不追求最优解，快速找到满意解。
输入：
20 6
电脑 200 20
收音机 20 4
钟 175 10
花瓶 50 2
书 10 1
油画 90 9
"""
class Thing(object):
    """物品"""

    def __init__(self, name, price, weight):
        self.name = name
        self.price = price
        self.weight = weight

    @property
    def value(self):
        """价格重量比"""
        return self.price / self.weight


def input_thing():
    """输入物品信息"""
    name_str, price_str, weight_str = input().split()
    return name_str, int(price_str), int(weight_str)


def main():
    """主函数"""
    max_weight, num_of_things = map(int, input().split())
    all_things = []
    for _ in range(num_of_things):
        all_things.append(Thing(*input_thing()))
    all_things.sort(key=lambda x: x.value, reverse=True)
    total_weight = 0
    total_price = 0
    for thing in all_things:
        if total_weight + thing.weight <= max_weight:
            print(f'小偷拿走了{thing.name}')
            total_weight += thing.weight
            total_price += thing.price
    print(f'总价值: {total_price}美元')


if __name__ == '__main__':
    main()

# %%
#分治法
#快速排序 - 快速排序 - 选择枢轴对元素进行划分，左边都比枢轴小右边都比枢轴大
def quick_sort(origin_items,comp=lambda x , y: x<=y):
    items = origin_items[:]     #对半切片
    _quick_sort(items,0,len(items)-1,comp)
    return items

def _quick_sort(items, start, end, comp):
    if start < end:
        pos = _partition(items, start, end, comp)
        _quick_sort(items, start, pos - 1, comp)
        _quick_sort(items, pos + 1, end, comp)

def _partition(items, start, end, comp):
    pivot = items[end]
    i = start - 1
    for j in range(start, end):
        if comp(items[j], pivot):
            i += 1
            items[i], items[j] = items[j], items[i]
    items[i + 1], items[end] = items[end], items[i + 1]
    return i + 1


i = quick_sort([4,6,125,2,96,5,63])
print(i)


# %%
#回溯法


#%%
#动态规划1
#斐波那契数列

"""
动态规划 - 适用于有重叠子问题和最优子结构性质的问题
使用动态规划方法所耗时间往往远少于朴素解法(用空间换取时间)
"""

def fib(num,temp={}):
    """用递归计算Fibonacci数"""
    if num in (1,2):
        return 1
    try :
        return temp[num]
    except KeyError:
        temp[num] = fib(num-1) + fib(num-2)
        return temp[num]


#%%
#动态规划2
"""
子列表元素之和的最大值。（使用动态规划可以避免二重循环）
子列表指的是列表中索引（下标）连续的元素构成的列表；
列表中的元素是int类型，可能包含正整数、0、负整数；
程序输入列表中的元素，输出子列表元素求和的最大值
"""
def sum_list():
    items =list(map(int,input().split()))
    size = len(items)
    overall , partial = {},{}
    overall[size-1] = partial[size - 1] = items[size-1]
    for i in range(size -2 , -1 ,-1):
        partial[i] = max(items[i],partial[i + i] + items[i])
        overall[i] = max(partial[i],overall[i + i])
    print(overall[0])

sum_list()

# %%
#输出函数执行时间的装饰器
from time import time

def record_time(func):
    """自定义装饰函数的装饰器"""

    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time()
        result = func(*args, ** kwargs)
        print(f'{func.__name__}:{time()-start}秒')
        return result

    return wrapper


#%%
#用装饰器来实现单例模式
from functools import wraps
from threading import Lock

def singleton(cls):
    """线程安全的单例装饰器"""
    instances = {}
    locker = Lock()

    @wraps(cls)
    def wrapper(*args,**kwargs):
        if cls not in instances:
            with locker:
                if cls not in instances:
                    instances[cls] = cls(*args, ** kwargs)
        return instances[cls]   
    
    return wrapper
