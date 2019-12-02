#%%
def main():
    f = open('data/test.txt','r',encoding = 'utf-8')
    print(f.read())
    f.close()

if __name__ == '__main__':
    main()


# %%
#通过with关键字指定文件对象的上下文环境
#并在离开上下文环境时自动释放文件资源
def main():
    try:
        with open('python_practice/data/test.txt','r',encoding = 'utf-8') as f:
            print(f.read())
    except FileNotFoundError:
        print('无法打开指定的文件!')
    except LookupError:
        print('指定了未知的编码!')
    except UnicodeDecodeError:
        print('读取文件时解码错误')

if __name__ == "__main__":
    main()

# %%
import time

def read_for():
    with open('python_practice/data/test.txt','r',encoding = 'utf-8') as f:
            print(f.read())
    #通过for-in循环逐行读取
    with open('python_practice/data/test.txt',mode='r') as f:
            for line in f:
                print(line,end = '')
                time.sleep(0.5)
    print()

    #读取文件按行读取到列表中
    with open('python_practice/data/test.txt',mode='r') as f:
        lines = f.readlines()
    print(lines)

read_for()

# %%
#写入
"""
要将文本信息写入文件文件也非常简单，
在使用open函数时指定好文件名并将文件模式设置为'w'即可。
注意如果需要对文件内容进行追加式写入，应该将模式设置为'a'。
如果要写入的文件不存在会自动创建文件而不是引发异常。
"""
from math import sqrt

def is_prime(n):
    """判断素数的函数"""
    assert n>0
    for factor in range(2,int(sqrt(n))+1):
        if n % factor == 0:
            return False
    return True if n!= 1 else False

def write_txt():
    filenames = ('python_practice/data/write_1.txt','python_practice/data/write_2.txt','python_practice/data/write_3.txt')
    fs_list = []
    try:
        for filename in filenames:
            fs_list.append(open(filename,'w',encoding ='utf-8'))
        for number in range(1,10000):
            if is_prime(number):
                if number < 100:
                    fs_list[0].write(str(number) + '\n')
                elif number < 1000:
                    fs_list[1].write(str(number) + '\n')
                else:
                    fs_list[2].write(str(number) + '\n')
    except IOError as ex:
        print(ex)
        print('写文件时发生错误!')
    finally:
        for fs in fs_list:
            fs.close()
    print('操作完成!')

write_txt()



# %%
#读写二进制文件
def rw_jpg():
    try:
        with open('test_r.jpg','rb') as fs1:
            data = fs1.read()
            print(type(data))
        with open('test_w.jpg','wb') as fs2:
            fs2.write(data)
    except FileNotFoundError as ex:
        print('指定文件无法打开。')
    except IOError as ex:
        print('无法读写')
    print('程序执行完毕')

rw_jpg()

#%%
#json
"""
dump - 将Python对象按照JSON格式序列化到文件中
dumps - 将Python对象处理成JSON格式的字符串
load - 将文件中的JSON数据反序列化成对象
loads - 将字符串的内容反序列化成Python对象
"""
import json

def json_test():
    mydict = {
        'name':'扶楚旸',
        'age' : 23,
        'qq'  : 362483671,
        'friends':['陈菲娴','乔柔'],
        'cars':[
            {'barnd':'BYD','max_speed':180},
            {'barnd':'Audi','max_speed':250},
            {'barnd':'Benz','max_speed':320}
        ]
    }
    try:
        with open('python_practice/data/data.json','w',encoding = 'utf-8') as fs:
            json.dump(mydict,fs)
    except IOError as ex:
        print(ex)
    print('保存数据完成')

json_test()
    

# %%
import requests
import json


def news_test():
    resp = requests.get('http://api.tianapi.com/guonei/?key=APIKey&num=10')
    data_model = json.loads(resp.text)
    for news in data_model['newslist']:
        print(news['title'])

news_test()


# %%
