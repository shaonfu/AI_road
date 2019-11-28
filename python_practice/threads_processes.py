from random import randint
from time import time,sleep

def download_task(filename):
    print('开始下载%s...' % filename)
    time_to_download = randint(5,10)
    sleep(time_to_download)
    print('%s下载完成'耗费了%d)