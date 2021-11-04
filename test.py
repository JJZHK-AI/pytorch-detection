"""
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: myjjzhk@126.com
@Software : PyCharm
@file: test.py
@time: 2021-11-03 15:14:57
@desc: 
"""
import logging

import logging

logger1 = logging.getLogger('A')
logger1.setLevel(logging.DEBUG)

logger2 = logging.getLogger('B')
logger2.setLevel(logging.INFO)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('test.log')
# 再创建一个handler，用于输出到控制台
f2 = logging.FileHandler('test2.log')
ch = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)
f2.setFormatter(formatter)

logger1.addHandler(fh)
logger2.addHandler(f2)

logger1.debug('logger1 debug message')
logger1.info('logger1 info message')
logger1.warning('logger1 warning message')
logger1.error('logger1 error message')
logger1.critical('logger1 critical message')

logger2.info("ccccccc")

# print("灰白色","\033[29;1mhello\033[0m")
# print("红色","\033[31;1m  hello  \033[0m")
# print("黄绿色","\033[32;1m  hello  \033[0m")
# print("土黄色","\033[33;1m  hello  \033[0m")
# print("蓝色","\033[34;1m  hello  \033[0m")
# print("紫色","\033[35;1m  hello  \033[0m")
# print("绿色","\033[36;1m  hello  \033[0m")
# print("背景红色","\033[41;1mhello\033[0m")

