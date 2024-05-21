from statsmodels.sandbox.stats.runs import runstest_1samp
# x=[1,0,1,0,1,1,1,0,1,0,0,1,0,1,0,0,1,1,0,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0]
# print(runstest_1samp(x))
import random

# 生成长度为50的01序列
sequence = [random.choice([0, 1]) for _ in range(50)]

print(sequence)