#连乘形式
from astropy.io import fits # 加载科学包
import math
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import random
rl = pd.read_csv("rlll.csv", sep=",")
rl1=rl.loc[:, ["x"]]
rl2=rl.loc[:, ["y"]]
rl1.dropna(how='any', subset=["x"], inplace=True)
rl2.dropna(how='any', subset=["y"], inplace=True)
rl1=np.array(rl1)
rl2=np.array(rl2)
i0=42.875
n=1/4
x0=0
x1=0
x2=0
i2=0
z1=np.ones(1000)
def I0(r,r0,i0):
    return i0*np.exp(-((r/r0)**(1/4)))


def pr(i1,r,r0):
    return np.exp(-(((i1-I0(r,r0,i0))**2)/i1**2))


for i in range(1000):
    rt = random.randint(14, 80)
    print(rt)
    i1=rl2[rt]
    r=rl1[rt]
    y=np.arange(0.001,10,0.01)
    z=pr(i1,r,y)
    z1=z1*z

fig = plt.figure()
plt.plot(rl1, rl2,'g-')
y3=y=np.arange(0.001,10,0.01)
fig = plt.figure()
plt.plot(y3, z1,'g-')
plt.axvline(np.argmax(z1)/100,ls=':')
print('r0最大概率：',np.where(z1==max(z1)))
plt.show()