from astropy.io import fits
import math
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import random
import scipy.stats as stats
import scipy.special as special
import emcee
import corner
import math

hdu=fits.open( 'outerdisk_sample.fits' )
hdu.info()
y=hdu[1].data.field('feh')
dv=hdu[1].data.field('feh_err')
r=hdu[1].data.field('rgc')
nansy = np.isnan(y)
nansdv = np.isnan(dv)
nansr = np.isnan(r)
print(np.where(nansr))
y = np.delete(y, np.where(nansr))
dv= np.delete(dv, np.where(nansr))
r= np.delete(r, np.where(nansr))
nansr = np.isnan(r)
print(np.where(nansr))
print(np.max(y))






def houyan1(x,y,r,dv):
    d=x[0]
    d0=x[1]
    re=abs((r-8.34)*0.2)
    vv=d*r+d0
    dvv=abs(d*re)
    print(dvv)
    der = stats.norm.rvs(loc=vv, scale=dvv, size=len(r))
    L=np.sum((-(y-der)**2)/(2*dv)-0.5*np.log(2.*np.pi)-0.5*np.log(dv))
    return L

def houyantest(x,y,r,dv):
    d=x
    d0=0.53
    re=abs((r-8.34)*0.2)
    vv=d*r+d0
    dvv=abs(d*re)
    der = stats.norm.rvs(loc=vv, scale=dvv, size=len(r))
    L=np.sum((-(y-der)**2)/(2*dv)-0.5*np.log(2.*np.pi)-0.5*np.log(dv))
    return L





x=-0.5
z=np.ones(1000)
for i in range(1000):
    x+=0.001
    z[i]=houyantest(x,y,r,dv)
print(z)
x3=np.arange(-0.5,0.5,0.001)
fig = plt.figure()
plt.plot(x3, z,'g-')
plt.show()





