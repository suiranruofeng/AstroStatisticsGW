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
print(hdu[1].header)
r=hdu[1].data.field('rgc')
rv=hdu[1].data.field('rv')
rv_err=hdu[1].data.field('rv_err')
nansr = np.isnan(r)
rv = np.delete(rv, np.where(nansr))
rv_err= np.delete(rv_err, np.where(nansr))
r= np.delete(r, np.where(nansr))
m=np.mean(rv)
print(m)

def wudibaolongzhansheng(rv,rv_err,r,x):
    a=x[0]
    hr=x[1]
    dr = abs((r - 8.34) * 0.2)
    r0=stats.norm.rvs(loc=r, scale=dr, size=len(r))
    sig2=a*np.exp(-r/hr)
    sig2z=sig2+(rv_err**2)
    L = np.sum(-(rv-m)**2/sig2z-0.5*np.log(sig2z))
    return L


