def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys as sys
import scipy.integrate
from scipy import interpolate
import gp, dgp, covariance
import pickle
import numpy as np
from numpy import array,concatenate,loadtxt,savetxt,zeros
import matplotlib.pyplot as plt
from matplotlib import rc

if __name__=="__main__":

# load the data from inputdata.txt
    filename = 'hz_cc_sdss'
    (X,Y,Sigma,hid) = loadtxt(filename+'.dat',unpack='True')

    
    # nstar points of the function will be reconstructed 
    # between xmin and xmax
    xmin = 0.0
    xmax = 2.5
    nstar = 200

    # initial values of the hyperparameters
    initheta = [2.0, 2.0]

    # initialization of the Gaussian Process
    g = dgp.DGaussianProcess(X, Y, Sigma,covfunction=covariance.SquaredExponential,cXstar=(xmin,xmax,nstar))

    # training of the hyperparameters and reconstruction of the function
    (rec, theta) = g.gp(thetatrain='True')

    # reconstruction of the first, second and third derivatives.
    # theta is fixed to the previously determined value.
    (drec, theta) = g.dgp(thetatrain='True')
    (d2rec, theta) = g.d2gp()
    (d3rec, theta) = g.d3gp()

    # save the output
    savetxt("f.txt", rec)
    savetxt("df.txt", drec)
    savetxt("d2f.txt", d2rec)
    savetxt("d3f.txt", d3rec)


    # test if matplotlib is installed
    try:
        import matplotlib.pyplot
    except:
        print("matplotlib not installed. no plots will be produced.")
        exit
    # create plot


zrec = drec[:,0]
hzrec = rec[:,1]
sighzrec = rec[:,2]
hzdrec = drec[:,1]
sighdzrec = drec[:,2]


def function(y, z,H,dH):
  f1,f2,df1dz,df2dz = y
  ddf1dz          = ((1+z)*dH(z)*(-3-(1+z)*df1dz)+H(z)*(3+6*f1+6*f2+2*(1+z)*df1dz))/((1+z)**2*H(z))
  ddf2dz          = ((1+z)*dH(z)*(1-(1+z)*df2dz)+H(z)*(-3+2*f1+2*f2+2*(1+z)*df2dz))/((1+z)**2*H(z))
  return df1dz, df2dz, ddf1dz, ddf2dz

init = [
  0, #x[0]
  0,     #y[0]
  2*hzrec[0],     #x'[0]
  0.0    #y'[0]
]

k = 1

z  = zrec
H = interpolate.interp1d(zrec, hzrec, fill_value="extrapolate")
dH = interpolate.interp1d(zrec, hzdrec, fill_value="extrapolate")

values = scipy.integrate.odeint(function, init, z,args=(H,dH,), tfirst=False)


plt.plot(z,values[:,0], c ='#F08080')

z  = zrec
H = interpolate.interp1d(zrec, hzrec+1.*sighzrec, fill_value="extrapolate")
dH = interpolate.interp1d(zrec, hzdrec+1.*sighdzrec, fill_value="extrapolate")

values_upp = scipy.integrate.odeint(function, init, z,args=(H,dH,), tfirst=False)



z  = zrec
H = interpolate.interp1d(zrec, hzrec-1.*sighzrec, fill_value="extrapolate")
dH = interpolate.interp1d(zrec, hzdrec-1.*sighdzrec, fill_value="extrapolate")

values_low = scipy.integrate.odeint(function, init, z,args=(H,dH,), tfirst=False)

plt.fill_between(z, values_upp[:,0], values_low[:,0], facecolor='#F08080', alpha=0.30, interpolate=True)


plt.show()
