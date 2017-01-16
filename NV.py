import glob
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lorentzian(X,x0,c):
    return c**2/(np.pi*c*(c**2+(X-x0)**2))
    
def lorentzian_2(X,a,x0,c):
    return a*c**2/(np.pi*c*(c**2+(X-x0)**2))

def ODMRfit(X,a1,f1,w1,a2,f2,w2):
    return a1*lorentzian(X,f1,w1)+a2*lorentzian(X,f2,w2)

def ODMR_theory_min(B,theta):
    D = 2.87E9
    g = 28E6 #MHz/mT, B should be in mT
    
    f_0 = D + 3*g**2*B**2/(2*D)*np.sin(theta)**2
    f_1 = g*B*np.cos(theta)*np.sqrt( 1 + (g*B/(2*D)*np.tan(theta)*np.sin(theta))**2)
    
    return f_0-f_1

def ODMR_theory_max(B,theta):
    D = 2.87E9
    g = 28E6 #MHz/mT, B should be in mT
    
    f_0 = D + 3*g**2*B**2/(2*D)*np.sin(theta)**2
    f_1 = g*B*np.cos(theta)*np.sqrt( 1 + (g*B/(2*D)*np.tan(theta)*np.sin(theta))**2)
    
    return f_0+f_1

def RabiFit(t,a,f,phi,T1,c):
    return a*np.exp(-t/T1)*np.cos(2*np.pi*f*(t+phi))+c
                            

def RabiFit2(t,a1,f,phi,T1,c,a2,a3):
    return np.exp(-t/T1)*(a1*np.cos(2*np.pi*f*(t+phi))+a2*np.cos(2*np.pi*(f+0.0022)*(t+phi))+a3*np.cos(2*np.pi*(f-0.0022)*(t+phi)))+c
       
def pol(X,phi,theta,a):
    return a*(np.cos(X-phi)**2+(np.cos(theta)*np.sin(X-phi))**2)

def OrientODMR(X,Y,Z):
    X = X[1:,:]
    Y = Y[1:,:]
    Z = Z[1:,:]
    
    poptX, pcovX = curve_fit(ODMRfit,X[:,0],X[:,1],p0=(-4E8,X[np.argmin(X[:,1]),0],1E7,-4E8,2.87E9*2-X[np.argmin(X[:,1]),0],1E7),
                           maxfev=2500)
    plt.plot(X[:,0],X[:,1],'ro')
    plt.plot(X[:,0],ODMRfit(X[:,0],*poptX),'r-')

    poptY, pcovY = curve_fit(ODMRfit,Y[:,0],Y[:,1],p0=(-4E8,Y[np.argmin(Y[:,1]),0],1E7,-4E8,2.87E9*2-Y[np.argmin(Y[:,1]),0],1E7),
                           maxfev=2500)
    plt.plot(Y[:,0],Y[:,1],'go')
    plt.plot(Y[:,0],ODMRfit(Y[:,0],*poptY),'g-')

    poptZ, pcovZ = curve_fit(ODMRfit,Z[:,0],Z[:,1],p0=(-4E8,Z[np.argmin(Z[:,1]),0],1E7,-4E8,2.87E9*2-Z[np.argmin(Z[:,1]),0],1E7),
                           maxfev=2500)
    plt.plot(Z[:,0],Z[:,1],'bo')
    plt.plot(Z[:,0],ODMRfit(Z[:,0],*poptZ),'b-')

    g = 28E6 #NV gyromagnetic ratio in Hz/mT
    Bx = np.abs(poptX[4]-poptX[1])/(2*g)
    By = np.abs(poptY[4]-poptY[1])/(2*g)
    Bz = np.abs(poptZ[4]-poptZ[1])/(2*g)
    
    Sx = 1/(2*g)*np.sqrt(np.diag(pcovX)[1]+np.diag(pcovX)[4])
    Sy = 1/(2*g)*np.sqrt(np.diag(pcovY)[1]+np.diag(pcovY)[4])
    Sz = 1/(2*g)*np.sqrt(np.diag(pcovZ)[1]+np.diag(pcovZ)[4])

    print "X field: %.2f +/- %.2f mT \nY field: %.2f +/- %.2f mT \nZ field: %.2f +/- %.2f mT" % (Bx,Sx,By,Sy,Bz,Sz)

    Bz = Bz/0.755 #scale Bz to account for inhomogenous field strength under z rotation
    B = np.sqrt(Bx**2+By**2+Bz**2)
    SB = np.sqrt((2*Bx/B)**2*Sx**2 + (2*By/B)**2*Sy**2 + (2*Bz/B)**2*Sz**2)
    phi = np.arctan(By/Bx)*180/np.pi
    Sphi = np.sqrt( (-1/((1+(By/Bx)**2)*Bx**2))**2*Sx**2 + (By**2/(1+(By/Bx)**2))**2*Sy**2 )
    theta = np.arccos(Bz/B)*180/np.pi
    print "B: %.2f +/- %.2f\nPhi: %.2f +/- %.2f\nTheta: %.2f or %.2f" % (B,SB,phi,Sphi,theta,180-theta)

def OrientPol(T,mirror=False):
    
    T[:,1] = T[:,1]+np.linspace(0,T[0,1]-T[-1,1],num=T.shape[0])
    
    if mirror:
        T_half = T[T[:,0]<180]
        T_half_copy = T[T[:,0]<180]
        T_half_copy[:,0] = T_half_copy[:,0]+180
        T = np.append(T_half,T_half_copy,axis=0)
    
    T[:,0] = T[:,0]*2*np.pi/360
   
    phi_g = T[np.argmin(T[:,1]),0]-np.pi/2
    th_g = np.arccos(np.sqrt(np.min(T[:,1])/np.max(T[:,1])))
    opt1,cov1 = curve_fit(pol,T[:,0],T[:,1],p0=(phi_g,th_g,np.max(T[:,1])))

    plt.polar(T[:,0],T[:,1],'ko')
    plt.polar(T[:,0],pol(T[:,0],*opt1),'r-')

    print "Phi: %2f +/- %2f" % (opt1[0]*180/np.pi,np.sqrt(np.diag(cov1))[0]*180/np.pi)
    print "Theta: %2f or +/- %2f" % (opt1[1]*180/np.pi,np.sqrt(np.diag(cov1))[1]*180/np.pi)
    print "Phi(naive): %2f" % (phi_g*180/np.pi)
    print "Theta(naive): %2f" % (th_g*180/np.pi)
    
    
def EPR(EPRfiles,Pfiles):
    files = glob.glob(EPRfiles)
    pseq = np.loadtxt(Pfiles)
    d_sum = np.zeros((pseq.shape[0],))
    l = pseq.shape[0]
    for f in files:
        epr = np.loadtxt(f)
        diff = (epr[l:,0]-epr[l:,1])
        diff[np.isinf(diff)] = 0
        diff = np.nan_to_num(diff)
        diff.resize(((diff.shape[0]- (diff.shape[0] % l)),1))
        diff = np.reshape(diff,(pseq.shape[0],-1),order='F')
        d_sum += np.mean(diff,axis=1)
        
    return pseq,d_sum/len(files)



def EPR_NoRef(EPRfiles,Pfiles):
    pseq = np.loadtxt(Pfiles)
    files = glob.glob(EPRfiles)
    d_sum = np.zeros((pseq.shape[0],))
    l = pseq.shape[0]
    for f in files:
        epr = np.loadtxt(f)
        diff = epr[l:,0]
        diff = np.nan_to_num(diff)
        diff.resize(((diff.shape[0] - (diff.shape[0] % l)),1))
        diff = np.reshape(diff,(pseq.shape[0],-1),order='F')
        d_sum += np.mean(diff,axis=1)
        
    return pseq,d_sum/len(files)


def EPR_Test(EPRfiles,pseq):
    files = glob.glob(EPRfiles)
    d_sum = np.zeros((pseq.shape[0],))
    l = pseq.shape[0]
    for f in files:
        epr = np.loadtxt(f)
        diff = epr[l:,0]
        diff = np.nan_to_num(diff)
        diff.resize(((diff.shape[0] - (diff.shape[0] % l)),1))
        diff = np.reshape(diff,(pseq.shape[0],-1),order='F')
        d_sum += np.sum(diff,axis=1)
        
    return d_sum
    
def BatchEpr(base):
    files = glob.glob(base+'*')
    regex = re.compile(re.escape(base)+'(x\d+ y\d+ \d+) sec (\d+\.\d+) V   \d+')
    batches = sorted(list(set([(m.group(1),m.group(2)) for f in files for m in [regex.search(f)] if m])), key=lambda x: x[1])
    fits = np.zeros((len(batches),11))
    p_0=(15,15E-3,0,2000,100)
    bound = ([0,1E-4,-200,100,80],[1000,1E-1,200,5E4,150])
    for i,e in enumerate(batches):
        time,rabi = EPR(base+e[0]+' sec '+e[1]+' V *[0-9]',base+e[0]+' sec '+e[1]+' V   Pulse Seq')
        rabi = 100*rabi/np.mean(rabi)
        try:
            opt,cov = curve_fit(RabiFit,time[1:],rabi[1:],p0=p_0,bounds=bound,max_nfev=5500)
            err = np.sqrt(np.diag(cov))
        except Exception as exc:
            print "could not fit "+e[1]
            print exc
            opt = [0]*5
            err = [0]*5

        fits[i,0] = e[1]
        fits[i,1:6] = opt
        fits[i,6:12] = err

        plt.plot(time[1:],rabi[1:],'ko:')
        t = np.arange(0,np.max(time),1)
        plt.plot(t,RabiFit(t,*opt),'b-')
        plt.savefig(base+e[0]+' 1500 sec '+e[1]+' V.png')
        plt.clf()
        
    return fits