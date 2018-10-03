import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import scipy.signal as signal
    
ONE_FREQ  = 1200
CERO_FREQ = 2400
MULT = 10
SAMPLE_FREQ=ONE_FREQ*MULT


def modSignal(x):
    t = np.linspace(0,1/1200,MULT)
    sig=[]
    time=[]
    off=0
    for i in x:
        if(i==True):
            sig.extend(np.sin(2*np.pi*t*ONE_FREQ))
        else:
            sig.extend(np.sin(2*np.pi*t*CERO_FREQ))
        time.extend([x+off for x in t])
        off+=1/1200
    return sig,time;

def demSignal(s):
    d=int(round(446e-6*SAMPLE_FREQ,0))
    pad=[0]*d
    s1=s+pad
    s2=pad+s
    r=np.multiply(s1,s2)
    filter=signal.firwin(18,cutoff=ONE_FREQ/SAMPLE_FREQ,window="hamming")
    r=signal.lfilter(filter,[1.0],r)
    return r[:len(r)-d]


    


x=[0,0,0,0,0,1,1,1,1,1,0,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0]
tx=[]
for i in x:
    tx.extend([i]*MULT)
s,t=modSignal(x)
r=demSignal(s)
#t = np.linspace(0,1/1200*len(x),100*len(x))
tr=[]
for i in r:
    if(i>0):
        tr.append(0)
    else:
        tr.append(1)
plt.subplot(2,1,1);
plt.plot(t, s)
plt.plot(t, r)
plt.subplot(2,1,2);
for i in range(0,len(x)):
    plt.plot([i/ONE_FREQ,i/ONE_FREQ],[1,-1],color=[0.8,0.8,0.8],dashes=[6,2,2,2]);
plt.plot(t,tx)
plt.plot(t,tr)
plt.show()

