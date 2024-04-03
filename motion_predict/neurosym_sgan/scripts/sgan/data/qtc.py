# -*- coding: utf-8 -*-
"""
Copyright Nicola Bellotto, University of Lincoln, UK (2020)

"""

import numpy as np
from numpy.linalg import norm


def qtcc1(k, l, qbits= 6, continuous = False, base3 = False):

    #N = k.shape[0]

    #print("k shape=", k.shape)
    #print("l shape=", l.shape)

    N = len(k)
    if N != len(l):
       #ERROR
       print('ERROR: The input series must have the same length')
       return
    
    # interval QTC
    qi = np.ndarray((N-1,qbits), dtype='int') # qtc relations
    
    for n in range(0, N-1):
        # if (all(x==0 for x in k[n,:]) or all(x==0 for x in l[n,:])):
        #     qi[n] = 10* np.ones(qbits)
        #     continue
        # else:
        # k wrt l: away, static, close
        d0 = norm(k[n,:] - l[n,:])
        d1 = norm(k[n+1,:] - l[n,:])
        if d1 < d0: # getting closer
            qi[n,0] = -1
        elif d1 > d0:
            qi[n,0] = +1
        else:
            qi[n,0] = 0
        # l wrt k
        d1 = norm(l[n+1,:] - k[n,:])
        if d1 < d0:
            qi[n,1] = -1
        elif d1 > d0:
            qi[n,1] = +1
        else:
            qi[n,1] = 0
        # difference between point direction and reference line, within [0,2pi)
        # k direction
        ak = np.cross(k[n+1,:] - k[n,:], l[n,:] - k[n,:])
        if (norm(ak) < 0): # pointing left
            qi[n,2] = -1
        elif (norm(ak) > 0): # pointing right
            qi[n,2] = +1
        else:
            qi[n,2] = 0
        # l direction
        al = np.cross(l[n+1,:] - l[n,:], k[n,:] - l[n,:])
        if (norm(al) < 0): # pointing left
            qi[n,3] = -1
        elif (norm(al) > 0): # pointing right
            qi[n,3] = +1
        else:
            qi[n,3] = 0
        ### comment out below for qtc 4
        # # velocity of k wrt l
        # vk = norm(k[n+1,:] - k[n,:])
        # vl = norm(l[n+1,:] - l[n,:])
        # if vk < vl:
        #     qi[n,4] = -1
        # elif vk > vl:
        #     qi[n,4] = +1
        # else:
        #     qi[n,4] = 0
        # angle of k wrt l (relative angle)
        # NOTE: From note 49 of Van de Weghe PhD thesis (2004):
        # " if at least one of both objects stands still, then the [angle] character will be 0."

        # if (vk == 0) or (vl == 0):
        #     qi[n,5] = 0
        # elif (norm(ak) < norm(al)): # pointing left (of l)
        #     qi[n,5] = -1
        # elif (norm(ak) > norm(al)): # pointing right
        #     qi[n,5] = +1
        # else:
        #     qi[n,5] = 0

        if (all(x == 0 for x in qi[n])):
           qi[n] = 10* np.ones(qbits) 
    res = qi


    if continuous:    
        # include time instants
        qt = np.ndarray((2*(N-1),6), dtype='b')
        qt[0] = [0,0,0,0,0,0]
        for n in range(1, N-1):
            qt[2*n-1] = qi[n-1]
            for m in range(qt.shape[1]):
                if qi[n-1,m] == qi[n,m]:
                    qt[2*n,m] = qi[n,m]
                else:
                    qt[2*n,m] = 0
        qt[-1] = qi[-1]
        res = qt            
    

    if base3:
        L = len(res)
        dum = np.ndarray((L,1), dtype='H') # 16bit integer
        for n in range(0, L):
            dum[n] = sum((res[n,i]+1.0)*(3**i) for i in range(qbits))
        res = dum

    return res    
