import numpy as np
def trans(x):
    alpha=x[0];  a=x[1]; d=x[2]; t=x[3]

    T=np.array([[np.cos(t)                  ,     -np.sin(t)   ,       0    ,                              a],
                [np.sin(t)*np.cos(alpha)    ,     np.cos(t)*np.cos(alpha), -np.sin(alpha) , -np.sin(alpha)*d],
                [np.sin(t)*np.sin(alpha)     ,    np.cos(t)*np.sin(alpha) , np.cos(alpha) ,  np.cos(alpha)*d],
                [ 0                        ,  0                   ,      0    ,                         1   ]  
                ])
                
    return T


pi=np.pi/2
def dh(t):
    t1=t[0];t2=t[1];t3=t[2]
    dh = np.array([0 , 0,0,t1,1,0,0,t2,1.2,pi/2,0,t3,1.3,0,0,pi/2]).reshape(4,4)
    return dh

def pos(t):
    T=np.eye(4,4)
    for i in range(0,3):
        dhr= dh(t)
        T=T@trans(dhr[i,:])
    return T[0:3,3]

data=np.array([0,0,0,0,0,0])
for t1 in np.linspace(0,2*pi,100):
    for t2 in np.linspace(0,2*pi,100):
        for t3 in np.linspace(0,2*pi,100):
            t= np.array([t1,t2,t3])
            data=np.vstack((data,np.hstack((t,pos(t)))))

np.save("data.npy",data)
            