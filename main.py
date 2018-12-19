
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close("all")

data = 'well_log'
#data = 'bitcoin'


# In[2]:


if data ==  'bitcoin':
    T = 500
    btc = pd.read_csv('data/btc.csv')
#    ltc = pd.read_csv('ltc.csv')
    y_ = btc.iloc[-T:]['price(USD)'].values
    y_  = y_[::5]
    #y_[1,:] = etc.iloc[-T:]['price(USD)'].values
    y = np.log(y_[1:] / y_[0:-1])


# ======================

    thresh = .95
    y_label = 'Price (USD)'
    x_label = 'Time (weeks)'

    f, (ax1, ax2) = plt.subplots(2, 1,figsize=(12,10))

    ax1.plot(y_)
    ax1.set_ylabel(y_label)
    ax1.set_title('Bitcoin Data')
    ax1.set_xlabel(x_label)
    ax2.plot(y)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Log ratio')
    


    
if data == 'well_log':
    y = np.loadtxt('data/well_log.txt')
    y = np.log(y[::10])-11.7
    y_ = y

    thresh = .8
    y_label = 'Response'
    x_label = 'Time'

    plt.figure(figsize=(12,6))
    plt.plot(y_,'r')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Well Log Data')
    
    axes = plt.gca()
    axes.set_ylim([-.6,.2])


# In[3]:

y.T.shape


# In[4]:


if data == 'well_log':
    A = np.array([[1]])
    C = np.array([[1]])
    T = y.shape[0]
    u_0 = np.array([0]) 
    R = np.array([[0.0005]])
    Q =  np.array([[0.000001]])
    P_0 = np.array([[0.01]])
if data == 'bitcoin':
    A = np.array([[1]])
    C = np.array([[1]])
    T = y.shape[0]-1
    u_0 = np.array([0]) 
    R = np.array([[0.001]])
    Q =  np.array([[0.000001]])
    P_0 = np.array([[0.005]])


# In[5]:


def KalmanFilter(m,P,y,reset = False):
    global A,C,Q,R
    if reset is False:
        m = np.dot(A,m)
        P = np.dot(A,np.dot(P,A.T)) + Q
    else:
        m = u_0
        P = P_0
    ############################
    S = np.dot(C,np.dot(P,C.T)) + R
    G = np.dot(P,np.dot(C.T,np.linalg.inv(S)))
    e = y - np.dot(C,m)
    m = m + np.dot(G,e)
    P = P - np.dot(G,np.dot(C,P))
    _ , l_ = np.linalg.slogdet(2*np.pi*S)
    l =  -0.5* l_ - 0.5 * np.dot(e.T,np.dot(np.linalg.inv(S),e))
    ############################
    return m,P,l
                   


# In[6]:


class packet():
    def __init__(self, u, P, pro):
            self.u = u
            self.P = P
            self.pro = pro
            self.name = 'NoReset'
    def estimate(self, y): 
        return KalmanFilter(self.u,self.P,y)


# In[7]:


packet_0 = packet(u_0,P_0,1)


# In[8]:


packets = [[packet_0]]
for i in range(T):
    if i % 100 == 0:
        print(i)
    new_packets = []
    reset_u,reset_p,reset_l = KalmanFilter(0,0,y[i],reset=True)
    reset_packet = packet(reset_u,reset_p,0)
    reset_packet.name = 'Reset'
    for pckt in packets[-1]:
        m,P,l = pckt.estimate(y[i])
        reset_pro = np.exp(reset_l) /(np.exp(l) + np.exp(reset_l))
        #print(l,reset_l)
        new_packet = packet(m,P,(1-reset_pro)*pckt.pro)
        reset_packet.pro += pckt.pro * reset_pro 
        new_packets.append(new_packet)
    new_packets.append(reset_packet)
    packets.append(new_packets)
        


# In[9]:


reset_pro = []
estimate = []
no_reset_estimate = []
for i in packets:
    reset_pro.append(i[-1].pro)
    no_reset_estimate.append(i[0].u)
    est = 0
    for j in i:
        est += j.pro * j.u
    estimate.append(est)
reset_pro = np.asarray(reset_pro)
reset_pro = reset_pro[1:]
estimate = np.asarray(estimate)
estimate = estimate[1:]
no_reset_estimate = no_reset_estimate[1:]


# In[10]:


plt.figure(figsize=(12,6))

plt.plot(y,'r',label='observation')
plt.plot(estimate,'g',label='estimate with change-point ')
plt.plot(no_reset_estimate,'b',label='estimate without change-point ')

plt.xlabel(x_label)
plt.title('Estimated Hidden States')
plt.legend(loc='upper right')

if data == 'bitcoin':
    plt.ylabel('Log ratio')
else:
    plt.ylabel(y_label)
    axes = plt.gca()
    axes.set_ylim([-.3,.2])
    
    
# In[12]:

plot_reset = np.copy(reset_pro)
plot_reset[plot_reset<thresh] = 0

fig, ax1 = plt.subplots(figsize=(12,6))

color = 'tab:red'
ax1.set_xlabel(x_label)
ax1.set_ylabel(y_label, color=color)
ax1.plot(y_, color=color,linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)

ax1.set_title('Chanpe Point Probabilities')

if data == 'well_log':
#    axes = plt.gca()
    ax1.set_ylim([-.3,.2])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('change point probability', color=color)  # we already handled the x-label with ax1
ax2.stem(plot_reset, color=color)
ax2.tick_params(axis='y', labelcolor=color)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



"""
plt.figure(figsize=(12,6))
y = y - min(y)
plt.plot(y_,'r')
plot_reset = np.copy(reset_pro)
plot_reset[plot_reset<0.80] = 0
#plot_reset[plot_reset>0.6] = 1
plt.stem(plot_reset)
"""
