#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


# In[11]:


lowcut = 500  # Hz # Low cut for our butter bandpass filter
highcut = 3500 # Hz # High cut for our butter bandpass filter


# In[12]:


def _butter_bandpass(lowcut, highcut,fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    print(low,high)
    b, a = butter(order, [low, high], btype="band")
    return b, a
    
def butter_bandpass_filter(path,lowcut,highcut, order=5):
    fs, data = wavfile.read(path)
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# In[13]:


#output=butter_bandpass_filter('untitled.wav',lowcut,highcut)


# In[ ]:




