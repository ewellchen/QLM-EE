# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 08:42:49 2019

@author: quartz
"""

import keras.backend as K
def m_value(y_true, y_pred):
    m = K.mean(y_pred)
    return m
