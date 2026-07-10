# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:13:54 2026

@author: codett
"""

import numpy as np

def make_private_load_profile(B, epsilon, P_profile, num_houses=1):
    b = 2 * B / epsilon
    noise = np.random.laplace(0, b, size=(num_houses, len(P_profile)))
    P_tilde = P_profile + np.sum(noise, axis=0)
    return P_tilde