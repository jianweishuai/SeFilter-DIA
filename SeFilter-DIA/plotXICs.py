# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:26:47 2023

@author: hqz
"""

import pickle
import matplotlib.pyplot as plt

data_folder = "F:/SeFilter-DIA/data/PXD027512/"
xic_file = data_folder + "P20715-PHOS-DIA-Colo-SL-Vir-3.npy"

with open(xic_file, 'rb') as file:
    xic_data = pickle.load(file)

plt.figure(figsize=(5,20))

count = 0
for seq, info in xic_data.items():
    if '(' not in seq:
        continue
    
    count += 1
    
    plt.subplot(20,1,count)
    
    for xic in info['xics']:
        plt.plot(xic)
    
    if count == 20:
        break