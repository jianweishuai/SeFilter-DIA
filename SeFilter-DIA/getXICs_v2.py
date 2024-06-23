# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:30:57 2023

@author: hqz
"""

import bisect
import pickle
import numpy as np
import pandas as pd
from pyteomics import mzxml
from multiprocessing import Pool

import matplotlib.pyplot as plt

XIC_WIDTH = 85
MS2_TOL = 20 * 1e-6

def ParserMzXMLFile(params):
    
    mzxml_file, pept_dict = params
    
    cycle_count = 0
    msms_spectra = []
    window_info = dict()
    
    with mzxml.read(mzxml_file) as reader:
        
        for spectrum in reader:
            if spectrum['msLevel'] == 1:
                cycle_count += 1
                data = {'msLevel':1, 'RT': spectrum['retentionTime']}
            if spectrum['msLevel'] == 2:
                
                precursor_mz = spectrum['precursorMz'][0]['precursorMz']
                window_width = spectrum['precursorMz'][0]['windowWideness']
                
                data = {'msLevel':2, "precursor":precursor_mz, \
                        "mz": spectrum['m/z array'], \
                        "intensity": spectrum['intensity array']}
                
                window_info[str(precursor_mz)] = window_width
                
            msms_spectra.append(data)
    
    window_range = []
    for mz, width in window_info.items():
        window_range.append([float(mz)-width/2, float(mz)+width/2])
    
    window_range = sorted(window_range, key=lambda x: x[0])
    
    print(len(window_range))
    
    return
    
    for i in range(1, len(window_range)):
        window_range[i][0] = window_range[i-1][1]
    
    rt_arr = []
    window = [w[0] for w in window_range]
    msms_data = [[[] for _ in range(cycle_count)] for _ in range(len(window))]
    
    cycle_id = -1
    for data in msms_spectra:
        if data['msLevel'] == 1:
            cycle_id += 1
            rt_arr.append(data['RT'])
        else:
            precursor = data['precursor']
            win_id = bisect.bisect_left(window, precursor) - 1
            
            msms_data[win_id][cycle_id] = data
    
    rt_arr = np.array(rt_arr)
    
    half_width = XIC_WIDTH // 2
    
    for seq, info in pept_dict.items():
        
        rt = info['RT']
        ms1_mz = info['ms1Mz']
        ms2_mz = info['ms2Mz']
        
        rt_id = bisect.bisect_left(rt_arr, rt) - 1
        win_id = bisect.bisect_left(window, ms1_mz) - 1
        
        xic_arr = []
        for mz in ms2_mz:
            mz_left = mz * (1.0 - MS2_TOL)
            mz_right = mz * (1.0 + MS2_TOL)
            
            tid = -1
            xic = [0] * XIC_WIDTH
            for t in range(rt_id-half_width, rt_id+half_width+1):
                tid += 1
                
                if t < 0 or t >= len(rt_arr):
                    continue
                
                if msms_data[win_id][t] == []:
                    continue
                
                mz_arr = msms_data[win_id][t]['mz']
                inten_arr = msms_data[win_id][t]['intensity']
                
                mz_id = bisect.bisect_left(mz_arr, mz_left)
                
                if mz_id >= len(mz_arr) - 1:
                    continue
                
                if mz_arr[mz_id] > mz_right:
                    continue
                
                for k in range(mz_id, len(mz_arr)):
                    xic[tid] += inten_arr[k]
                    
                    if mz_arr[k] > mz_right:
                        break
            
            xic_arr.append(xic)
        
        pept_dict[seq]['xics'] = xic_arr
        
    out_file = mzxml_file.replace('.mzXML', '.npy')
    
    with open(out_file, 'wb') as file:
        pickle.dump(pept_dict, file)


def GetXICFromDIANN(data_folder):
    
    report_file = data_folder + "DIA-NN/report.tsv"
    lib_file = data_folder + "DIA-NN/report-lib.tsv"
    
    lib_data = pd.read_csv(lib_file, sep='\t')
    report_data = pd.read_csv(report_file, sep='\t')
    
    lib_pept = dict()
        
    for i in range(len(lib_data)):
        
        seq = lib_data['FullUniModPeptideName'][i] + str(lib_data['PrecursorCharge'][i])
        
        ms1Mz = lib_data['PrecursorMz'][i]
        ms2Mz = lib_data['ProductMz'][i]
        
        if seq not in lib_pept:
            lib_pept[seq] = {'seq':seq, 'RT':0, 'ms1Mz': ms1Mz, 'ms2Mz': [ms2Mz]}
        else:
            lib_pept[seq]['ms1Mz'] = ms1Mz
            lib_pept[seq]['ms2Mz'].append(ms2Mz)
    
    uniq_files = list(set(lib_data['FileName']))
    
    for file in uniq_files:
        
        sub_data = report_data[report_data['File.Name'] == file]
        sub_data.reset_index(drop=True, inplace=True)
        
        pept_dict = {}
        
        for i in range(len(sub_data)):
            seq = sub_data['Modified.Sequence'][i] + str(sub_data['Precursor.Charge'][i])
            
            if seq in lib_pept:
                info = lib_pept[seq]
                info['RT'] = sub_data['RT'][i]
                
                pept_dict[seq] = info
        
        mzxml_file = data_folder + file.split('\\')[-1].split('.')[0] + ".mzXML"
        
        params = [mzxml_file, pept_dict]
        
        ParserMzXMLFile(params)
    
if __name__ == '__main__':
    
    data_folder = "F:/SeFilter-DIA/data/PXD027512/"
    
    GetXICFromDIANN(data_folder)
