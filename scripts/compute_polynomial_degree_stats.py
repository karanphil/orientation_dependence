import os

import numpy as np


data = "/media/karp2601/T7/data/MT_Diffusion/myelo_inferno/characterization_mean/"
bundles = ["AF_L", "AF_R", "CC_1", "CC_2a", "CC_2b", "CC_3", "CC_4", "CC_5",
           "CC_6", "CC_7", "CG_L", "CG_R", "CR_L", "CR_R", "CST_L", "CST_R",
           "ICP_L", "ICP_R", "IFOF_L", "IFOF_R", "ILF_L", "ILF_R", "MCP",
           "OR_L", "OR_R", "SLF_1_L", "SLF_1_R", "SLF_2_L", "SLF_2_R",
           "SLF_3_L", "SLF_3_R", "UF_L", "UF_R"]

subjects = ['03', '04', '05', '06', '07', '08', '10', '11', '12', '14', '15',
            '16', '18', '19', '20', '21', '22', '23', '24', '26']
sessions = ['1', '2', '3', '4', '5']

measures = ['MTR', 'MTsat', 'ihMTR', 'ihMTsat']

degs = np.zeros((33, 4, 100))

n = 0
for sub in subjects:
    for ses in sessions:
        for i, bundle in enumerate(bundles):
            r = np.load(data + "sub-0{}-hc_ses-{}/5_degree_bins/{}/1f_polyfits.npz".format(sub, ses, bundle))
            for j, m in enumerate(measures):
                polyfit = r['{}_polyfit'.format(m)]
                deg = np.sum(polyfit != 0) - 1
                degs[i, j, n] = deg
        n += 1

mean_deg = np.mean(degs, axis=-1)
std_deg = np.std(degs, axis=-1)

print(np.round(mean_deg, decimals=2))
print(np.round(std_deg, decimals=2))
