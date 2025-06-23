import os

import numpy as np
import nibabel as nib


fixel_anal = "/media/karp2601/T7/data/MT_Diffusion/myelo_inferno/fixel_analysis/"
fodf = "/media/karp2601/T7/data/MT_Diffusion/myelo_inferno/FODF_metrics/"
bundles = ["AF_L", "AF_R", "CC_1", "CC_2a", "CC_2b", "CC_3", "CC_4", "CC_5",
           "CC_6", "CC_7", "CG_L", "CG_R", "CR_L", "CR_R", "CST_L", "CST_R",
           "ICP_L", "ICP_R", "IFOF_L", "IFOF_R", "ILF_L", "ILF_R", "MCP",
           "OR_L", "OR_R", "SLF_1_L", "SLF_1_R", "SLF_2_L", "SLF_2_R",
           "SLF_3_L", "SLF_3_R", "UF_L", "UF_R"]

subjects = ['03', '04', '05', '06', '07', '08', '10', '11', '12', '14', '15',
            '16', '18', '19', '20', '21', '22', '23', '24', '26']
sessions = ['1', '2', '3', '4', '5']

fraction = 0
for sub in subjects:
    for ses in sessions:
        vol = nib.load(fixel_anal + "sub-0{}-hc_ses-{}/nb_bundles_per_voxel_voxel-norm.nii.gz".format(sub, ses))
        nb = vol.get_fdata()
        vol = nib.load(fodf + "sub-0{}-hc_ses-{}/new_peaks/nufo.nii.gz".format(sub, ses))
        nufo = vol.get_fdata()
        nufo1 = nufo == 1
        single_bundle = np.sum(nb[nufo1] == 1)
        multi_bundle = np.sum(nb[nufo1] > 1)
        frac = multi_bundle / (single_bundle + multi_bundle)
        fraction += frac

fraction /= 100
print(fraction)
