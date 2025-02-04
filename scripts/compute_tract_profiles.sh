#!/usr/bin/bash
# This is a script for computing the tract-profiles for the MT orientation
# dependence project. This HAS TO be launched from the
# myelo_inferno directory.

data=$1;  # The first input of the script is the subject/session ID.
source=$2;  # The second input of the script is the source directory.

python ${source}/orientation_dependence/scripts/scil_bundle_profiles.py tract_profiles/${data} --in_fixel_density_masks fixel_analysis/${data}/fixel_density_masks_voxel-norm.nii.gz --measures ihMT/${data}/${data}__MTR_warped.nii.gz ihMT/${data}/${data}__ihMTR_warped.nii.gz ihMT/${data}/${data}__MTsat_warped.nii.gz ihMT/${data}/${data}__ihMTsat_warped.nii.gz  --measures_names MTR ihMTR MTsat ihMTsat --bundles_names AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R MCP --bundles labels/${data}/* --suffix original

python ${source}/orientation_dependence/scripts/scil_bundle_profiles.py tract_profiles/${data} --in_fixel_density_masks fixel_analysis/${data}/fixel_density_masks_voxel-norm.nii.gz --measures ihMT/${data}/MTR_mean_corrected.nii.gz ihMT/${data}/ihMTR_mean_corrected.nii.gz ihMT/${data}/MTsat_mean_corrected.nii.gz ihMT/${data}/ihMTsat_mean_corrected.nii.gz  --measures_names MTR ihMTR MTsat ihMTsat --bundles_names AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R MCP --bundles labels/${data}/* --suffix mean_corrected

python ${source}/orientation_dependence/scripts/scil_bundle_profiles.py tract_profiles/${data} --in_fixel_density_masks fixel_analysis/${data}/fixel_density_masks_voxel-norm.nii.gz --measures ihMT/${data}/MTR_maximum_corrected.nii.gz ihMT/${data}/ihMTR_maximum_corrected.nii.gz ihMT/${data}/MTsat_maximum_corrected.nii.gz ihMT/${data}/ihMTsat_maximum_corrected.nii.gz  --measures_names MTR ihMTR MTsat ihMTsat --bundles_names AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R MCP --bundles labels/${data}/* --suffix maximum_corrected