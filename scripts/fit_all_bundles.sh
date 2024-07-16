# wdir="/home/pkaran/Samsung/data/MT_Diffusion/myelo_inferno";
wdir="/home/local/USHERBROOKE/karp2601/Samsung/data/MT_Diffusion/myelo_inferno";
# source_dir="source";
source_dir="Research/source";
cd $wdir;
bin_width=5;
bin_width_dir="${bin_width}_degree_bins";
# polyfit_commands="--save_polyfit --use_weighted_polyfit --poly_order 10 --scale_poly_order";
polyfit_commands="--save_polyfit --use_weighted_polyfit";
# polyfit_commands="";
for bundle in bundles/sub-026-hc_ses-3/bundles/*;
    do bundles_names+=$(basename -- "${bundle%%.*}");
    bundles_names+=" ";
done;

roi_argument="test_tracto_to_bundle/fixel_analysis/bundle_masks/voxel_density_mask_*.nii.gz";
pathdir="toto";

python ~/${source_dir}/orientation_dependence/scripts/scil_orientation_dependence_fit.py FODF_metrics/sub-026-hc_ses-3/new_peaks/peaks.nii.gz  FODF_metrics/sub-026-hc_ses-3/new_peaks/peak_values.nii.gz DTI_metrics/sub-026-hc_ses-3/sub-026-hc_ses-3__dti_fa.nii.gz FODF_metrics/sub-026-hc_ses-3/new_peaks/nufo.nii.gz wm_mask/sub-026-hc_ses-3/sub-026-hc_ses-3__wm_mask.nii.gz $pathdir --measures ihMT/sub-026-hc_ses-3/sub-026-hc_ses-3__MTR_b1_warped.nii.gz ihMT/sub-026-hc_ses-3/sub-026-hc_ses-3__ihMTR_b1_warped.nii.gz ihMT/sub-026-hc_ses-3/sub-026-hc_ses-3__MTsat_b1_warped.nii.gz ihMT/sub-026-hc_ses-3/sub-026-hc_ses-3__ihMTsat_b1_warped.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles $roi_argument --bundles_names $bundles_names $polyfit_commands --bin_width_sf $bin_width --min_nb_voxels 1;