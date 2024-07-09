wdir="/home/pkaran/Samsung/data/MT_Diffusion/myelo_inferno";
#wdir="/home/local/USHERBROOKE/karp2601/Samsung/data/MT_Diffusion/myelo_inferno";
source_dir="source";
#source_dir="Research/source";
cd $wdir;
bin_width=5;
bin_width_dir="${bin_width}_degree_bins";
polyfit_commands="--save_polyfit --use_weighted_polyfit --poly_order 10 --scale_poly_order";
#polyfit_commands="";
for bundle in bundles/sub-026-hc_ses-3/bundles/*;
    do bundle_name=$(basename -- "${bundle%%.*}");
    if [ 1 == 1 ];
    # if [ $bundle_name != "CR_L" ] && [ $bundle_name != "CR_R" ];
        then echo $bundle_name;
        #roi_argument="new_correction/sub-026-hc_ses-3/fixel_analysis/bundle_mask_only_${bundle_name}.nii.gz";
        roi_argument="bundles/sub-026-hc_ses-3/masks/${bundle_name}.nii.gz";
        #pathdir="new_correction/sub-026-hc_ses-3/single_bundle_analysis/${bin_width_dir}/${bundle_name}";
        correction_path="new_correction/sub-026-hc_ses-3/multi_bundle_analysis/${bin_width_dir}/correction";
        pathdir="new_correction/sub-026-hc_ses-3/multi_bundle_analysis/${bin_width_dir}/${bundle_name}/post_correction";
        mkdir -p $pathdir;
        python ~/${source_dir}/orientation_dependence/scripts/scil_characterize_orientation_dependence.py FODF_metrics/sub-026-hc_ses-3/new_peaks/peaks.nii.gz  FODF_metrics/sub-026-hc_ses-3/new_peaks/peak_values.nii.gz DTI_metrics/sub-026-hc_ses-3/sub-026-hc_ses-3__dti_fa.nii.gz FODF_metrics/sub-026-hc_ses-3/new_peaks/nufo.nii.gz wm_mask/sub-026-hc_ses-3/sub-026-hc_ses-3__wm_mask.nii.gz $pathdir --measures ${correction_path}/MTR_corrected.nii.gz ${correction_path}/ihMTR_corrected.nii.gz ${correction_path}/MTsat_corrected.nii.gz ${correction_path}/ihMTsat_corrected.nii.gz --in_e1 DTI_metrics/sub-026-hc_ses-3/sub-026-hc_ses-3__dti_evecs_v1.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --in_roi $roi_argument $polyfit_commands --save_plots --bin_width_1f $bin_width --min_nb_voxels 1 --compute_three_fiber_crossings;
    fi;
done;