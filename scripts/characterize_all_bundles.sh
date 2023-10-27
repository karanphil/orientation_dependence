wdir="/home/pkaran/Samsung/data/MT_Diffusion/myelo_inferno";
cd $wdir;
bin_width=1
bin_width_dir="${bin_width}_degree_bins"
poly_order=15;
for bundle in bundles/sub-004-hc_ses-4/masks/*;
    do bundle_name=$(basename -- "${bundle%%.*}");
    if [ 1 == 1 ];
    #if [ "$bundle_name" = "AF_L" ];
        then echo $bundle_name;
        mkdir "correction_per_bundle/characterization/${bin_width_dir}/${bundle_name}";
        python ~/source/orientation_dependence/scripts/scil_characterize_orientation_dependence.py FODF_metrics/sub-004-hc_ses-4/new_at/peaks.nii.gz  FODF_metrics/sub-004-hc_ses-4/new_at/peak_values.nii.gz DTI_metrics/sub-004-hc_ses-4/sub-004-hc_ses-4__dti_fa.nii.gz FODF_metrics/sub-004-hc_ses-4/new_at/nufo.nii.gz wm_mask/sub-004-hc_ses-4/sub-004-hc_ses-4__wm_mask.nii.gz correction_per_bundle/characterization/${bin_width_dir}/${bundle_name} --measures ihMT/sub-004-hc_ses-4/sub-004-hc_ses-4__MTR_warped.nii.gz ihMT/sub-004-hc_ses-4/sub-004-hc_ses-4__ihMTR_warped.nii.gz --in_e1 DTI_metrics/sub-004-hc_ses-4/sub-004-hc_ses-4__dti_evecs_v1.nii.gz --measures_names MTR ihMTR --save_npz_files --save_plots --in_roi $bundle --bin_width_1f $bin_width --poly_order $poly_order;
    fi;
done;