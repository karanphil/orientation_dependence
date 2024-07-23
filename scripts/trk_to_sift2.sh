path=$(pwd);
split_path=$(echo $path | tr "/" "\n");
divided_path=( ${split_path[0]} );
location=${divided_path[1]};
if [ $location = "local" ];
    then
    wdir="/home/local/USHERBROOKE/karp2601/Samsung/data/MT_Diffusion/myelo_inferno";
    source_dir="Research/source";
fi;
if [ $location = "pkaran" ];
    then
    wdir="/home/pkaran/Samsung/data/MT_Diffusion/myelo_inferno";
    source_dir="source";
fi;
cd $wdir;
initial_trk="tractograms/sub-026-hc_ses-3/sub-026-hc_ses-3__whole_brain_ensemble_tracts_comp.trk";
out_dir="test_tracto_to_bundle/test_tracto";
main_dir="test_tracto_to_bundle";

scil_tractogram_remove_invalid.py $initial_trk ${out_dir}/remove_invalid.trk --remove_single_point --remove_overlapping_points;
scil_tractogram_detect_loops.py ${out_dir}/remove_invalid.trk ${out_dir}/remove_loops.trk --display_counts --processes 8
scil_volume_math.py addition ${main_dir}/sub-026-hc_ses-3__wm_mask.nii.gz ${main_dir}/sub-026-hc_ses-3__gm_mask.nii.gz ${out_dir}/wm_gm_map.nii.gz;
scil_volume_math.py lower_threshold_eq ${out_dir}/wm_gm_map.nii.gz 0.1 ${out_dir}/wm_gm_mask.nii.gz;
scil_volume_math.py convert ${out_dir}/wm_gm_mask.nii.gz ${out_dir}/wm_gm_mask.nii.gz --data_type uint8 -f;
scil_tractogram_cut_streamlines.py ${out_dir}/remove_loops.trk ${out_dir}/cut_streamlines.trk --mask ${out_dir}/wm_gm_mask.nii.gz --processes 8 --trim_endpoints;
scil_volume_math.py lower_threshold_eq ${main_dir}/sub-026-hc_ses-3__gm_mask.nii.gz 0.1 ${out_dir}/gm_mask.nii.gz;
scil_volume_math.py dilation ${out_dir}/gm_mask.nii.gz 1 ${out_dir}/gm_mask_dilated.nii.gz;
scil_volume_math.py convert ${out_dir}/gm_mask_dilated.nii.gz ${out_dir}/gm_mask_dilated.nii.gz --data_type uint8 -f;
scil_tractogram_filter_by_roi.py ${out_dir}/cut_streamlines.trk ${out_dir}/filter_by_roi_gm.trk --drawn_roi ${out_dir}/gm_mask_dilated.nii.gz 'either_end' 'include' --display_counts;
scil_tractogram_filter_by_length.py ${out_dir}/filter_by_roi_gm.trk ${out_dir}/filter_by_length.trk --minL 30 --maxL 200 --display_counts;