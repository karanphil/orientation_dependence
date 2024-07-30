# This is a script for computing the correction of orientation dependence of MT
# measures on a single subject/session. This HAS TO be launched from the
# myelo_inferno directory.

data=$1;  # The only input of the script is the subject/session ID.

# All steps:

do_filter_trk=true;
do_sift2=true;
do_bundles=true;

#---------------------------------- FIRST STEP --------------------------------
# Filter tractogram from Imeka (which is messy).

initial_trk="tractograms/${data}/${data}__whole_brain_ensemble_tracts_comp.trk";
filtered_trk="tractograms/${data}/filtered_tracts.trk";

if $do_filter_trk;
    then
    scil_tractogram_remove_invalid.py $initial_trk $filtered_trk --remove_single_point --remove_overlapping_points -f;
    scil_tractogram_detect_loops.py $filtered_trk $filtered_trk --display_counts --processes 8 -f;

    scil_volume_math.py addition tissue_masks/${data}__wm_mask.nii.gz tissue_masks/${data}__gm_mask.nii.gz tissue_masks/wm_gm_map.nii.gz -f;
    scil_volume_math.py lower_threshold_eq tissue_masks/wm_gm_map.nii.gz 0.1 tissue_masks/wm_gm_mask.nii.gz -f;
    scil_volume_math.py convert tissue_masks/wm_gm_mask.nii.gz tissue_masks/wm_gm_mask.nii.gz --data_type uint8 -f;
    scil_tractogram_cut_streamlines.py $filtered_trk $filtered_trk --mask tissue_masks/wm_gm_mask.nii.gz --processes 8 --trim_endpoints -f;

    scil_volume_math.py lower_threshold_eq tissue_masks/${data}__gm_mask.nii.gz 0.1 tissue_masks/gm_mask.nii.gz -f;
    scil_volume_math.py dilation tissue_masks/gm_mask.nii.gz 1 tissue_masks/gm_mask_dilated.nii.gz -f;
    scil_volume_math.py convert tissue_masks/gm_mask_dilated.nii.gz tissue_masks/gm_mask_dilated.nii.gz --data_type uint8 -f;
    scil_tractogram_filter_by_roi.py $filtered_trk $filtered_trk --drawn_roi tissue_masks/gm_mask_dilated.nii.gz 'either_end' 'include' --display_counts -f;

    scil_tractogram_filter_by_length.py $filtered_trk $filtered_trk --minL 30 --maxL 200 --display_counts -f;

fi;

#---------------------------------- SECOND STEP -------------------------------
# Run SIFT2 on the filtered tractogram.

d_fodf=FODF_metrics/${data}/${data}_fodf.nii.gz;
t_fodf=FODF_metrics/${data}/fodf_tournier.nii.gz
converted_trk="tractograms/${data}/filtered_tracts.tck";
weights="tractograms/${data}/sift2_weights.txt";
weighted_trk="tractograms/${data}/weighted_tracts.tck";

if $do_sift2;
    then

    scil_sh_convert.py $d_fodf $t_fodf descoteaux07_legacy tournier07 -f;
    scil_tractogram_convert.py $filtered_trk $converted_trk -f;

    tcksift2 $converted_trk $t_fodf $weights;
    scil_tractogram_add_dps.py $filtered_trk $weights sift2 $weighted_trk;

    rm $converted_trk $t_fodf $trk;

fi;

#---------------------------------- THIRD STEP --------------------------------
# Extract bundles from weighted tractogram.

if $do_bundles;
    then
    for bundle in bundles/${data}/bundles/*.trk;
        do base_name=$(basename $bundle);
        # TODO add filtering of bundle here.
        weighted_bundle="bundles/${data}/bundles/${base_name}_weighted.trk";
        scil_tractogram_math.py intersection $weighted_trk $bundle $weighted_bundle -p 3 -f; 
    done;

fi;