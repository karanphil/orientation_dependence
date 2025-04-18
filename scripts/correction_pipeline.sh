#!/usr/bin/bash
# This is a script for computing the correction of orientation dependence of MT
# measures on a single subject/session. This HAS TO be launched from the
# myelo_inferno directory.

data=$1;  # The first input of the script is the subject/session ID.
source=$2;  # The second input of the script is the source directory.
reference=$3; # The third input of the script is the reference type.

# reference="mean";
# reference="maximum";
# reference="max_mean"; # be careful with this one.

# All steps:

do_filter_trk=false;
do_sift2=false;
do_bundles=false;
do_fixel_density=false;
do_characterize_original=false; do_characterize_original_mf=false;
do_plot_original=false;
do_correction=false;
do_characterize_corrected=false; do_characterize_corrected_mf=true;
do_plot_corrected=false;

#---------------------------------- FIRST STEP --------------------------------
# Filter tractogram from Imeka (which is messy).

initial_trk="tractograms/${data}/${data}__whole_brain_ensemble_tracts_comp.trk";
filtered_trk="tractograms/${data}/filtered_tracts.trk";

if $do_filter_trk;
    then
    echo "FIRST STEP";
    echo "remove invalid";
    scil_tractogram_remove_invalid.py $initial_trk $filtered_trk --remove_single_point --remove_overlapping_points -f;
    #cp $initial_trk $filtered_trk;
    echo "detect loops";
    scil_tractogram_detect_loops.py $filtered_trk $filtered_trk --display_counts --processes 8 -f;

    echo "cut streamlines";
    scil_volume_math.py addition tissue_segmentation/${data}/${data}__wm_mask.nii.gz tissue_segmentation/${data}/${data}__gm_mask.nii.gz tissue_segmentation/${data}/wm_gm_map.nii.gz -f;
    scil_volume_math.py lower_threshold_eq tissue_segmentation/${data}/wm_gm_map.nii.gz 0.1 tissue_segmentation/${data}/wm_gm_mask.nii.gz -f;
    scil_volume_math.py convert tissue_segmentation/${data}/wm_gm_mask.nii.gz tissue_segmentation/${data}/wm_gm_mask.nii.gz --data_type uint8 -f;
    scil_tractogram_cut_streamlines.py $filtered_trk $filtered_trk --mask tissue_segmentation/${data}/wm_gm_mask.nii.gz --processes 8 --trim_endpoints -f;

    echo "filter by roi";
    scil_volume_math.py lower_threshold_eq tissue_segmentation/${data}/${data}__gm_mask.nii.gz 0.1 tissue_segmentation/${data}/gm_mask.nii.gz -f;
    scil_volume_math.py dilation tissue_segmentation/${data}/gm_mask.nii.gz 1 tissue_segmentation/${data}/gm_mask_dilated.nii.gz -f;
    scil_volume_math.py convert tissue_segmentation/${data}/gm_mask_dilated.nii.gz tissue_segmentation/${data}/gm_mask_dilated.nii.gz --data_type uint8 -f;
    scil_tractogram_filter_by_roi.py $filtered_trk $filtered_trk --drawn_roi tissue_segmentation/${data}/gm_mask_dilated.nii.gz 'either_end' 'include' --display_counts -f;

    echo "filter by length";
    scil_tractogram_filter_by_length.py $filtered_trk $filtered_trk --minL 30 --maxL 200 --display_counts -f;

fi;

#---------------------------------- SECOND STEP -------------------------------
# Run SIFT2 on the filtered tractogram.

d_fodf=FODF_metrics/${data}/${data}__fodf.nii.gz;
t_fodf=FODF_metrics/${data}/fodf_tournier.nii.gz
converted_trk="tractograms/${data}/filtered_tracts.tck";
weights="tractograms/${data}/sift2_weights.txt";
weighted_trk="tractograms/${data}/weighted_tracts.trk";

if $do_sift2;
    then
    echo "SECOND STEP";

    scil_sh_convert.py $d_fodf $t_fodf descoteaux07_legacy tournier07 -f;
    scil_tractogram_convert.py $filtered_trk $converted_trk -f;

    tcksift2 $converted_trk $t_fodf $weights -force;
    scil_tractogram_add_dps.py $filtered_trk $weights sift2 $weighted_trk -f;

    rm $converted_trk $t_fodf $filtered_trk;

fi;

#---------------------------------- THIRD STEP --------------------------------
# Extract bundles from weighted tractogram.

if $do_bundles;
    then
    echo "THIRD STEP";
    mkdir -p bundles/${data}/bundles_weighted;
    for bundle in bundles/${data}/bundles/*.trk;
        do base_name=$(basename -- "${bundle%%.*}");
        filtered_bundle="bundles/${data}/bundles/${base_name}_filtered.trk";
        scil_tractogram_remove_invalid.py $bundle $filtered_bundle --remove_single_point --remove_overlapping_points -f;
        scil_tractogram_detect_loops.py $filtered_bundle $filtered_bundle --display_counts --processes 8 -f;
        scil_tractogram_cut_streamlines.py $filtered_bundle $filtered_bundle --mask tissue_segmentation/${data}/wm_gm_mask.nii.gz --processes 8 --trim_endpoints -f;
        scil_tractogram_filter_by_roi.py $filtered_bundle $filtered_bundle --drawn_roi tissue_segmentation/${data}/gm_mask_dilated.nii.gz 'either_end' 'include' --display_counts -f;
        scil_tractogram_filter_by_length.py $filtered_bundle $filtered_bundle --minL 30 --maxL 200 --display_counts -f;

        weighted_bundle="bundles/${data}/bundles_weighted/${base_name}.trk";
        scil_tractogram_math.py intersection $weighted_trk $filtered_bundle $weighted_bundle -p 3 -f; 

        rm $filtered_bundle;
    done;

fi;

#---------------------------------- FOURTH STEP -------------------------------
# Compute fixel density maps.

weighted_bundles="bundles/${data}/bundles_weighted";

fixel_analysis="fixel_analysis/${data}";
# fixel_analysis="fixel_analysis_bundle_subset/${data}";

bundle_subset=${weighted_bundles}/*.trk
# bundle_subset="${weighted_bundles}/AF_L.trk ${weighted_bundles}/AF_R.trk ${weighted_bundles}/CC_1.trk ${weighted_bundles}/CC_2a.trk ${weighted_bundles}/CC_2b.trk ${weighted_bundles}/CC_3.trk ${weighted_bundles}/CC_4.trk ${weighted_bundles}/CC_5.trk ${weighted_bundles}/CC_6.trk ${weighted_bundles}/CC_7.trk ${weighted_bundles}/CG_L.trk ${weighted_bundles}/CG_R.trk ${weighted_bundles}/CST_L.trk ${weighted_bundles}/CST_R.trk ${weighted_bundles}/IFOF_L.trk ${weighted_bundles}/IFOF_R.trk ${weighted_bundles}/ILF_L.trk ${weighted_bundles}/ILF_R.trk ${weighted_bundles}/OR_L.trk ${weighted_bundles}/OR_R.trk ${weighted_bundles}/UF_L.trk ${weighted_bundles}/UF_R.trk";

bundles_list="AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R MCP";
# bundles_list="AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CST_L CST_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R UF_L UF_R";

if $do_fixel_density;
    then
    echo "FOURTH STEP";
    mkdir -p $fixel_analysis;
    scil_bundle_fixel_analysis.py FODF_metrics/${data}/new_peaks/peaks.nii.gz --in_bundles $bundle_subset --dps_key sift2 --split_bundles --out_dir $fixel_analysis --rel_thr 0.1 --abs_thr 1.5 --processes 8 -f --norm voxel none;

    # rm ${fixel_analysis}/fixel_density_mask*;
    # rm ${fixel_analysis}/nb_bundles*;
    rm ${fixel_analysis}/voxel_density_map_*;
    # rm ${fixel_analysis}/voxel_density_masks*;
    rm ${fixel_analysis}/voxel_density_mask_none-norm*;
    rm ${fixel_analysis}/fixel_density_map_*;

fi;

#---------------------------------- FIFTH STEP --------------------------------
# Characterize the bundles on original measures.

bundles_masks="";
bundles_names="";
bin_width=5;
bin_width_dir="${bin_width}_degree_bins";
out_original="characterization_${reference}/${data}/${bin_width_dir}";

# for bundle in bundles/${data}/bundles_weighted/*.trk;
for bundle in $bundle_subset;
    do bundle_name=$(basename -- "${bundle%%.*}");
    bundles_masks+=${fixel_analysis}/voxel_density_mask_voxel-norm_${bundle_name}.nii.gz;
    bundles_masks+=" ";
    bundles_names+=$bundle_name;
    bundles_names+=" ";
    if $do_characterize_original;
        then
        mkdir -p ${out_original}/${bundle_name};

    fi;

done;

if $do_characterize_original;
    then
    echo "FIFTH STEP";

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_characterization.py FODF_metrics/${data}/new_peaks/peaks.nii.gz DTI_metrics/${data}/${data}__dti_fa.nii.gz FODF_metrics/${data}/new_peaks/nufo.nii.gz wm_mask/${data}/${data}__wm_mask.nii.gz $out_original --measures ihMT/${data}/${data}__MTR_warped.nii.gz ihMT/${data}/${data}__ihMTR_warped.nii.gz ihMT/${data}/${data}__MTsat_warped.nii.gz ihMT/${data}/${data}__ihMTsat_warped.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles $bundles_masks --bundles_names $bundles_names --bin_width_sf $bin_width --min_nb_voxels 1 --min_frac_pts 0.5 --stop_crit 0.055 --save_polyfit --use_weighted_polyfit --patch --reference $reference;

    scil_volume_math.py union ${fixel_analysis}/voxel_density_mask_voxel-norm_*.nii.gz ${fixel_analysis}/voxel_density_mask_WM.nii.gz -f;
    mkdir -p ${out_original}/WM;
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_characterization.py FODF_metrics/${data}/new_peaks/peaks.nii.gz DTI_metrics/${data}/${data}__dti_fa.nii.gz FODF_metrics/${data}/new_peaks/nufo.nii.gz wm_mask/${data}/${data}__wm_mask.nii.gz $out_original --measures ihMT/${data}/${data}__MTR_warped.nii.gz ihMT/${data}/${data}__ihMTR_warped.nii.gz ihMT/${data}/${data}__MTsat_warped.nii.gz ihMT/${data}/${data}__ihMTsat_warped.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles ${fixel_analysis}/voxel_density_mask_WM.nii.gz --bundles_names WM --bin_width_sf $bin_width --min_nb_voxels 1 --stop_crit 0.055 --save_polyfit --use_weighted_polyfit --reference $reference;

fi;

if $do_characterize_original_mf;
    then

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_characterization_multi_fibers.py FODF_metrics/${data}/new_peaks/peaks.nii.gz ${fixel_analysis}/fixel_density_masks_voxel-norm.nii.gz $out_original --measures ihMT/${data}/${data}__MTR_warped.nii.gz ihMT/${data}/${data}__ihMTR_warped.nii.gz ihMT/${data}/${data}__MTsat_warped.nii.gz ihMT/${data}/${data}__ihMTsat_warped.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles_names $bundles_names --lookuptable ${fixel_analysis}/bundles_LUT.txt --bin_width_mf $bin_width --min_nb_voxels 1;

fi;

#---------------------------------- SIXTH STEP --------------------------------
# Plot the characterization of the original measures.

if $do_plot_original;
    then
    echo "SIXTH STEP";
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures MTR MTsat --in_bundles ${out_original}/*/1f_results.npz --bundles_order $bundles_list WM -f --out_filename  ${out_original}/orientation_dependence_MT.png --in_polyfits ${out_original}/*/1f_polyfits.npz;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures ihMTR ihMTsat --in_bundles ${out_original}/*/1f_results.npz --bundles_order $bundles_list WM -f --out_filename  ${out_original}/orientation_dependence_ihMT.png --in_polyfits ${out_original}/*/1f_polyfits.npz;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures MTR MTsat ihMTR ihMTsat --in_bundles ${out_original}/*/1f_results.npz --bundles_order AF_L AF_R CC_3 CC_4 CG_L CG_R CST_L CST_R IFOF_L IFOF_R OR_L OR_R WM -f --in_polyfits ${out_original}/*/1f_polyfits.npz --out_filename ${out_original}/orientation_dependence.png;

fi;

#---------------------------------- SEVENTH STEP ------------------------------
# Correct the original measures.
if [ "$reference" = "max_mean" ];
    then
    max_mean_polyfits=$(find characterization_max_mean/means/*/1f_polyfits.npz ! -path '*WM*');
    max_mean_cmd="--max_mean_refs $max_mean_polyfits";
    out_original="characterization_mean/${data}/${bin_width_dir}";
else
    max_mean_cmd=""
fi;

if $do_correction;
    then
    echo "SEVENTH STEP";

    polyfits=$(find ${out_original}/*/1f_polyfits.npz ! -path '*WM*');

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_correction.py FODF_metrics/${data}/new_peaks/peaks.nii.gz ${fixel_analysis}/fixel_density_maps_voxel-norm.nii.gz ihMT/${data}/ --polyfits $polyfits --in_measures ihMT/${data}/${data}__MTR_warped.nii.gz ihMT/${data}/${data}__MTsat_warped.nii.gz ihMT/${data}/${data}__ihMTR_warped.nii.gz ihMT/${data}/${data}__ihMTsat_warped.nii.gz --measures_names MTR MTsat ihMTR ihMTsat --lookuptable ${fixel_analysis}/bundles_LUT.txt --keyword $reference $max_mean_cmd;

fi;

#---------------------------------- EIGHTH STEP -------------------------------
# Characterize the bundles on corrected measures.

out_corrected="correction_${reference}/${data}/${bin_width_dir}";

if $do_characterize_corrected;
    then
    echo "EIGHTH STEP";
    # for bundle in bundles/${data}/bundles/*.trk;
    for bundle in $bundle_subset;
        do bundle_name=$(basename -- "${bundle%%.*}");
        mkdir -p ${out_corrected}/${bundle_name};

    done;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_characterization.py FODF_metrics/${data}/new_peaks/peaks.nii.gz DTI_metrics/${data}/${data}__dti_fa.nii.gz FODF_metrics/${data}/new_peaks/nufo.nii.gz wm_mask/${data}/${data}__wm_mask.nii.gz $out_corrected --measures ihMT/${data}/MTR_${reference}_corrected.nii.gz ihMT/${data}/ihMTR_${reference}_corrected.nii.gz ihMT/${data}/MTsat_${reference}_corrected.nii.gz ihMT/${data}/ihMTsat_${reference}_corrected.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles $bundles_masks --bundles_names $bundles_names --bin_width_sf $bin_width --min_nb_voxels 1;

    mkdir -p ${out_corrected}/WM;
    scil_volume_math.py union ${fixel_analysis}/voxel_density_mask_voxel-norm_*.nii.gz ${fixel_analysis}/voxel_density_mask_WM.nii.gz -f;
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_characterization.py FODF_metrics/${data}/new_peaks/peaks.nii.gz DTI_metrics/${data}/${data}__dti_fa.nii.gz FODF_metrics/${data}/new_peaks/nufo.nii.gz wm_mask/${data}/${data}__wm_mask.nii.gz $out_corrected --measures ihMT/${data}/MTR_${reference}_corrected.nii.gz ihMT/${data}/ihMTR_${reference}_corrected.nii.gz ihMT/${data}/MTsat_${reference}_corrected.nii.gz ihMT/${data}/ihMTsat_${reference}_corrected.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles ${fixel_analysis}/voxel_density_mask_WM.nii.gz --bundles_names WM --bin_width_sf $bin_width --min_nb_voxels 1;

fi;

if $do_characterize_corrected_mf;
    then

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_characterization_multi_fibers.py FODF_metrics/${data}/new_peaks/peaks.nii.gz ${fixel_analysis}/fixel_density_masks_voxel-norm.nii.gz $out_corrected --measures ihMT/${data}/MTR_${reference}_corrected.nii.gz ihMT/${data}/ihMTR_${reference}_corrected.nii.gz ihMT/${data}/MTsat_${reference}_corrected.nii.gz ihMT/${data}/ihMTsat_${reference}_corrected.nii.gz --measures_names MTR ihMTR MTsat ihMTsat --bundles_names $bundles_names --lookuptable ${fixel_analysis}/bundles_LUT.txt --bin_width_mf $bin_width --min_nb_voxels 1;

fi;

#---------------------------------- NINTH STEP --------------------------------
# Plot the characterization of the corrected measures.

if $do_plot_corrected;
    then
    echo "NINTH STEP";
    # Corrected only
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures MTR MTsat --in_bundles ${out_corrected}/*/1f_results.npz --bundles_order $bundles_list WM -f --out_filename  ${out_corrected}/orientation_dependence_MT.png --in_polyfits ${out_corrected}/*/1f_polyfits.npz;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures ihMTR ihMTsat --in_bundles ${out_corrected}/*/1f_results.npz --bundles_order $bundles_list WM -f --out_filename  ${out_corrected}/orientation_dependence_ihMT.png --in_polyfits ${out_corrected}/*/1f_polyfits.npz;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures MTR MTsat ihMTR ihMTsat --in_bundles ${out_corrected}/*/1f_results.npz --bundles_order AF_L AF_R CC_3 CC_4 CG_L CG_R CST_L CST_R IFOF_L IFOF_R OR_L OR_R WM -f --in_polyfits ${out_corrected}/*/1f_polyfits.npz --out_filename ${out_corrected}/orientation_dependence.png;

    # Corrected and original
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures MTR MTsat --in_bundles ${out_original}/*/1f_results.npz --in_bundles ${out_corrected}/*/1f_results.npz --bundles_order $bundles_list WM -f --out_filename  ${out_corrected}/orientation_dependence_comparison_MT.png --in_polyfits ${out_original}/*/1f_polyfits.npz --in_polyfits ${out_corrected}/*/1f_polyfits.npz;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures ihMTR ihMTsat --in_bundles ${out_original}/*/1f_results.npz --in_bundles ${out_corrected}/*/1f_results.npz --bundles_order $bundles_list WM -f --out_filename  ${out_corrected}/orientation_dependence_comparison_ihMT.png --in_polyfits ${out_original}/*/1f_polyfits.npz --in_polyfits ${out_corrected}/*/1f_polyfits.npz;

    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures MTR MTsat ihMTR ihMTsat --in_bundles ${out_original}/*/1f_results.npz --in_bundles ${out_corrected}/*/1f_results.npz --bundles_order AF_L AF_R CC_3 CC_4 CG_L CG_R CST_L CST_R IFOF_L IFOF_R OR_L OR_R WM -f --in_polyfits ${out_original}/*/1f_polyfits.npz --in_polyfits ${out_corrected}/*/1f_polyfits.npz --out_filename ${out_corrected}/orientation_dependence_comparison.png;

fi;

#---------------------------------- TENTH STEP --------------------------------
# Do the tractometry!

# This has to be done outside this script, on all bundles. Also, change the input path.

# if $do_tractometry;
#     then
#     nextflow run ${source}/my_scripts/tractometry_flow_modify_light/main.nf --input ../input/ --nb_points 10 --processes 8 --use_provided_centroids false

# fi;