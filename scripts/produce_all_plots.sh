#!/usr/bin/bash
# This is a script for computing all the plots of the paper.
# This HAS TO be launched from the myelo_inferno directory.

source=$1;

# nb_voxels_args="--check_nb_voxels_std --min_nb_voxels 10";
nb_voxels_args="--min_nb_voxels 30";

for method in 'correction_max_mean' 'correction_mean' 'correction_maximum';
    do cd $method;
    if [ "$method" = "correction_max_mean" ];
    then polyfits="--in_polyfits ../characterization_max_mean/means/*/1f_polyfits.npz --in_polyfits ../characterization_max_mean/means/*/1f_polyfits.npz";
    else polyfits="";
    fi
    # -----------Plot subset of bundles-----------
    # single-fiber
    bundles="AF_R CC_2a CC_4 CG_R CST_R IFOF_R OR_R WM";
    name_bundle_choice="_subset";
    measures="MTR MTsat ihMTR ihMTsat";
    name="MT_ihMT";
    figsize="8 4.5";
    set_yticks="--set_yticks 26.3 19.4 --set_yticks 5.4 2.1 --set_yticks 12.2 4.1 --set_yticks 1.6 0.3"
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures $measures --in_bundles ../characterization_mean/means/*/1f_results.npz --in_bundles means/*/1f_results.npz --bundles_order $bundles $polyfits --out_filename ${name}_means_comparison${name_bundle_choice}.png --plot_std --write_mean_std --horizontal_test --figsize $figsize -f $set_yticks $nb_voxels_args --polyfits_to_plot 0;
    # multi-fiber
    bundles="AF_R CC_2a CC_4 CG_R CST_R IFOF_R OR_R";
    figsize="8 4";
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures $measures --in_bundles ../characterization_mean/means/*/mf_results.npz --in_bundles means/*/mf_results.npz --bundles_order $bundles --out_filename ${name}_mf_means_comparison${name_bundle_choice}.png --plot_std --write_mean_std --horizontal_test --figsize $figsize -f $set_yticks $nb_voxels_args;

    # -----------Plot all bundles-----------
    # single-fiber
    bundles="AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R MCP WM";
    name_bundle_choice="";
    figsize="8 8";
    # MT
    measures="MTR MTsat"; name="MT"; set_yticks="--set_yticks 26.3 19.4 --set_yticks 5.4 2.1";
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures $measures --in_bundles ../characterization_mean/means/*/1f_results.npz --in_bundles means/*/1f_results.npz --bundles_order $bundles $polyfits --out_filename ${name}_means_comparison${name_bundle_choice}.png --plot_std --write_mean_std --horizontal_test --figsize $figsize -f -v $set_yticks $nb_voxels_args --polyfits_to_plot 0 --save_stats;
    # ihMT
    measures="ihMTR ihMTsat"; name="ihMT"; set_yticks="--set_yticks 12.2 4.1 --set_yticks 1.6 0.3";
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures $measures --in_bundles ../characterization_mean/means/*/1f_results.npz --in_bundles means/*/1f_results.npz --bundles_order $bundles $polyfits --out_filename ${name}_means_comparison${name_bundle_choice}.png --plot_std --write_mean_std --horizontal_test --figsize $figsize -f -v $set_yticks $nb_voxels_args --polyfits_to_plot 0 --save_stats;
    # multi-fiber
    bundles="AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R MCP";
    name_bundle_choice="";
    figsize="8 8";
    # MT
    measures="MTR MTsat"; name="MT"; set_yticks="--set_yticks 26.3 19.4 --set_yticks 5.4 2.1";
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures $measures --in_bundles ../characterization_mean/means/*/mf_results.npz --in_bundles means/*/mf_results.npz --bundles_order $bundles --out_filename ${name}_mf_means_comparison${name_bundle_choice}.png --plot_std --write_mean_std --horizontal_test --figsize $figsize -f -v $set_yticks $nb_voxels_args --save_stats;
    # ihMT
    measures="ihMTR ihMTsat"; name="ihMT"; set_yticks="--set_yticks 12.2 4.1 --set_yticks 1.6 0.3";
    python ${source}/orientation_dependence/scripts/scil_orientation_dependence_plot.py --measures $measures --in_bundles ../characterization_mean/means/*/mf_results.npz --in_bundles means/*/mf_results.npz --bundles_order $bundles --out_filename ${name}_mf_means_comparison${name_bundle_choice}.png --plot_std --write_mean_std --horizontal_test --figsize $figsize -f -v $set_yticks $nb_voxels_args --save_stats;
    cd ..;
done;

# Plot F and V values
cd correction_max_mean;
python ${source}/orientation_dependence/scripts/plot_f_v_values.py --in_values_sf MT_means_comparison_f_values.txt ihMT_means_comparison_f_values.txt --in_values_sf ../correction_maximum/MT_means_comparison_f_values.txt ../correction_maximum/ihMT_means_comparison_f_values.txt --in_values_sf ../correction_mean/MT_means_comparison_f_values.txt ../correction_mean/ihMT_means_comparison_f_values.txt --in_values_mf MT_mf_means_comparison_f_values.txt ihMT_mf_means_comparison_f_values.txt --in_values_mf ../correction_maximum/MT_mf_means_comparison_f_values.txt ../correction_maximum/ihMT_mf_means_comparison_f_values.txt --in_values_mf ../correction_mean/MT_mf_means_comparison_f_values.txt ../correction_mean/ihMT_mf_means_comparison_f_values.txt  --out_filename f_values_plot.png -f --value_type f;

python ${source}/orientation_dependence/scripts/plot_f_v_values.py --in_values_sf MT_means_comparison_v_values.txt ihMT_means_comparison_v_values.txt --in_values_sf ../correction_maximum/MT_means_comparison_v_values.txt ../correction_maximum/ihMT_means_comparison_v_values.txt --in_values_sf ../correction_mean/MT_means_comparison_v_values.txt ../correction_mean/ihMT_means_comparison_v_values.txt --in_values_mf MT_mf_means_comparison_v_values.txt ihMT_mf_means_comparison_v_values.txt --in_values_mf ../correction_maximum/MT_mf_means_comparison_v_values.txt ../correction_maximum/ihMT_mf_means_comparison_v_values.txt --in_values_mf ../correction_mean/MT_mf_means_comparison_v_values.txt ../correction_mean/ihMT_mf_means_comparison_v_values.txt  --out_filename v_values_plot.png -f --value_type v;