from cmcrameri import cm
import nibabel as nib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

from modules.utils import compute_peaks_fraction


def extract_measures(measures_arg, data_shape, names_arg=[]):
    measures = np.ndarray((data_shape) + (len(measures_arg),))
    measures_name = np.ndarray((len(measures_arg),), dtype=object)
    for i, measure in enumerate(measures_arg):
        measures[..., i] = (nib.load(measure)).get_fdata()
        # measures[..., i] = np.clip(measures[..., i], 0, None)
        measures_name[i] = Path(measure).name.split(".")[0]
    if names_arg != []:
        measures_name = np.array(names_arg)
    return measures, measures_name


def extract_measures_as_list(measures_arg, names_arg=[]):
    measures = []
    measures_names = []
    for i, measure in enumerate(measures_arg):
        measures.append((nib.load(measure)).get_fdata())
        measures_names.append(Path(measure).name.split(".")[0])
    if names_arg != []:
        measures_names = names_arg
    return measures, measures_names


def initialize_plot(args):
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.figsize'] = (args.figsize[0], args.figsize[1]) or (12.0, 5.0)
    plt.rcParams['font.size'] = args.font_size or 10
    plt.rcParams['axes.labelsize'] = args.axes_labelsize or args.font_size
    plt.rcParams['axes.titlesize'] = args.axes_titlesize or 1.2 * args.font_size
    plt.rcParams['legend.fontsize'] = args.legend_fontsize or args.font_size
    plt.rcParams['xtick.labelsize'] = args.xtick_labelsize or args.font_size
    plt.rcParams['ytick.labelsize'] = args.ytick_labelsize or args.font_size
    plt.rcParams['axes.linewidth'] = args.axes_linewidth or 1
    plt.rcParams['lines.linewidth'] = args.lines_linewidth or 0.5
    plt.rcParams['lines.markersize'] = args.lines_markersize or 3


def plot_init(dims=(12.0, 5.0), font_size=12):
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.color'] = "darkgrey"
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = "-"
    plt.rcParams['grid.alpha'] = "0.5"
    plt.rcParams['figure.figsize'] = dims
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.linewidth'] =1
    plt.rcParams['lines.linewidth']=1
    plt.rcParams['lines.markersize']=4


def plot_all_bundles_means(bins, means, nb_voxels, is_measures, max_count,
                           polyfits, bundles_names, measure_name, out_folder,
                           bundles_order=None):
    if bundles_order is None:
        bundles_order = bundles_names
    nb_bundles = bundles_order.shape[0]
    nb_rows = int(np.ceil(nb_bundles / 2))

    mid_bins = (bins[:-1] + bins[1:]) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    out_path = out_folder / str("all_bundles_original_" + str(measure_name) + "_1f.png")
    plot_init(dims=(8, 10), font_size=10)
    fig, ax = plt.subplots(nb_rows, 2, layout='constrained')
    for i in range(nb_bundles):
        col = i % 2
        row = i // 2
        if bundles_order[i] in bundles_names:
            bundle_idx = np.argwhere(bundles_order[i] == bundles_names).squeeze()
            is_measure = is_measures[..., bundle_idx].squeeze()
            is_not_measure = np.invert(is_measure)
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            colorbar = ax[row, col].scatter(mid_bins[is_measure],
                                            means[is_measure][..., bundle_idx],
                                            c=nb_voxels[is_measure][..., bundle_idx],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col].scatter(mid_bins[is_not_measure],
                                 means[is_not_measure][..., bundle_idx],
                                 c=nb_voxels[is_not_measure][..., bundle_idx],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            polynome = np.poly1d(polyfits[..., bundle_idx])
            ax[row, col].plot(highres_bins, polynome(highres_bins), "--",
                              color="C0")
            ax[row, col].set_ylim(0.975 * np.nanmin(means[..., bundle_idx]),
                                  1.025 * np.nanmax(means[..., bundle_idx]))
            ax[row, col].set_yticks([np.round(np.nanmin(means[..., bundle_idx]),
                                              decimals=1),
                                     np.round(np.nanmax(means[..., bundle_idx]),
                                              decimals=1)])
            ax[row, col].set_xlim(0, 90)
        else:
            ax[row, col].set_yticks([])
        ax[row, col].legend(handles=[colorbar],
                            labels=[bundles_names[bundle_idx]],
                            loc='center left', bbox_to_anchor=(1, 0.5),
                            markerscale=0, handletextpad=-2.0, handlelength=2)
        if row != nb_rows - 1:
            ax[row, col].get_xaxis().set_ticks([])
        if row == (nb_rows - 1) / 2:
            ax[row, 0].set_ylabel(str(measure_name) + ' mean')
            ax[row, 0].yaxis.set_label_coords(-0.25, 0.5)
    fig.colorbar(colorbar, ax=ax[:, 1], location='right',
                 label="Voxel count", aspect=100)
    ax[nb_rows - 1, 0].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 0].set_xlim(0, 90)
    ax[nb_rows - 1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 1].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 1].set_xlim(0, 90)
    ax[nb_rows - 1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    if nb_bundles % 2 != 0:
        ax[nb_rows - 1, 1].set_yticks([])
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_means(bins, means, nb_voxels, names, out_folder, is_measures=None):
    if is_measures is None:
        is_measures = np.ones(means.shape[0])
    is_not_measures = np.invert(is_measures)
    max_count = np.max(nb_voxels)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    mid_bins = (bins[:-1] + bins[1:]) / 2.
    plot_init(dims=(8, 3), font_size=10)
    for i in range(means.shape[-1]):
        out_path = out_folder / str(str(names[i]) + "_1f.png")
        fig, ax1 = plt.subplots(1, 1,
                                       gridspec_kw={"width_ratios":[1]},
                                       layout='constrained')
        colorbar = ax1.scatter(mid_bins[is_measures], means[is_measures][..., i],
                               c=nb_voxels[is_measures], cmap='Greys',
                               norm=norm, edgecolors=cm.naviaS(10), linewidths=1)
        ax1.scatter(mid_bins[is_not_measures], means[is_not_measures][..., i],
                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                    alpha=0.5, edgecolors=cm.naviaS(10), linewidths=1)
        ax1.set_xlabel(r'$\theta_a$')
        ax1.set_xlim(0, 90)
        min_measure = np.nanmin(means[is_measures][..., i])
        max_measure = np.nanmax(means[is_measures][..., i])
        ax1.set_ylim(0.975 * min_measure,
                     1.025 * max_measure)
        ax1.set_yticks([np.round(min_measure, decimals=1),
                        np.round(np.mean((min_measure, max_measure)), decimals=1),
                        np.round(max_measure, decimals=1)])
        ax1.set_ylabel(str(names[i]))
        fig.colorbar(colorbar, ax=ax1, label="Voxel count")
        fig.get_layout_engine().set(h_pad=0.02, hspace=0.05, wspace=0.05)
        # fig.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_3d_means(bins, means, out_folder, names):
    mid_bins = (bins[:-1] + bins[1:]) / 2.
    plot_init(dims=(10, 8), font_size=10)
    plt.rcParams['axes.labelsize'] = 12
    for i in range(means.shape[-1]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(mid_bins, mid_bins)
        ax.plot_surface(X, Y, means[..., i], cmap=cm.navia)
        ax.set_xlabel(r'$\theta_{a1}$')
        ax.set_ylabel(r'$\theta_{a2}$')
        ax.set_zlabel(str(names[i]))
        fig.tight_layout()
        views = np.array([[45, -135], [45, 45]])
        for v, view in enumerate(views[:]):
            out_path = out_folder / str(str(names[i]) + "_3D_2f_view_" + str(v + 1) + ".png")
            ax.view_init(view[0], view[1])
            plt.savefig(out_path, dpi=300)
        plt.close()


def plot_multiple_means(bins, means, nb_voxels, out_folder, names,
                        endname="2f", labels=None, color_start=2,
                        legend_title=None, markers="o", is_measures=None):
    if is_measures is None:
        is_measures = np.ones((means.shape[0], means.shape[1])).astype(bool)
        is_not_measures = np.zeros((means.shape[0], means.shape[1])).astype(bool)
    else:
        is_not_measures = np.invert(is_measures)

    max_count = np.max(nb_voxels)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    mid_bins = (bins[:-1] + bins[1:]) / 2.

    for j in range(means.shape[-1]):
        plot_init()
        fig, (ax1, cax) = plt.subplots(1, 2,
                                       gridspec_kw={"width_ratios":[1, 0.05]})
        out_path = out_folder / str(str(names[j]) + "_" + str(endname) + ".png")
        for i in range(means.shape[0]):
            if labels is not None:
                colorbar = ax1.scatter(mid_bins[is_measures[i]],
                                       means[i, is_measures[i], j],
                                       c=nb_voxels[i, is_measures[i]],
                                       cmap='Greys', norm=norm,
                                       label=labels[i], linewidths=1,
                                       edgecolors=cm.naviaS(i+color_start),
                                       marker=markers)
                ax1.scatter(mid_bins[is_not_measures[i]],
                            means[i, is_not_measures[i], j],
                            c=nb_voxels[i, is_not_measures[i]],
                            cmap='Greys', norm=norm,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i+color_start),
                            marker=markers)
            else:
                colorbar = ax1.scatter(mid_bins[is_measures[i]],
                                       means[i, is_measures[i], j],
                                       c=nb_voxels[i, is_measures[i]],
                                       cmap='Greys', norm=norm,
                                       linewidths=1,
                                       edgecolors=cm.naviaS(i+color_start),
                                       marker=markers)
                ax1.scatter(mid_bins[is_not_measures[i]],
                            means[i, is_not_measures[i], j],
                            c=nb_voxels[i, is_not_measures[i]],
                            cmap='Greys', norm=norm,
                            linewidths=1, edgecolors=cm.naviaS(i+color_start),
                            marker=markers, alpha=0.5)
        ax1.set_xlabel(r'$\theta_a$')
        ax1.set_xlim(0, 90)
        ax1.set_ylabel(str(names[j]))
        if labels is not None:
            ax1.legend()
        if legend_title is not None:
            ax1.get_legend().set_title(legend_title)

        fig.colorbar(colorbar, cax=cax, label="Voxel count")
        fig.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()



def save_angle_maps(peaks, fa, wm_mask, affine, output_path, fodf_peaks,
                    peak_values, nufo, bin_width=1, fa_thr=0.5):
    # Find the direction of the B0 field
    rot = affine[0:3, 0:3]
    z_axis = np.array([0, 0, 1])
    b0_field = np.dot(rot.T, z_axis)

    # Define the bins
    bins = np.arange(0, 90 + bin_width, bin_width)

    # Calculate the angle between e1 and B0 field
    cos_theta = np.dot(peaks[..., :3], b0_field)
    theta = np.arccos(cos_theta) * 180 / np.pi

    peaks_fraction = compute_peaks_fraction(peak_values)

    cos_theta_f1 = np.dot(fodf_peaks[..., 0:3], b0_field)
    theta_f1 = np.arccos(cos_theta_f1) * 180 / np.pi
    cos_theta_f2 = np.dot(fodf_peaks[..., 3:6], b0_field)
    theta_f2 = np.arccos(cos_theta_f2) * 180 / np.pi
    cos_theta_f3 = np.dot(fodf_peaks[..., 6:9], b0_field)
    theta_f3 = np.arccos(cos_theta_f3) * 180 / np.pi

    peak_1 = np.zeros(wm_mask.shape)
    peak_2 = np.zeros(wm_mask.shape)
    peak_3 = np.zeros(wm_mask.shape)

    # Apply the WM mask and FA threshold
    wm_mask_bool = (wm_mask > 0.9) & (fa > fa_thr) & (nufo == 1)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta >= bins[i]) & (theta < bins[i+1]) 
        angle_mask_90_180 = (180 - theta >= bins[i]) & (180 - theta < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask = wm_mask_bool & angle_mask
        peak_1[mask] = (bins[i] + bins[i + 1]) /2.
    
    peak_1_sf = np.copy(peak_1)

    wm_mask_bool = (wm_mask > 0.9) & (nufo == 2)
    fraction_mask_bool = (peaks_fraction[..., 0] >= 0.5) & (peaks_fraction[..., 0] < 0.9)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta_f1 >= bins[i]) & (theta_f1 < bins[i+1])
        angle_mask_90_180 = (180 - theta_f1 >= bins[i]) & (180 - theta_f1 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f1 = angle_mask & fraction_mask_bool & wm_mask_bool
        peak_1[mask_f1] = (bins[i] + bins[i + 1]) /2.

        angle_mask_0_90 = (theta_f2 >= bins[i]) & (theta_f2 < bins[i+1]) 
        angle_mask_90_180 = (180 - theta_f2 >= bins[i]) & (180 - theta_f2 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f2 = angle_mask & fraction_mask_bool & wm_mask_bool
        peak_2[mask_f2] = (bins[i] + bins[i + 1]) /2.

    wm_mask_bool = (wm_mask > 0.9) & (nufo == 3)
    fraction_mask_bool = (peaks_fraction[..., 0] >= 0.33) & (peaks_fraction[..., 0] < 0.8)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta_f1 >= bins[i]) & (theta_f1 < bins[i+1])
        angle_mask_90_180 = (180 - theta_f1 >= bins[i]) & (180 - theta_f1 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f1 = angle_mask & wm_mask_bool & fraction_mask_bool
        peak_1[mask_f1] = (bins[i] + bins[i + 1]) /2.

        angle_mask_0_90 = (theta_f2 >= bins[i]) & (theta_f2 < bins[i+1]) 
        angle_mask_90_180 = (180 - theta_f2 >= bins[i]) & (180 - theta_f2 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f2 = angle_mask & wm_mask_bool & fraction_mask_bool
        peak_2[mask_f2] = (bins[i] + bins[i + 1]) /2.

        angle_mask_0_90 = (theta_f3 >= bins[i]) & (theta_f3 < bins[i+1]) 
        angle_mask_90_180 = (180 - theta_f3 >= bins[i]) & (180 - theta_f3 < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask_f3 = angle_mask & wm_mask_bool & fraction_mask_bool
        peak_3[mask_f3] = (bins[i] + bins[i + 1]) /2.

    map_1_name = "peak_1_1f_angles_map.nii.gz"
    map_1_path = output_path / map_1_name
    nib.save(nib.Nifti1Image(peak_1_sf, affine), map_1_path)

    map_1_name = "peak_1_angles_map.nii.gz"
    map_1_path = output_path / map_1_name
    nib.save(nib.Nifti1Image(peak_1, affine), map_1_path)

    map_2_name = "peak_2_angles_map.nii.gz"
    map_2_path = output_path / map_2_name
    nib.save(nib.Nifti1Image(peak_2, affine), map_2_path)

    map_3_name = "peak_3_angles_map.nii.gz"
    map_3_path = output_path / map_3_name
    nib.save(nib.Nifti1Image(peak_3, affine), map_3_path)


def save_masks_by_angle_bins(peaks, fa, wm_mask, affine, output_path,
                             nufo=None, bin_width=10, fa_thr=0.5):
    # Find the direction of the B0 field
    rot = affine[0:3, 0:3]
    z_axis = np.array([0, 0, 1])
    b0_field = np.dot(rot.T, z_axis)

    # Define the bins
    bins = np.arange(0, 90 + bin_width, bin_width)

    # Calculate the angle between e1 and B0 field
    cos_theta = np.dot(peaks[..., :3], b0_field)
    theta = np.arccos(cos_theta) * 180 / np.pi

    # Apply the WM mask and FA threshold
    if nufo is not None:
        wm_mask_bool = (wm_mask >= 0.9) & (fa > fa_thr) & (nufo == 1)
    else:
        wm_mask_bool = (wm_mask >= 0.9) & (fa > fa_thr)
    for i in range(len(bins) - 1):
        angle_mask_0_90 = (theta >= bins[i]) & (theta < bins[i+1]) 
        angle_mask_90_180 = (180 - theta >= bins[i]) & (180 - theta < bins[i+1])
        angle_mask = angle_mask_0_90 | angle_mask_90_180
        mask = wm_mask_bool & angle_mask
        mask_name = "1f_mask_" + str(bins[i]) + "_to_" + str(bins[i+1]) \
            + "_degrees.nii.gz"
        mask_path = output_path / mask_name
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)


def save_results_as_npz(bins, measure_means, measure_stds, nb_voxels,
                        pts_origin, names, out_path):
    # Save the results to a npz files
    savez_dict = dict()
    savez_dict['Angle_min'] = bins[:-1]
    savez_dict['Angle_max'] = bins[1:]
    for i in range(measure_means.shape[-1]):
        savez_dict[str(names[i])] = measure_means[..., i]
        savez_dict[str(names[i]) + '_std'] = measure_stds[..., i]
        savez_dict['Origin_' + str(names[i])] = pts_origin[..., i].astype(str)
        savez_dict['Nb_voxels_' + str(names[i])] = nb_voxels[..., i]
    np.savez(str(out_path), **savez_dict)


def save_results_as_npz_mean(bins, measure_means, measure_stds, nb_voxels,
                             nb_voxels_stds, pts_origin, names, out_path):
    # Save the results to a npz files
    savez_dict = dict()
    savez_dict['Angle_min'] = bins[:-1]
    savez_dict['Angle_max'] = bins[1:]
    for i in range(measure_means.shape[-1]):
        savez_dict[str(names[i])] = measure_means[i]
        savez_dict[str(names[i]) + '_std'] = measure_stds[i]
        savez_dict['Origin_' + str(names[i])] = pts_origin[i].astype(str)
        savez_dict['Nb_voxels_' + str(names[i])] = nb_voxels[i]
        savez_dict['Nb_voxels_std_' + str(names[i])] = nb_voxels_stds[i]
    np.savez(str(out_path), **savez_dict)


def save_profiles_as_npz(measure_means, measure_stds, nb_voxels, names,
                         out_path):
    # Save the results to a npz files
    savez_dict = dict()
    for i in range(measure_means.shape[-2]):
        savez_dict[str(names[i]) + '_af'] = measure_means[..., i, 0]
        savez_dict[str(names[i]) + '_std_af'] = measure_stds[..., i, 0]
        savez_dict[str(names[i]) + '_voxel_count_af'] = nb_voxels[..., i, 0]
        savez_dict[str(names[i]) + '_sf'] = measure_means[..., i, 1]
        savez_dict[str(names[i]) + '_std_sf'] = measure_stds[..., i, 1]
        savez_dict[str(names[i]) + '_voxel_count_sf'] = nb_voxels[..., i, 1]
        savez_dict[str(names[i]) + '_mf'] = measure_means[..., i, 2]
        savez_dict[str(names[i]) + '_std_mf'] = measure_stds[..., i, 2]
        savez_dict[str(names[i]) + '_voxel_count_mf'] = nb_voxels[..., i, 2]
    np.savez(str(out_path), **savez_dict)


def save_profiles_mean_as_npz(measure_means, measure_stds, nb_voxels, names,
                              out_path):
    # Save the results to a npz files
    savez_dict = dict()
    for i in range(measure_means.shape[-2]):
        savez_dict[str(names[i]) + '_af'] = measure_means[i, 0]
        savez_dict[str(names[i]) + '_std_af'] = measure_stds[i, 0]
        savez_dict[str(names[i]) + '_voxel_count_af'] = nb_voxels[i, 0]
        savez_dict[str(names[i]) + '_sf'] = measure_means[i, 1]
        savez_dict[str(names[i]) + '_std_sf'] = measure_stds[i, 1]
        savez_dict[str(names[i]) + '_voxel_count_sf'] = nb_voxels[i, 1]
        savez_dict[str(names[i]) + '_mf'] = measure_means[i, 2]
        savez_dict[str(names[i]) + '_std_mf'] = measure_stds[i, 2]
        savez_dict[str(names[i]) + '_voxel_count_mf'] = nb_voxels[i, 2]
    np.savez(str(out_path), **savez_dict)


def save_polyfits_as_npz(polyfits, measures_ref, names, out_path):
    savez_dict = dict()
    for i in range(polyfits.shape[-1]):
        savez_dict[str(names[i]) + "_polyfit"] = polyfits[..., i]
        savez_dict[str(names[i]) + "_reference"] = measures_ref[i]
    np.savez(str(out_path), **savez_dict)


def save_polyfits_as_npz_mean(polyfits, measures_ref, names, out_path):
    savez_dict = dict()
    for i in range(polyfits.shape[-1]):
        savez_dict[str(names[i]) + "_polyfit"] = polyfits[i]
        savez_dict[str(names[i]) + "_reference"] = measures_ref[i]
    np.savez(str(out_path), **savez_dict)