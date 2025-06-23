import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import plot_init, extract_measures


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_image',
                   help='Path of the output image.')
    
    p.add_argument('--maps_original', nargs='+', required=True,
                   help='List of images to visualize.')
    
    p.add_argument('--maps_corrected', nargs='+', required=True,
                   help='List of images to visualize.')
    
    p.add_argument('--reference', default=[], required=True,
                   help='Reference image.')
    
    p.add_argument('--wm_mask', default=[],
                   help='WM mask image.')
    
    p.add_argument('--slices', nargs=3, default=[105, 125, 75],
                   action='append',
                   help='List indices for where to slice the images.')
    
    p.add_argument('--combine_colorbar', action='store_false',
                   help='Combine colorbar or not.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ref = nib.load(args.reference).get_fdata()

    data_shape = ref.shape

    maps_original, _ = extract_measures(args.maps_original, data_shape)
    maps_original = np.ma.masked_where(maps_original == 0, maps_original)

    maps_corrected, _ = extract_measures(args.maps_corrected, data_shape)
    maps_corrected = np.ma.masked_where(maps_corrected == 0, maps_corrected)

    maps = (maps_corrected - maps_original) / maps_original * 100

    if args.wm_mask:
        mask = nib.load(args.wm_mask).get_fdata()
        mask = (mask >= 0.9)
    else:
        mask = np.ones((data_shape))

    x_index = int(args.slices[0]) # 53, 54 or 55 (55)
    y_index = int(args.slices[1]) # 92, 93, or 94 (93)
    z_index = int(args.slices[2]) # 53 (53)

    # vmax = np.array([90, 90, 90, 1, 1])
    vmax = [100, 100, 100, 100]
    # vmax = [np.max(maps[..., 0]), np.max(maps[..., 1]), np.max(maps[..., 2]), np.max(maps[..., 3])]

    plot_init(dims=(10, 15), font_size=20)

    COLOR = 'white'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR

    if args.combine_colorbar:
        fig, ax = plt.subplots(maps.shape[-1], 3,
                               gridspec_kw={"width_ratios":[1.0, 1.1, 0.8]},
                               layout='constrained')
    else:
        fig, ax = plt.subplots(maps.shape[-1], 4,
                               gridspec_kw={"width_ratios":[1.1, 1.3, 1.0, 0.05]},
                               layout='constrained')

    # for i in range(maps.shape[-1]):
    for i in range(maps.shape[-1]):
        # x_image = np.flip(np.rot90(ref[x_index, :, :]), axis=1)
        # y_image = np.rot90(ref[:, y_index, :])
        # z_image = np.rot90(ref[:, :, z_index])

        # colorbar = ax[i , 0].imshow(y_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        # ax[i, 1].imshow(x_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        # ax[i, 2].imshow(z_image, cmap="gray", vmin=0, vmax=1, interpolation='none')

        map = maps[..., i] * mask
        map = np.ma.masked_where(map == 0, map)

        x_mask = np.flip(np.rot90(map[x_index, :, :]), axis=1)
        y_mask = np.rot90(map[:, y_index, :])
        z_mask = np.rot90(map[:, :, z_index])

        colorbar = ax[i, 0].imshow(y_mask, cmap=cm.navia, vmin=0, vmax=vmax[i], interpolation='none')
        ax[i, 1].imshow(x_mask, cmap=cm.navia, vmin=0, vmax=vmax[i], interpolation='none')
        ax[i, 2].imshow(z_mask, cmap=cm.navia, vmin=0, vmax=vmax[i], interpolation='none')

        if args.combine_colorbar:
            if i == 1:
                cb = fig.colorbar(colorbar, ax=ax[0:4, 2], location='right', aspect=40, pad=0.1)
                cb.outline.set_color('white')
            # if i == 4:
            #     cb = fig.colorbar(colorbar, ax=ax[3:, 2], location='right', aspect=33, pad=0.1)
            #     cb.outline.set_color('white')
        else:
            cb = fig.colorbar(colorbar, ax=ax[i, 3])
            cb.outline.set_color('white')

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_axis_off()
            ax[i, j].autoscale(False)

    fig.get_layout_engine().set(h_pad=0.1, hspace=0.1) #, w_pad=0, wspace=0)
    # fig.tight_layout()
    # plt.show()
    plt.savefig(args.out_image, dpi=300, transparent=True)


if __name__ == "__main__":
    main()
