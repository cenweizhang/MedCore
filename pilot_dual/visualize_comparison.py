#!/usr/bin/env python3
"""
Qualitative comparison visualization.

Layout: rows = 3 samples (Kvasir, ClinicDB, ColonDB)
        cols = Original | MedSAM | Ours-h70m95 | Ours-h80m95 |
               EfficientSAM-Ti | SlimSAM | SAMed | QMedSAM |
               EMCAD | MK-UNet | Swin-Unet | nnUNet | GT

Usage:
    cd /volume/med-train/users/jshan/testSAMpruning
    conda run -n medsam python pilot_dual/visualize_comparison.py \
        --mask_dir  results/viz_comparison/masks \
        --output    results/viz_comparison/comparison_1200.png \
        --dpi 200
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

BASE = '/volume/med-train/users/jshan/testSAMpruning'
NNUNET_PRED_DIR = f'{BASE}/nnunet_data/results/Dataset001_Polyp/predictions'

def build_samples(seed=42, n=5):
    pngs = [f for f in os.listdir(NNUNET_PRED_DIR) if f.endswith('.png')]
    kvasir  = sorted([f.replace('kvasir_','').replace('.png','')   for f in pngs if f.startswith('kvasir_')])
    clinic  = sorted([f.replace('clinicdb_','').replace('.png','') for f in pngs if f.startswith('clinicdb_')])
    colondb = sorted([f.replace('colondb_','').replace('.png','')  for f in pngs if f.startswith('colondb_')])
    rng = np.random.default_rng(seed)
    k = sorted(rng.choice(kvasir,  n, replace=False).tolist())
    c = sorted(rng.choice(clinic,  n, replace=False).tolist())
    d = sorted(rng.choice(colondb, n, replace=False).tolist())
    samples = []
    for stem in k:
        samples.append({'dataset':'Kvasir',  'stem':stem,
                        'img': f'{BASE}/asserts/kvasir-seg/Kvasir-SEG/images/{stem}.jpg',
                        'gt':  f'{BASE}/asserts/kvasir-seg/Kvasir-SEG/masks/{stem}.jpg'})
    for stem in c:
        samples.append({'dataset':'ClinicDB', 'stem':stem,
                        'img': f'{BASE}/asserts/CVC-ClinicDB/images/{stem}.png',
                        'gt':  f'{BASE}/asserts/CVC-ClinicDB/masks/{stem}.png'})
    for stem in d:
        samples.append({'dataset':'ColonDB',  'stem':stem,
                        'img': f'{BASE}/asserts/CVC-ColonDB/images/{stem}.png',
                        'gt':  f'{BASE}/asserts/CVC-ColonDB/masks/{stem}.png'})
    return samples

# Column definitions: (display_name, mask_dir_name, is_ours)
COLUMNS = [
    ('Original',        None,            False),
    ('MedSAM',          'medsam',        False),
    ('Ours\nh70_m95',   'h70_m95',       True),
    ('Ours\nh80_m95',   'h80_m95',       True),
    ('EfficientSAM-Ti', 'efficientSAM_ti', False),
    ('SlimSAM-50',      'slimsam',       False),
    ('SAMed',           'samed',         False),
    ('QMedSAM',         'qmedsam',       False),
    ('EMCAD',           'emcad',         False),
    ('MK-UNet',         'mkunet',        False),
    ('Swin-Unet',       'swinunet',      False),
    ('nnUNet',          'nnunet',        False),
    ('GT',              None,            False),
]


def load_metrics(mask_dir):
    """Load all metrics JSON files from viz_comparison directory."""
    metrics = {}
    viz_dir = os.path.dirname(mask_dir)
    for fname in ['metrics_medsam_env.json', 'metrics_emcad_env.json',
                  'metrics_swinunet_nnunet.json']:
        fpath = os.path.join(viz_dir, fname)
        if not os.path.exists(fpath):
            continue
        d = json.load(open(fpath))
        for model_name, model_data in d.items():
            metrics[model_name] = model_data
    return metrics


def get_mask(mask_dir, model_dir, dataset, stem):
    path = os.path.join(mask_dir, model_dir, dataset, f'{stem}.png')
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path).convert('L'))
    return (arr > 127).astype(np.uint8)


def mask_to_display(mask_np):
    """Binary mask → white-on-black RGB."""
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    rgb[mask_np > 0] = 255
    return rgb


def overlay_contour(img_rgb, mask_np, color=(255, 50, 50), width=2):
    """Draw mask contour on image copy."""
    from scipy.ndimage import binary_dilation, binary_erosion
    m = mask_np.astype(bool)
    if m.sum() == 0:
        return img_rgb.copy()
    s = np.ones((2*width+1, 2*width+1))
    contour = binary_dilation(m, s) & ~binary_erosion(m, s)
    out = img_rgb.copy()
    out[contour] = color
    return out


def format_metrics(m):
    if m is None:
        return ''
    return (f'Dice={m["Dice"]:.3f}\n'
            f'BF1={m["BF1"]:.3f}\n'
            f'HD95={m["HD95"]:.1f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mask_dir', default=f'{BASE}/results/viz_comparison/masks')
    p.add_argument('--output',   default=f'{BASE}/results/viz_comparison/comparison_1200.png')
    p.add_argument('--dpi',      type=int, default=200)
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--show_metrics', action='store_true', default=True)
    args = p.parse_args()

    SAMPLES = build_samples(seed=args.seed, n=args.n_samples)
    metrics = load_metrics(args.mask_dir)

    n_rows = len(SAMPLES)
    n_cols = len(COLUMNS)

    # Cell size in inches
    cell_w = 1.05
    cell_h = 1.05
    title_h = 0.45   # top header row
    label_w = 0.35   # left dataset label

    fig_w = label_w + n_cols * cell_w
    fig_h = title_h + n_rows * (cell_h + 0.35)  # extra for metric text

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor('white')

    # Column headers
    for ci, (col_name, _, is_ours) in enumerate(COLUMNS):
        x = (label_w + ci * cell_w + cell_w/2) / fig_w
        y = 1.0 - title_h / (2 * fig_h)
        color = '#c0392b' if is_ours else '#2c3e50'
        weight = 'bold' if is_ours else 'normal'
        fig.text(x, 1 - title_h/(2*fig_h), col_name,
                 ha='center', va='center', fontsize=5.5,
                 color=color, fontweight=weight,
                 transform=fig.transFigure)

    for ri, sample in enumerate(SAMPLES):
        ds, stem = sample['dataset'], sample['stem']
        img_np = np.array(Image.open(sample['img']).convert('RGB'))
        gt_np  = (np.array(Image.open(sample['gt']).convert('L')) > 127).astype(np.uint8)

        # Dataset label on left: only draw on the middle row of each 5-row group
        if ri % 5 == 2:
            group_top = ri - 2
            y_center = 1 - (title_h + (group_top + 2.5) * (cell_h + 0.35)) / fig_h
            fig.text(label_w / (2 * fig_w), y_center, ds,
                     ha='center', va='center', fontsize=7, rotation=90,
                     fontweight='bold', color='#2c3e50',
                     transform=fig.transFigure)

        for ci, (col_name, model_dir, is_ours) in enumerate(COLUMNS):
            # Compute axes position
            ax_left   = (label_w + ci * cell_w) / fig_w
            ax_bottom = 1 - (title_h + (ri + 1) * cell_h + ri * 0.35) / fig_h
            ax_w      = cell_w / fig_w
            ax_h      = cell_h / fig_h
            ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
            ax.axis('off')

            if col_name == 'Original':
                ax.imshow(img_np)
                # thin border
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_linewidth(0.5); spine.set_color('#7f8c8d')
            elif col_name == 'GT':
                gt_display = overlay_contour(img_np, gt_np, color=(50,200,50), width=2)
                ax.imshow(gt_display)
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color('#27ae60')
            else:
                mask = get_mask(args.mask_dir, model_dir, ds, stem)
                if mask is not None:
                    display = overlay_contour(img_np, mask,
                                              color=(220,50,50) if is_ours else (50,100,220),
                                              width=2)
                    ax.imshow(display)
                    # border highlight for our models
                    lw = 1.5 if is_ours else 0.5
                    ec = '#c0392b' if is_ours else '#7f8c8d'
                    for spine in ax.spines.values():
                        spine.set_visible(True); spine.set_linewidth(lw); spine.set_color(ec)

                    # metric text below cell
                    if args.show_metrics:
                        key = f'{ds}/{stem}'
                        m = metrics.get(model_dir, {}).get(key)
                        if m:
                            txt = f'D={m["Dice"]:.3f}  B={m["BF1"]:.3f}\nHD={m["HD95"]:.1f}'
                            fig.text(ax_left + ax_w/2,
                                     ax_bottom - 0.015/fig_h,
                                     txt,
                                     ha='center', va='top',
                                     fontsize=3.5,
                                     color='#c0392b' if is_ours else '#555555',
                                     fontweight='bold' if is_ours else 'normal',
                                     transform=fig.transFigure)
                else:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            fontsize=6, color='#aaaaaa', transform=ax.transAxes)

    # Draw horizontal separator lines between dataset groups (after rows 4 and 9)
    row_h = cell_h + 0.35
    for sep_after in [4, 9]:  # after Kvasir block, after ClinicDB block
        y_sep = 1 - (title_h + (sep_after + 1) * row_h - 0.08) / fig_h
        line = plt.Line2D([label_w / fig_w, 1.0], [y_sep, y_sep],
                          transform=fig.transFigure, color='#aaaaaa',
                          linewidth=0.8, linestyle='--')
        fig.add_artist(line)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    # Also save 1200px wide version
    img_out = Image.open(args.output)
    w_orig, h_orig = img_out.size
    target_w = 1200
    if w_orig != target_w:
        scale = target_w / w_orig
        new_h = int(h_orig * scale)
        img_out = img_out.resize((target_w, new_h), Image.LANCZOS)
        out_1200 = args.output.replace('.png', '_1200px.png')
        img_out.save(out_1200, dpi=(300, 300))
        print(f'Saved 1200px → {out_1200}')

    print(f'Saved → {args.output}  ({img_out.size[0]}×{img_out.size[1]})')


if __name__ == '__main__':
    main()
