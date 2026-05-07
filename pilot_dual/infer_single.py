#!/usr/bin/env python3
"""
Single-image inference with the h0.7_m0.95 pruned MedSAM model.
Saves 4 images to output_dir:
  gt_mask.png      — binary GT mask (white on black)
  pred_mask.png    — binary predicted mask (white on black)
  gt_overlay.png   — original image with green GT contour
  pred_overlay.png — original image with red predicted contour

Usage:
    cd /volume/med-train/users/jshan/testSAMpruning
    conda run -n medsam python pilot_dual/infer_single.py \
        --img  asserts/CVC-ColonDB/images/32.png \
        --mask asserts/CVC-ColonDB/masks/32.png \
        --output_dir results/single_infer \
        --device cuda:0
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

BASE = '/volume/med-train/users/jshan/testSAMpruning'
sys.path.insert(0, BASE)
os.chdir(BASE)


def get_bbox(mask_np, pad=2):
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        h, w = mask_np.shape
        return np.array([0., 0., w-1., h-1.])
    return np.array([
        max(0, xs.min()-pad), max(0, ys.min()-pad),
        min(mask_np.shape[1]-1, xs.max()+pad),
        min(mask_np.shape[0]-1, ys.max()+pad),
    ], dtype=float)


def overlay_contour(img_rgb, mask_np, color=(255, 50, 50), width=2):
    m = mask_np.astype(bool)
    if m.sum() == 0:
        return img_rgb.copy()
    s = np.ones((2*width+1, 2*width+1))
    contour = binary_dilation(m, s) & ~binary_erosion(m, s)
    out = img_rgb.copy()
    out[contour] = color
    return out


def make_pruned_predictor(h_sp, m_sp, device):
    from segment_anything import sam_model_registry, SamPredictor
    from pilot_dual.run_cascade_v8 import combine_scores_dist_adaptive
    from pilot_dual.pruning import (
        allocate_nonuniform_head_sparsity, allocate_nonuniform_neuron_sparsity,
        generate_head_mask_nonuniform, generate_neuron_mask_nonuniform,
        apply_head_mask_to_model, apply_mlp_mask_to_model,
    )

    model = sam_model_registry['vit_b'](checkpoint=f'{BASE}/work_dir/MedSAM/medsam_vit_b.pth')
    model.to(device).eval()

    cache_path = f'{BASE}/results/pilot_cascade_v8_sweep/_phase_b_cache/scores.npz'
    sc   = np.load(cache_path, allow_pickle=True)
    R    = len(sc['pi_r'])
    beta = float(sc['dist_beta'])
    alpha = sc['alpha_per_block']
    sens  = sc['block_sensitivity']
    pi    = sc['pi_r']

    dz_h = [sc['dz_head_per_subset'][i] for i in range(R)]
    dr_h = [sc['dr_head_per_subset'][i] for i in range(R)]
    dz_m = [sc['dz_mlp_per_subset'][i]  for i in range(R)]
    dr_m = [sc['dr_mlp_per_subset'][i]  for i in range(R)]

    head_sc = combine_scores_dist_adaptive(dz_h, dr_h, alpha, items_per_block=12,   pi=pi, beta=beta)
    mlp_sc  = combine_scores_dist_adaptive(dz_m, dr_m, alpha, items_per_block=3072, pi=pi, beta=beta)

    pb_head = allocate_nonuniform_head_sparsity(sens, h_sp, num_heads=12, min_keep=1, protected_blocks=[10,11])
    pb_mlp  = allocate_nonuniform_neuron_sparsity(sens, m_sp, mlp_dim=3072, protected_blocks=[10,11])

    head_mask   = generate_head_mask_nonuniform(head_sc, pb_head)
    neuron_mask = generate_neuron_mask_nonuniform(mlp_sc, pb_mlp)

    apply_head_mask_to_model(model, head_mask)
    apply_mlp_mask_to_model(model, neuron_mask)

    return SamPredictor(model)


def _dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    if p.sum()==0 and g.sum()==0: return 1.0
    if p.sum()==0 or  g.sum()==0: return 0.0
    return float(2*(p&g).sum()/(p.sum()+g.sum()))

def _bf1(p, g, r=2):
    def bd(m):
        m = m.astype(bool)
        if m.sum()==0: return m
        s = np.ones((2*r+1, 2*r+1))
        return binary_dilation(m,s) & ~binary_erosion(m,s)
    pb, gb = bd(p), bd(g)
    if pb.sum()==0 and gb.sum()==0: return 1.0
    if pb.sum()==0 or  gb.sum()==0: return 0.0
    prec = (pb&gb).sum()/(pb.sum()+1e-8)
    rec  = (pb&gb).sum()/(gb.sum()+1e-8)
    return float(2*prec*rec/(prec+rec+1e-8))

def _hd95_safe(p, g):
    try:
        from medpy.metric.binary import hd95 as _hd95_fn
        if p.astype(bool).sum()>0 and g.astype(bool).sum()>0:
            return float(_hd95_fn(p.astype(bool), g.astype(bool)))
    except: pass
    return 100.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img',        required=True,  help='Path to input image (jpg/png)')
    p.add_argument('--mask',       required=True,  help='Path to GT mask (jpg/png)')
    p.add_argument('--output_dir', default=f'{BASE}/results/single_infer')
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--h_sp',       type=float, default=0.7,  help='Head sparsity target')
    p.add_argument('--m_sp',       type=float, default=0.95, help='MLP sparsity target')
    p.add_argument('--contour_width', type=int, default=2)
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load inputs
    img_np = np.array(Image.open(args.img).convert('RGB'))
    gt_np  = (np.array(Image.open(args.mask).convert('L')) > 127).astype(np.uint8)

    print(f'Image : {args.img}  ({img_np.shape[1]}×{img_np.shape[0]})')
    print(f'Mask  : {args.mask}')
    print(f'Loading h{int(args.h_sp*100)}_m{int(args.m_sp*100)} pruned model...')

    predictor = make_pruned_predictor(args.h_sp, args.m_sp, device)

    # Run inference
    bbox = get_bbox(gt_np)
    predictor.set_image(img_np)
    with torch.no_grad():
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=bbox, multimask_output=False
        )
    pred_np = masks[0].astype(np.uint8)

    # Metrics
    dice = _dice(pred_np, gt_np)
    bf1  = _bf1(pred_np, gt_np)
    hd   = _hd95_safe(pred_np, gt_np)
    print(f'Metrics  →  Dice={dice:.4f}  BF1={bf1:.4f}  HD95={hd:.1f}')

    # Save 4 images
    w = args.contour_width

    gt_mask_img   = Image.fromarray((gt_np   * 255).astype(np.uint8))
    pred_mask_img = Image.fromarray((pred_np * 255).astype(np.uint8))
    gt_overlay    = Image.fromarray(overlay_contour(img_np, gt_np,   color=(50, 200, 50), width=w))
    pred_overlay  = Image.fromarray(overlay_contour(img_np, pred_np, color=(220, 50,  50), width=w))

    out = args.output_dir
    gt_mask_img.save(  os.path.join(out, 'gt_mask.png'))
    pred_mask_img.save(os.path.join(out, 'pred_mask.png'))
    gt_overlay.save(   os.path.join(out, 'gt_overlay.png'))
    pred_overlay.save( os.path.join(out, 'pred_overlay.png'))

    print(f'\nSaved to {out}/')
    print('  gt_mask.png      — GT binary mask')
    print('  pred_mask.png    — predicted binary mask')
    print('  gt_overlay.png   — original + green GT contour')
    print('  pred_overlay.png — original + red predicted contour')


if __name__ == '__main__':
    main()
