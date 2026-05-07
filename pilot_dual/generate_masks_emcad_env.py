#!/usr/bin/env python3
"""
Generate binary prediction masks for EMCAD and MK-UNet on selected polyp samples.
Runs in emcadenv conda env.

Usage:
    cd /volume/med-train/users/jshan/testSAMpruning
    conda run -n emcadenv python pilot_dual/generate_masks_emcad_env.py \
        --output_dir results/viz_comparison \
        --device cuda:0
"""
import argparse, os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from medpy.metric.binary import hd95 as _hd95_fn

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
                        'mask':f'{BASE}/asserts/kvasir-seg/Kvasir-SEG/masks/{stem}.jpg'})
    for stem in c:
        samples.append({'dataset':'ClinicDB', 'stem':stem,
                        'img': f'{BASE}/asserts/CVC-ClinicDB/images/{stem}.png',
                        'mask':f'{BASE}/asserts/CVC-ClinicDB/masks/{stem}.png'})
    for stem in d:
        samples.append({'dataset':'ColonDB',  'stem':stem,
                        'img': f'{BASE}/asserts/CVC-ColonDB/images/{stem}.png',
                        'mask':f'{BASE}/asserts/CVC-ColonDB/masks/{stem}.png'})
    return samples


def _dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    if p.sum()==0 and g.sum()==0: return 1.0
    if p.sum()==0 or  g.sum()==0: return 0.0
    return float(2*(p&g).sum()/(p.sum()+g.sum()))

def _iou(p, g):
    p, g = p.astype(bool), g.astype(bool)
    u = (p|g).sum()
    return 1.0 if u==0 else float((p&g).sum()/u)

def _bf1(p, g, r=2):
    def bd(m):
        m = m.astype(bool)
        if m.sum()==0: return m
        s = np.ones((2*r+1,2*r+1))
        return binary_dilation(m,s) & ~binary_erosion(m,s)
    pb, gb = bd(p), bd(g)
    if pb.sum()==0 and gb.sum()==0: return 1.0
    if pb.sum()==0 or  gb.sum()==0: return 0.0
    prec=(pb&gb).sum()/(pb.sum()+1e-8); rec=(pb&gb).sum()/(gb.sum()+1e-8)
    return float(2*prec*rec/(prec+rec+1e-8))

def _hd95_safe(p, g):
    try:
        if p.astype(bool).sum()>0 and g.astype(bool).sum()>0:
            return float(_hd95_fn(p.astype(bool), g.astype(bool)))
    except: pass
    return 100.0

def compute_metrics(pred, gt):
    return {'Dice': round(_dice(pred,gt),4), 'IoU': round(_iou(pred,gt),4),
            'BF1':  round(_bf1(pred,gt),4),  'HD95': round(_hd95_safe(pred,gt),2)}

def save_mask(mask_np, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask_np*255).astype(np.uint8)).save(path)

def preprocess_cnn(img_np, img_size):
    img_r = np.array(Image.fromarray(img_np).resize((img_size, img_size), Image.BILINEAR))
    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    img_f = (img_r.astype(np.float32)/255.0 - mean) / std
    return torch.from_numpy(img_f.transpose(2,0,1)).float().unsqueeze(0)


def run_emcad(device, output_dir, samples):
    sys.path.insert(0, f'{BASE}/contrast_exp/EMCAD')
    from lib.networks import EMCADNet

    model = EMCADNet(
        num_classes=1, kernel_sizes=[1,3,5,7], expansion_factor=2,
        dw_parallel=True, add=True, lgag_ks=3, activation='relu',
        encoder='pvt_v2_b2', pretrain=False
    ).to(device)
    model.load_state_dict(
        torch.load(f'{BASE}/contrast_exp/EMCAD/model_pth/emcad_polyp/emcad_polyp-best.pth',
                   map_location=device, weights_only=False),
        strict=False)
    model.eval()
    img_size = 352
    metrics = {}

    with torch.no_grad():
        for s in samples:
            img_np = np.array(Image.open(s['img']).convert('RGB'))
            gt_np  = (np.array(Image.open(s['mask']).convert('L')) > 127).astype(np.uint8)
            oh, ow = img_np.shape[:2]

            img_t = preprocess_cnn(img_np, img_size).to(device)
            ress  = model(img_t)
            if not isinstance(ress, list): ress = [ress]
            pred_logit = ress[-1]   # [1,1,H,W]
            pred_full  = F.interpolate(pred_logit, size=(oh,ow), mode='bilinear', align_corners=False)
            pred_sig   = pred_full.sigmoid().squeeze().cpu().numpy()
            pred_sig   = (pred_sig - pred_sig.min()) / max(pred_sig.max()-pred_sig.min(), 1e-8)
            pred_np    = (pred_sig >= 0.5).astype(np.uint8)

            mask_path = os.path.join(output_dir,'masks','emcad',s['dataset'],f'{s["stem"]}.png')
            save_mask(pred_np, mask_path)
            m = compute_metrics(pred_np, gt_np)
            metrics[f'{s["dataset"]}/{s["stem"]}'] = m
            print(f'  EMCAD {s["dataset"]}/{s["stem"]}: {m}')

    return metrics


def run_mkunet(device, output_dir, samples):
    sys.path.insert(0, f'{BASE}/contrast_exp/MK-UNet')
    from mkunet_network import MK_UNet

    model = MK_UNet(num_classes=1, input_channels=3,
                    channels=[16,32,64,96,160]).to(device)
    model.load_state_dict(
        torch.load(f'{BASE}/contrast_exp/MK-UNet/model_pth/mkunet_polyp/mkunet_polyp-best.pth',
                   map_location=device, weights_only=False),
        strict=False)
    model.eval()
    img_size = 352
    metrics = {}

    with torch.no_grad():
        for s in samples:
            img_np = np.array(Image.open(s['img']).convert('RGB'))
            gt_np  = (np.array(Image.open(s['mask']).convert('L')) > 127).astype(np.uint8)
            oh, ow = img_np.shape[:2]

            img_t = preprocess_cnn(img_np, img_size).to(device)
            ress  = model(img_t)
            pred_logit = ress[0] if isinstance(ress, list) else ress
            pred_full  = F.interpolate(pred_logit, size=(oh,ow), mode='bilinear', align_corners=False)
            pred_sig   = pred_full.sigmoid().squeeze().cpu().numpy()
            pred_sig   = (pred_sig - pred_sig.min()) / max(pred_sig.max()-pred_sig.min(), 1e-8)
            pred_np    = (pred_sig >= 0.5).astype(np.uint8)

            mask_path = os.path.join(output_dir,'masks','mkunet',s['dataset'],f'{s["stem"]}.png')
            save_mask(pred_np, mask_path)
            m = compute_metrics(pred_np, gt_np)
            metrics[f'{s["dataset"]}/{s["stem"]}'] = m
            print(f'  MK-UNet {s["dataset"]}/{s["stem"]}: {m}')

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default=f'{BASE}/results/viz_comparison')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--n_samples', type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    samples = build_samples(seed=args.seed, n=args.n_samples)
    print(f'Seed={args.seed}, {len(samples)} samples: '
          + ', '.join(f'{s["dataset"]}/{s["stem"]}' for s in samples))

    all_metrics = {}
    print('\n=== EMCAD ===')
    all_metrics['emcad'] = run_emcad(device, args.output_dir, samples)
    torch.cuda.empty_cache()

    print('\n=== MK-UNet ===')
    all_metrics['mkunet'] = run_mkunet(device, args.output_dir, samples)

    out_path = os.path.join(args.output_dir, 'metrics_emcad_env.json')
    with open(out_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\nMetrics saved → {out_path}')


if __name__ == '__main__':
    main()
