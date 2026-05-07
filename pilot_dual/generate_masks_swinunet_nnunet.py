#!/usr/bin/env python3
"""
Generate binary prediction masks for Swin-Unet and copy nnUNet predictions.
Can run in medsam env (all deps available: torch, timm, ml-collections, scipy).

Usage:
    cd /volume/med-train/users/jshan/testSAMpruning
    conda run -n medsam python pilot_dual/generate_masks_swinunet_nnunet.py \
        --output_dir results/viz_comparison \
        --device cuda:0
"""
import argparse, os, sys, json, shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from medpy.metric.binary import hd95 as _hd95_fn

BASE        = '/volume/med-train/users/jshan/testSAMpruning'
SWINUNET_DIR = f'{BASE}/contrast_exp/Swin-Unet'
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
                        'img':  f'{BASE}/asserts/kvasir-seg/Kvasir-SEG/images/{stem}.jpg',
                        'mask': f'{BASE}/asserts/kvasir-seg/Kvasir-SEG/masks/{stem}.jpg',
                        'nnunet_pred': f'{NNUNET_PRED_DIR}/kvasir_{stem}.png'})
    for stem in c:
        samples.append({'dataset':'ClinicDB', 'stem':stem,
                        'img':  f'{BASE}/asserts/CVC-ClinicDB/images/{stem}.png',
                        'mask': f'{BASE}/asserts/CVC-ClinicDB/masks/{stem}.png',
                        'nnunet_pred': f'{NNUNET_PRED_DIR}/clinicdb_{stem}.png'})
    for stem in d:
        samples.append({'dataset':'ColonDB',  'stem':stem,
                        'img':  f'{BASE}/asserts/CVC-ColonDB/images/{stem}.png',
                        'mask': f'{BASE}/asserts/CVC-ColonDB/masks/{stem}.png',
                        'nnunet_pred': f'{NNUNET_PRED_DIR}/colondb_{stem}.png'})
    return samples


def _dice(p, g):
    p,g = p.astype(bool),g.astype(bool)
    if p.sum()==0 and g.sum()==0: return 1.0
    if p.sum()==0 or  g.sum()==0: return 0.0
    return float(2*(p&g).sum()/(p.sum()+g.sum()))

def _iou(p, g):
    p,g=p.astype(bool),g.astype(bool)
    u=(p|g).sum(); return 1.0 if u==0 else float((p&g).sum()/u)

def _bf1(p, g, r=2):
    def bd(m):
        m=m.astype(bool)
        if m.sum()==0: return m
        s=np.ones((2*r+1,2*r+1))
        return binary_dilation(m,s)&~binary_erosion(m,s)
    pb,gb=bd(p),bd(g)
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
    return {'Dice':round(_dice(pred,gt),4), 'IoU':round(_iou(pred,gt),4),
            'BF1': round(_bf1(pred,gt),4),  'HD95':round(_hd95_safe(pred,gt),2)}

def save_mask(mask_np, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask_np*255).astype(np.uint8)).save(path)


# ── Swin-Unet ────────────────────────────────────────────────────────────────
IMG_MEAN = np.array([0.485,0.456,0.406])
IMG_STD  = np.array([0.229,0.224,0.225])

def run_swinunet(device, output_dir, samples):
    sys.path.insert(0, SWINUNET_DIR)
    # patch sys.argv so get_config doesn't choke
    import sys as _sys
    _sys.argv = [_sys.argv[0],
                 '--cfg', f'{SWINUNET_DIR}/configs/swin_tiny_patch4_window7_224_lite.yaml']
    from config import get_config
    from networks.vision_transformer import SwinUnet as ViT_seg
    import argparse as ap
    fake_args = ap.Namespace(cfg=f'{SWINUNET_DIR}/configs/swin_tiny_patch4_window7_224_lite.yaml',
                             opts=None, batch_size=1, zip=False, cache_mode='no',
                             resume='', accumulation_steps=1, use_checkpoint=False,
                             amp_opt_level='O1', tag='', eval=True, throughput=False,
                             data_path='', output='')
    config = get_config(fake_args)
    img_size = 224
    net = ViT_seg(config, img_size=img_size, num_classes=2).to(device)
    snapshot = f'{BASE}/results/swinunet_polyp/best_model.pth'
    ckpt = torch.load(snapshot, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        net.load_state_dict(ckpt['model'], strict=False)
    else:
        net.load_state_dict(ckpt, strict=False)
    net.eval()

    metrics = {}
    with torch.no_grad():
        for s in samples:
            img_np = np.array(Image.open(s['img']).convert('RGB'))
            gt_np  = (np.array(Image.open(s['mask']).convert('L')) > 127).astype(np.uint8)
            oh, ow = img_np.shape[:2]

            img_r = np.array(Image.fromarray(img_np).resize((img_size, img_size), Image.BILINEAR))
            img_f = (img_r.astype(np.float32)/255.0 - IMG_MEAN) / IMG_STD
            img_t = torch.from_numpy(img_f.transpose(2,0,1)).float().unsqueeze(0).to(device)

            out = net(img_t)                                          # [1,2,224,224]
            out_full = F.interpolate(out, size=(oh,ow), mode='bilinear', align_corners=False)
            pred_np  = torch.argmax(torch.softmax(out_full,dim=1),dim=1).squeeze().cpu().numpy().astype(np.uint8)

            mask_path = os.path.join(output_dir,'masks','swinunet',s['dataset'],f'{s["stem"]}.png')
            save_mask(pred_np, mask_path)
            m = compute_metrics(pred_np, gt_np)
            metrics[f'{s["dataset"]}/{s["stem"]}'] = m
            print(f'  Swin-Unet {s["dataset"]}/{s["stem"]}: {m}')

    return metrics


# ── nnUNet (copy existing PNG predictions) ───────────────────────────────────
def copy_nnunet(output_dir, samples):
    metrics = {}
    for s in samples:
        pred_src = s['nnunet_pred']
        dst = os.path.join(output_dir,'masks','nnunet',s['dataset'],f'{s["stem"]}.png')
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        pred_np = np.array(Image.open(pred_src))  # values 0 or 1
        gt_np   = (np.array(Image.open(s['mask']).convert('L')) > 127).astype(np.uint8)

        # nnUNet masks may need resizing to original GT shape
        oh, ow = gt_np.shape
        if pred_np.shape != (oh, ow):
            pred_np = np.array(Image.fromarray((pred_np*255).astype(np.uint8)).resize((ow,oh),Image.NEAREST))//255
        pred_np = pred_np.astype(np.uint8)

        Image.fromarray((pred_np*255).astype(np.uint8)).save(dst)
        m = compute_metrics(pred_np, gt_np)
        metrics[f'{s["dataset"]}/{s["stem"]}'] = m
        print(f'  nnUNet {s["dataset"]}/{s["stem"]}: {m}')
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default=f'{BASE}/results/viz_comparison')
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--n_samples',  type=int, default=5)
    p.add_argument('--skip_swinunet', action='store_true')
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    samples = build_samples(seed=args.seed, n=args.n_samples)
    print(f'Seed={args.seed}, {len(samples)} samples: '
          + ', '.join(f'{s["dataset"]}/{s["stem"]}' for s in samples))

    all_metrics = {}

    if not args.skip_swinunet:
        print('\n=== Swin-Unet ===')
        all_metrics['swinunet'] = run_swinunet(device, args.output_dir, samples)
        torch.cuda.empty_cache()

    print('\n=== nnUNet (copy existing predictions) ===')
    all_metrics['nnunet'] = copy_nnunet(args.output_dir, samples)

    out_path = os.path.join(args.output_dir, 'metrics_swinunet_nnunet.json')
    with open(out_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\nMetrics saved → {out_path}')


if __name__ == '__main__':
    main()
