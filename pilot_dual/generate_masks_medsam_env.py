#!/usr/bin/env python3
"""
Generate binary prediction masks for SAM-based models on selected polyp samples.
Runs in medsam conda env.

Models: MedSAM, Ours-h70_m95, Ours-h80_m95,
        EfficientSAM-Ti, SlimSAM-50, SAMed, QMedSAM

Usage:
    cd /volume/med-train/users/jshan/testSAMpruning
    conda run -n medsam python -m pilot_dual.generate_masks_medsam_env \
        --output_dir results/viz_comparison \
        --device cuda:0 --seed 42
"""
import argparse, os, sys, json, types
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
        samples.append({'dataset':'Kvasir',   'stem':stem,
                        'img': f'asserts/kvasir-seg/Kvasir-SEG/images/{stem}.jpg',
                        'mask':f'asserts/kvasir-seg/Kvasir-SEG/masks/{stem}.jpg'})
    for stem in c:
        samples.append({'dataset':'ClinicDB',  'stem':stem,
                        'img': f'asserts/CVC-ClinicDB/images/{stem}.png',
                        'mask':f'asserts/CVC-ClinicDB/masks/{stem}.png'})
    for stem in d:
        samples.append({'dataset':'ColonDB',   'stem':stem,
                        'img': f'asserts/CVC-ColonDB/images/{stem}.png',
                        'mask':f'asserts/CVC-ColonDB/masks/{stem}.png'})
    return samples


# ── Metrics ───────────────────────────────────────────────────────────────────
def _dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    if p.sum() == 0 and g.sum() == 0: return 1.0
    if p.sum() == 0 or  g.sum() == 0: return 0.0
    return float(2*(p&g).sum()/(p.sum()+g.sum()))

def _iou(p, g):
    p, g = p.astype(bool), g.astype(bool)
    u = (p|g).sum()
    return 1.0 if u == 0 else float((p&g).sum()/u)

def _bf1(p, g, r=2):
    def bd(m):
        m = m.astype(bool)
        if m.sum() == 0: return m
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
        if p.astype(bool).sum()>0 and g.astype(bool).sum()>0:
            return float(_hd95_fn(p.astype(bool), g.astype(bool)))
    except: pass
    return 100.0

def get_bbox(mask_np, pad=2):
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        h, w = mask_np.shape
        return np.array([0.,0.,w-1.,h-1.])
    return np.array([
        max(0, xs.min()-pad), max(0, ys.min()-pad),
        min(mask_np.shape[1]-1, xs.max()+pad),
        min(mask_np.shape[0]-1, ys.max()+pad)
    ], dtype=float)

def compute_metrics(pred, gt):
    return {
        'Dice':  round(_dice(pred, gt), 4),
        'IoU':   round(_iou(pred, gt), 4),
        'BF1':   round(_bf1(pred, gt), 4),
        'HD95':  round(_hd95_safe(pred, gt), 2),
    }

def save_mask(mask_np, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask_np * 255).astype(np.uint8)).save(path)

# ── MedSAM ───────────────────────────────────────────────────────────────────
def load_medsam(device):
    sys.path.insert(0, BASE)
    from segment_anything import sam_model_registry, SamPredictor
    model = sam_model_registry['vit_b'](checkpoint=f'{BASE}/work_dir/MedSAM/medsam_vit_b.pth')
    model.to(device).eval()
    return SamPredictor(model)

def infer_sam_predictor(predictor, img_np, gt_np):
    bbox = get_bbox(gt_np)
    predictor.set_image(img_np)
    masks, _, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=bbox, multimask_output=False
    )
    return masks[0].astype(np.uint8)

# ── Pruned model (h,m sparsity from v8 sweep) ────────────────────────────────
def make_pruned_predictor(h_sp, m_sp, device):
    sys.path.insert(0, BASE)
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

# ── EfficientSAM-Ti ──────────────────────────────────────────────────────────
def load_efficientSAM_ti(device):
    import os as _os
    esam_dir = f'{BASE}/contrast_exp/EfficientSAM'
    sys.path.insert(0, esam_dir)
    _cwd = _os.getcwd()
    _os.chdir(esam_dir)
    try:
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
        model = build_efficient_sam_vitt().to(device).eval()
    finally:
        _os.chdir(_cwd)
    return model

def infer_efficientSAM(model, img_np, gt_np, device):
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(Image.fromarray(img_np)).unsqueeze(0).to(device)
    x1,y1,x2,y2 = get_bbox(gt_np)
    pts    = torch.tensor([[[[x1,y1],[x2,y2]]]], dtype=torch.float32, device=device)
    labels = torch.tensor([[[2,3]]],             dtype=torch.int32,   device=device)
    with torch.no_grad():
        pred_logits, pred_iou = model(img_t, pts, labels)
    best = torch.argmax(pred_iou, dim=-1)
    logit = pred_logits[0, 0, best[0,0]]
    pred_np = (logit >= 0).cpu().numpy().astype(np.uint8)
    # resize to GT if needed
    gh, gw = gt_np.shape
    if pred_np.shape != (gh, gw):
        pred_np = np.array(Image.fromarray(pred_np*255).resize((gw,gh),Image.NEAREST))//255
    return pred_np

# ── SlimSAM-50 ────────────────────────────────────────────────────────────────
def load_slimsam(device):
    sys.path.insert(0, f'{BASE}/contrast_exp/SlimSAM')
    from segment_anything import SamPredictor
    ckpt = f'{BASE}/contrast_exp/SlimSAM/checkpoints/SlimSAM-50.pth'
    model = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.image_encoder = model.image_encoder.module

    def _forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None: x = x + self.pos_embed
        for blk in self.blocks:
            x, qkv_emb, mid_emb, x_emb = blk(x)
        x = self.neck(x.permute(0,3,1,2))
        return x
    model.image_encoder.forward = types.MethodType(_forward, model.image_encoder)

    for m in model.modules():
        if isinstance(m, torch.nn.GELU) and not hasattr(m, 'approximate'):
            m.approximate = 'none'

    model.to(device).eval()
    return SamPredictor(model)

# ── SAMed ─────────────────────────────────────────────────────────────────────
def load_samed(device):
    samed_dir = f'{BASE}/contrast_exp/SAMed'
    sys.path.insert(0, samed_dir)

    # The main segment_anything is already cached; temporarily swap in SAMed's version
    _saved_sa = {k: v for k, v in sys.modules.items()
                 if k == 'segment_anything' or k.startswith('segment_anything.')}
    for k in list(_saved_sa.keys()):
        del sys.modules[k]

    try:
        from segment_anything import sam_model_registry
        from importlib import import_module
        sam, _ = sam_model_registry['vit_b'](
            512, 1,
            checkpoint=f'{BASE}/contrast_exp/SAMed/checkpoints/sam_vit_b_01ec64.pth',
            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]
        )
        pkg = import_module('sam_lora_image_encoder_mask_decoder')
        net = pkg.LoRA_Sam(sam, 4).to(device)
        net.load_lora_parameters(f'{BASE}/contrast_exp/SAMed/model_pth/samed_polyp/samed_polyp-best.pth')
        net.eval()
    finally:
        # Restore main segment_anything
        for k in list(sys.modules.keys()):
            if k == 'segment_anything' or k.startswith('segment_anything.'):
                del sys.modules[k]
        sys.modules.update(_saved_sa)
        if samed_dir in sys.path:
            sys.path.remove(samed_dir)

    return net

def infer_samed(net, img_np, gt_np, img_size, device):
    s = img_size
    img_r = np.array(Image.fromarray(img_np).resize((s,s), Image.BILINEAR))
    img_t = torch.from_numpy(img_r.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(img_t, False, s)
    logit = out['masks']                       # [1,2,512,512]
    oh, ow = gt_np.shape
    logit = F.interpolate(logit, size=(oh,ow), mode='bilinear', align_corners=False)
    pred_np = torch.argmax(torch.softmax(logit,dim=1),dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred_np

# ── QMedSAM ───────────────────────────────────────────────────────────────────
def load_qmedsam(device):
    sys.path.insert(0, f'{BASE}/contrast_exp/QMedSAM')
    from quantized_segment_anything import QuantLiteMedSAM
    model = QuantLiteMedSAM()
    ckpt = torch.load(f'{BASE}/results/qmedsam_polyp/best_model.pth', map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def infer_qmedsam(model, img_np, gt_np, image_size, device):
    oh, ow = img_np.shape[:2]
    scale = image_size / max(oh, ow)
    nh, nw = int(oh*scale+0.5), int(ow*scale+0.5)
    img_r = np.array(Image.fromarray(img_np).resize((nw,nh), Image.NEAREST))
    gt_r  = np.array(Image.fromarray(gt_np).resize((nw,nh), Image.NEAREST))
    ph, pw = image_size-nh, image_size-nw
    img_p = np.pad(img_r, ((0,ph),(0,pw),(0,0)))
    gt_p  = np.pad(gt_r,  ((0,ph),(0,pw)))

    img_01 = (img_p - img_p.min()) / max(img_p.max() - img_p.min(), 1e-8)
    img_t  = torch.from_numpy(img_01.transpose(2,0,1)).float().unsqueeze(0).to(device)
    gt_b   = (gt_p > 0).astype(np.uint8)
    y_idx, x_idx = np.where(gt_b)
    if len(y_idx)==0:
        bbox = torch.tensor([[[0.,0.,image_size-1.,image_size-1.]]]).to(device)
    else:
        bbox = torch.tensor([[[float(x_idx.min()), float(y_idx.min()),
                                float(x_idx.max()), float(y_idx.max())]]]).to(device)

    with torch.no_grad():
        pred_logit = model(img_t, bbox)    # [1,1,256,256]
    pred_256 = (pred_logit[0,0].cpu().numpy() > 0).astype(np.uint8)
    # crop padding and resize back to original
    pred_crop = pred_256[:nh, :nw]
    pred_np = np.array(Image.fromarray(pred_crop*255).resize((ow,oh), Image.NEAREST))//255
    return pred_np.astype(np.uint8)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', default=f'{BASE}/results/viz_comparison')
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--n_samples',  type=int, default=5)
    p.add_argument('--models',     nargs='+',
                   default=['medsam','h70_m95','h80_m95','efficientSAM_ti','slimsam','samed','qmedsam'])
    args = p.parse_args()

    device = torch.device(args.device)
    os.chdir(BASE)
    os.makedirs(args.output_dir, exist_ok=True)

    SAMPLES = build_samples(seed=args.seed, n=args.n_samples)
    print(f'Seed={args.seed}, {len(SAMPLES)} samples: '
          + ', '.join(f'{s["dataset"]}/{s["stem"]}' for s in SAMPLES))

    all_metrics = {}

    for model_name in args.models:
        print(f'\n{"="*60}\n  Loading {model_name}\n{"="*60}')

        if model_name == 'medsam':
            predictor = load_medsam(device)
        elif model_name == 'h70_m95':
            predictor = make_pruned_predictor(0.7, 0.95, device)
        elif model_name == 'h80_m95':
            predictor = make_pruned_predictor(0.8, 0.95, device)
        elif model_name == 'slimsam':
            predictor = load_slimsam(device)
        elif model_name == 'efficientSAM_ti':
            esam_model = load_efficientSAM_ti(device)
        elif model_name == 'samed':
            samed_net = load_samed(device)
        elif model_name == 'qmedsam':
            qmed_model = load_qmedsam(device)

        all_metrics[model_name] = {}

        for sample in SAMPLES:
            ds, stem = sample['dataset'], sample['stem']
            img_np = np.array(Image.open(sample['img']).convert('RGB'))
            gt_np  = (np.array(Image.open(sample['mask']).convert('L')) > 127).astype(np.uint8)

            with torch.no_grad():
                if model_name in ('medsam', 'h70_m95', 'h80_m95', 'slimsam'):
                    pred = infer_sam_predictor(predictor, img_np, gt_np)
                elif model_name == 'efficientSAM_ti':
                    pred = infer_efficientSAM(esam_model, img_np, gt_np, device)
                elif model_name == 'samed':
                    pred = infer_samed(samed_net, img_np, gt_np, 512, device)
                elif model_name == 'qmedsam':
                    pred = infer_qmedsam(qmed_model, img_np, gt_np, 256, device)

            mask_path = os.path.join(args.output_dir, 'masks', model_name, ds, f'{stem}.png')
            save_mask(pred, mask_path)

            m = compute_metrics(pred, gt_np)
            all_metrics[model_name][f'{ds}/{stem}'] = m
            print(f'  {ds}/{stem}: Dice={m["Dice"]:.4f} IoU={m["IoU"]:.4f} BF1={m["BF1"]:.4f} HD95={m["HD95"]:.1f}')

        # free GPU memory between models
        if model_name in ('medsam','h70_m95','h80_m95','slimsam'):
            del predictor
        elif model_name == 'efficientSAM_ti':
            del esam_model
        elif model_name == 'samed':
            del samed_net
        elif model_name == 'qmedsam':
            del qmed_model
        torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, 'metrics_medsam_env.json')
    with open(out_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\nMetrics saved → {out_path}')
    print(f'Masks saved   → {args.output_dir}/masks/')


if __name__ == '__main__':
    main()
