#!/usr/bin/env python3
"""
Full-model parameter count for all methods.
Run in medsam env: conda run -n medsam python pilot_dual/count_params.py
"""
import sys, os, types, json, argparse
import torch
import numpy as np

BASE = '/volume/med-train/users/jshan/testSAMpruning'
sys.path.insert(0, BASE)
os.chdir(BASE)

def n(model):
    return sum(p.numel() for p in model.parameters())

# ─────────────────────────────────────────────────────────────────────────────
# 1. SAM ViT-B component breakdown
# ─────────────────────────────────────────────────────────────────────────────
from segment_anything import sam_model_registry
sam = sam_model_registry['vit_b'](checkpoint=f'{BASE}/work_dir/MedSAM/medsam_vit_b.pth')
ENC_PARAMS  = n(sam.image_encoder)   # 89,670,912 (what sweep used as denominator)
PE_PARAMS   = n(sam.prompt_encoder)  # 6,220
DEC_PARAMS  = n(sam.mask_decoder)    # 4,058,340
FULL_PARAMS = n(sam)                 # 93,735,472  (correct denominator)
del sam

comparison = {
    'MedSAM (SAM ViT-B)': FULL_PARAMS,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. EfficientSAM-Ti
# ─────────────────────────────────────────────────────────────────────────────
esam_dir = f'{BASE}/contrast_exp/EfficientSAM'
sys.path.insert(0, esam_dir)
_cwd = os.getcwd(); os.chdir(esam_dir)
try:
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
    comparison['EfficientSAM-Ti'] = n(build_efficient_sam_vitt())
finally:
    os.chdir(_cwd)
    sys.path.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SlimSAM-50
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, f'{BASE}/contrast_exp/SlimSAM')
slim = torch.load(f'{BASE}/contrast_exp/SlimSAM/checkpoints/SlimSAM-50.pth',
                  map_location='cpu', weights_only=False)
slim.image_encoder = slim.image_encoder.module
comparison['SlimSAM-50'] = n(slim)
del slim
sys.path.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAMed (SAM ViT-B + LoRA rank=4)
# ─────────────────────────────────────────────────────────────────────────────
samed_dir = f'{BASE}/contrast_exp/SAMed'
sys.path.insert(0, samed_dir)
_saved_sa = {k: v for k, v in sys.modules.items()
             if k == 'segment_anything' or k.startswith('segment_anything.')}
for k in list(_saved_sa): del sys.modules[k]
try:
    from segment_anything import sam_model_registry as samed_reg
    sam_base, _ = samed_reg['vit_b'](
        512, 1,
        checkpoint=f'{samed_dir}/checkpoints/sam_vit_b_01ec64.pth',
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    from importlib import import_module
    pkg = import_module('sam_lora_image_encoder_mask_decoder')
    net = pkg.LoRA_Sam(sam_base, 4)
    comparison['SAMed'] = n(net)
    del net, sam_base
finally:
    for k in list(sys.modules):
        if k == 'segment_anything' or k.startswith('segment_anything.'):
            del sys.modules[k]
    sys.modules.update(_saved_sa)
    if samed_dir in sys.path: sys.path.remove(samed_dir)

# ─────────────────────────────────────────────────────────────────────────────
# 5. QMedSAM
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, f'{BASE}/contrast_exp/QMedSAM')
from quantized_segment_anything import QuantLiteMedSAM
comparison['QMedSAM'] = n(QuantLiteMedSAM())
sys.path.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. EMCAD
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, f'{BASE}/contrast_exp/EMCAD')
from lib.networks import EMCADNet
comparison['EMCAD'] = n(EMCADNet(
    num_classes=1, kernel_sizes=[1, 3, 5, 7], expansion_factor=2,
    dw_parallel=True, add=True, lgag_ks=3, activation='relu',
    encoder='pvt_v2_b2', pretrain=False))
sys.path.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# 7. MK-UNet
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, f'{BASE}/contrast_exp/MK-UNet')
from mkunet_network import MK_UNet
comparison['MK-UNet'] = n(MK_UNet(num_classes=1, input_channels=3,
                                    channels=[16, 32, 64, 96, 160]))
sys.path.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Swin-Unet
# ─────────────────────────────────────────────────────────────────────────────
SWINUNET_DIR = f'{BASE}/contrast_exp/Swin-Unet'
sys.path.insert(0, SWINUNET_DIR)
sys.argv = [sys.argv[0], '--cfg',
            f'{SWINUNET_DIR}/configs/swin_tiny_patch4_window7_224_lite.yaml']
from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
fake_args = argparse.Namespace(
    cfg=f'{SWINUNET_DIR}/configs/swin_tiny_patch4_window7_224_lite.yaml',
    opts=None, batch_size=1, zip=False, cache_mode='no', resume='',
    accumulation_steps=1, use_checkpoint=False, amp_opt_level='O1',
    tag='', eval=True, throughput=False, data_path='', output='')
config = get_config(fake_args)
comparison['Swin-Unet'] = n(ViT_seg(config, img_size=224, num_classes=2))
sys.path.pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# 9. nnUNet (from checkpoint state_dict)
# ─────────────────────────────────────────────────────────────────────────────
nnunet_ckpt = (f'{BASE}/nnunet_data/results/Dataset001_Polyp/'
               'nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth')
ckpt = torch.load(nnunet_ckpt, map_location='cpu', weights_only=False)
sd = ckpt['network_weights']
comparison['nnUNet (2D)'] = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))

# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print(f"{'Model':<22} {'Params':>14}  {'Params (M)':>10}")
print('='*65)
for name, p in comparison.items():
    print(f"{name:<22} {p:>14,}  {p/1e6:>10.3f}M")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep results: corrected param_reduction
# ─────────────────────────────────────────────────────────────────────────────
print('\n\nSAM ViT-B breakdown:')
print(f'  image_encoder  : {ENC_PARAMS:>12,}  ({ENC_PARAMS/1e6:.3f}M)  ← sweep used this as total')
print(f'  prompt_encoder : {PE_PARAMS:>12,}  ({PE_PARAMS/1e6:.4f}M)')
print(f'  mask_decoder   : {DEC_PARAMS:>12,}  ({DEC_PARAMS/1e6:.3f}M)')
print(f'  FULL MODEL     : {FULL_PARAMS:>12,}  ({FULL_PARAMS/1e6:.3f}M)  ← correct total')

d = json.load(open(f'{BASE}/results/pilot_cascade_v8_sweep/cascade_results_v8.json'))
results = d['cascade_results']

seen = set()
rows = []
for r in results:
    if r.get('phase') != 'cascade':
        continue
    h = round(r['head_sp_target'], 2)
    m = round(r.get('mlp_sp_target', 0), 2)
    key = (h, m)
    if key in seen:
        continue
    seen.add(key)
    pruned  = r['n_params_pruned']
    old_pct = r['param_reduction_pct']
    new_pct = round(pruned / FULL_PARAMS * 100, 2)
    rows.append((h, m, pruned, old_pct, new_pct,
                 FULL_PARAMS - pruned,
                 r['macro']['mean_dice'],
                 r['macro']['mean_boundary_f1'],
                 r['macro']['mean_hd95']))

rows.sort(key=lambda x: (x[0], x[1]))

print('\n\nCorrected sweep param_reduction (full model denominator):')
hdr = ('h_sp', 'm_sp', 'pruned_M', 'old_red%', 'new_red%', 'remaining_M', 'Dice', 'BF1', 'HD95')
print(f"{hdr[0]:>5} {hdr[1]:>5} | {hdr[2]:>9} | {hdr[3]:>9} | {hdr[4]:>9} | {hdr[5]:>11} | {hdr[6]:>6} {hdr[7]:>6} {hdr[8]:>7}")
print('-' * 85)
for h, m, pruned, old, new, rem, dice, bf1, hd in rows:
    print(f"{h:>5.2f} {m:>5.2f} | {pruned/1e6:>8.3f}M | {old:>8.2f}% | {new:>8.2f}% | {rem/1e6:>10.3f}M | {dice:.4f} {bf1:.4f} {hd:>7.2f}")
