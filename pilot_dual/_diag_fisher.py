import torch, sys, numpy as np
sys.path.insert(0, '/home/zhangcenwei/testSAMpruning')
from segment_anything import sam_model_registry
from pilot_phase1.dataset import build_dataloaders
import torch.nn.functional as F

device = torch.device('cuda:0')
model = sam_model_registry['vit_b'](checkpoint='work_dir/MedSAM/medsam_vit_b.pth')
model = model.to(device).eval()
cal_loader, _, _, _, _ = build_dataloaders('assert/CVC-ColonDB', n_calibration=16, batch_size=1, seed=42, num_workers=2)

for p in model.parameters(): p.requires_grad_(False)
for p in model.image_encoder.parameters(): p.requires_grad_(True)

fisher_fp16 = {n: torch.zeros_like(p, device='cpu') for n, p in model.image_encoder.named_parameters()}
fisher_fp32 = {n: torch.zeros_like(p, device='cpu') for n, p in model.image_encoder.named_parameters()}

for i, batch in enumerate(cal_loader):
    images = batch['image'].to(device)
    masks  = batch['mask_256'].to(device).float()
    bboxes = batch['bbox'].to(device).float()[:, None, :]
    
    # fp16
    model.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        img_emb = model.image_encoder(images)
        with torch.no_grad():
            se, de = model.prompt_encoder(points=None, boxes=bboxes, masks=None)
        lrm, _ = model.mask_decoder(image_embeddings=img_emb, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
    loss = F.binary_cross_entropy_with_logits(lrm.float(), masks, reduction='sum')
    loss.backward()
    for n, p in model.image_encoder.named_parameters():
        if p.grad is not None: fisher_fp16[n] += p.grad.detach().float().cpu() ** 2
    
    # fp32
    model.zero_grad()
    img_emb2 = model.image_encoder(images)
    with torch.no_grad():
        se2, de2 = model.prompt_encoder(points=None, boxes=bboxes, masks=None)
    lrm2, _ = model.mask_decoder(image_embeddings=img_emb2, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se2, dense_prompt_embeddings=de2, multimask_output=False)
    loss2 = F.binary_cross_entropy_with_logits(lrm2, masks, reduction='sum')
    loss2.backward()
    for n, p in model.image_encoder.named_parameters():
        if p.grad is not None: fisher_fp32[n] += p.grad.detach().float().cpu() ** 2
    model.zero_grad()

for f, name in [(fisher_fp16,'fp16'), (fisher_fp32,'fp32')]:
    for n in f: f[n] /= 16
    nan_count = sum(f[n].isnan().sum().item() for n in f)
    tot = sum(f[n].numel() for n in f)
    vals = torch.cat([f[n].reshape(-1) for n in f])
    print(f'{name}: NaN={nan_count}/{tot}  min={vals[~vals.isnan()].min():.3e}  max={vals[~vals.isnan()].max():.3e}  mean={vals[~vals.isnan()].mean():.3e}')
