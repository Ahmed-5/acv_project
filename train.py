import torch
import torch.nn.functional as F
import numpy as np

from models.dog_depth_net import DoGDepthNet
from losses.dog_losses    import DoGDepthLoss


def build_student(device, pretrained=True):
    return DoGDepthNet(
        sigma_pairs=[(0.5,1.0),(1.0,2.0),(2.0,4.0)],
        branch_ch=64,
        pretrained=pretrained
    ).to(device)


def build_optimizer(model, lr=1e-4):
    dog_params  = list(model.dog.parameters())
    base_params = [p for p in model.parameters()
                   if not any(p is dp for dp in dog_params)]
    return torch.optim.AdamW([
        {'params': base_params, 'lr': lr},
        {'params': dog_params,  'lr': lr * 0.1}  # slow sigma adaptation
    ], weight_decay=1e-5)


def train_one_epoch(student, teacher, loader, optimizer, criterion, device):
    student.train();  teacher.eval()
    totals = {k: 0.0 for k in ['total','si','distill','consistency','edge']}

    for batch in loader:
        image    = batch['image'].to(device)   # (B,3,H,W)
        depth_gt = batch['depth'].to(device)   # (B,1,H,W) metric depth
        mask     = batch.get('mask')
        if mask is not None: mask = mask.to(device)

        with torch.no_grad():
            pred_teacher = teacher(image)
            pred_teacher = F.interpolate(pred_teacher, size=image.shape[-2:],
                                         mode='bilinear', align_corners=False)

        pred_student = student(image)           # (B,1,H,W)
        boundary     = student.boundary_map(image)

        losses = criterion(pred_student, pred_teacher, depth_gt,
                           image, boundary, mask)
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals: totals[k] += losses[k].item()

    n = len(loader)
    return {k: v/n for k,v in totals.items()}


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    abs_rels, rmses, delta1s = [], [], []

    for batch in loader:
        image    = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask     = (depth_gt > 0).squeeze(1)

        pred = F.interpolate(student(image), size=depth_gt.shape[-2:],
                             mode='bilinear', align_corners=False).squeeze(1)
        gt   = depth_gt.squeeze(1)

        for b in range(pred.shape[0]):
            p, g = pred[b][mask[b]], gt[b][mask[b]]
            scale = torch.median(g) / (torch.median(p) + 1e-8)
            p = p * scale
            thresh = torch.max(p/(g+1e-8), g/(p+1e-8))
            abs_rels.append(((p-g).abs()/(g+1e-8)).mean().item())
            rmses.append(((p-g).pow(2).mean().sqrt()).item())
            delta1s.append((thresh < 1.25).float().mean().item())

    return dict(AbsRel=np.mean(abs_rels),
                RMSE  =np.mean(rmses),
                delta1=np.mean(delta1s))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student  = build_student(device)
    optim    = build_optimizer(student, lr=1e-4)
    crit     = DoGDepthLoss(alpha=1.0, beta=1.0, gamma=0.1, delta=0.1)

    total_p = sum(p.numel() for p in student.parameters())
    dog_p   = sum(p.numel() for p in student.dog.parameters())
    print(f"Total params : {total_p/1e6:.2f}M")
    print(f"DoG params   : {dog_p}  (6 learnable sigmas)")
    print(f"Optimizer LR groups: base={optim.param_groups[0]['lr']}, "
          f"sigma={optim.param_groups[1]['lr']}")
    print(f"Initial sigma pairs: {student.get_learned_sigmas()}")
