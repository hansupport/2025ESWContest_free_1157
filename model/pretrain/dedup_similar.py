# dedup_similar.py
# - model/pretrain/img_data 내의 이미지들 중 "과도하게 유사(거의-중복)" 샘플을 골라
#   별도 폴더로 옮기거나(기본) 목록만 출력합니다(--dry-run).
# - 기준 임베딩은 torchvision mobilenet_v3_small(pretrained, CPU).
# - Python 3.6, torch 1.10.0, torchvision 0.11.3( CPU )에서 동작.

import os, sys, glob, shutil, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def load_teacher():
    m = models.mobilenet_v3_small(pretrained=True)
    m.classifier[-1] = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m

def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for ext in exts:
        paths += glob.glob(str(Path(root) / "**" / ("*"+ext)), recursive=True)
    paths = sorted({os.path.abspath(p) for p in paths})
    return paths

def img_transform(size=224):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def compute_feats(paths, model, batch=32, size=224):
    tf = img_transform(size)
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="feat"):
            bs = []
            for p in paths[i:i+batch]:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                bs.append(tf(im))
            xb = torch.stack(bs, 0)  # CPU tensor
            f = model(xb)            # [B, 1024]
            f = torch.nn.functional.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
    F = np.concatenate(feats, axis=0).astype(np.float32)  # [N, D]
    return F

def greedy_dedup(F, paths, thr=0.993, order="mtime"):
    """
    간단 그리디: 기준 순서대로 하나씩 보며, 이미 keep된 것과 cos > thr 이면 중복으로 마킹.
    order:
      - "mtime": 파일 수정시간 오래된 것 우선 keep
      - "name" : 파일명 사전순으로 keep
    """
    assert F.shape[0] == len(paths)
    # 정렬
    idx = list(range(len(paths)))
    if order == "mtime":
        idx.sort(key=lambda i: os.path.getmtime(paths[i]))
    elif order == "name":
        idx.sort(key=lambda i: paths[i])
    # 진행
    keep = []
    dup  = []
    # 매 단계에서 keep된 것들과만 비교 → O(KN), N<=수천이면 충분
    K = []
    for i in idx:
        if not keep:
            keep.append(i)
            K.append(F[i])
            continue
        Kmat = np.stack(K, 0)           # [K, D]
        sims = np.dot(Kmat, F[i])       # [K]
        if np.any(sims > thr):
            dup.append(i)
        else:
            keep.append(i)
            K.append(F[i])
    return keep, dup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1] / "pretrain" / "img_data"))
    ap.add_argument("--thr", type=float, default=0.993, help="cosine similarity threshold to treat as duplicate")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--order", type=str, default="mtime", choices=["mtime","name"])
    ap.add_argument("--dry-run", action="store_true", help="move하지 않고 통계만 출력")
    ap.add_argument("--move-dir", type=str, default="img_dups", help="중복을 옮길 하위 폴더명")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print("[fatal] not found:", root); sys.exit(1)

    paths = list_images(root)
    if not paths:
        print("[fatal] no images under:", root); sys.exit(1)
    print(f"[scan] found {len(paths)} images")

    teacher = load_teacher()
    F = compute_feats(paths, teacher, batch=args.batch, size=args.size)

    keep_idx, dup_idx = greedy_dedup(F, paths, thr=args.thr, order=args.order)
    print(f"[result] keep={len(keep_idx)}  dups={len(dup_idx)}  (thr={args.thr})")

    # 목록 저장
    keep_list = [paths[i] for i in keep_idx]
    dup_list  = [paths[i] for i in dup_idx]
    (root / "KEEP_LIST.txt").write_text("\n".join(keep_list), encoding="utf-8")
    (root / "DUP_LIST.txt").write_text("\n".join(dup_list),  encoding="utf-8")

    if args.dry_run:
        print("[dry-run] no files moved. See KEEP_LIST.txt / DUP_LIST.txt")
        return

    # 중복 이동
    move_root = root / args.move_dir
    move_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in dup_list:
        rel = Path(p).relative_to(root)
        dst = move_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p, dst)
        moved += 1
    print(f"[move] moved {moved} duplicates into: {move_root}")

if __name__ == "__main__":
    main()
