# pretrain_weight.py
# Jetson Nano (Python 3.6, torch 1.10.0 CUDA, torchvision 0.11.3 CPU)
# - model/pretrain/img_data 의 ROI 이미지를 사용해 TinyMobileNet(img2emb.py)을
#   CPU teacher로부터 코사인 증류(pretext) 사전학습
# - 결과 가중치: model/weights/tinymnet_emb_128d_w035.pth
# - 학습 후 라벨 없이도 빠른 성능 확인(evaluate_student) 수행
#
# 사용 예:
#   cd model/pretrain
#   python3 pretrain_weight.py
#   # 또는 평가만:
#   python3 pretrain_weight.py --eval_only
#
# 옵션 보기:
#   python3 pretrain_weight.py -h

import os
import sys
import glob
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# torchvision은 CPU 빌드(0.11.3) 가정
from torchvision import models, transforms
from PIL import Image

# =====================
# 경로/기본값
# =====================
ROOT = Path(__file__).resolve().parents[1]                        # .../model
DATA_DIR = ROOT / "pretrain" / "img_data"                         # 캡처 이미지 폴더
OUT_PTH  = ROOT / "weights" / "tinymnet_emb_128d_w035.pth"        # 결과 가중치 경로
OUT_PTH.parent.mkdir(parents=True, exist_ok=True)

# 런타임과 반드시 일치
DEF_INPUT_SIZE   = 128
DEF_EMBED_DIM    = 128
DEF_WIDTH_SCALE  = 0.35

# 학습 기본값(환경에 맞게 조절 가능)
DEF_EPOCHS   = 5
DEF_BATCH    = 24
DEF_WORKERS  = 2
DEF_LR       = 1e-3
DEF_USE_AMP  = True   # CUDA + PyTorch 1.10.0이면 True 권장

# 디바이스: 교사=CPU, 학생=CUDA(가능 시)
STUDENT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_DEVICE = "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# img2emb.py(TinyMobileNet) import 경로 보장
sys.path.append(str(ROOT))
from img2emb import TinyMobileNet


# =====================
# 데이터셋
# =====================
class UnlabeledImageFolder(Dataset):
    """하위 폴더 재귀 탐색으로 이미지 모음(라벨 불필요, 학습용 증강 적용)."""
    def __init__(self, root, size=128):
        root = Path(root)
        if not root.exists():
            raise RuntimeError("DATA_DIR가 존재하지 않습니다: %s" % str(root))
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        paths = []
        for ext in exts:
            paths += glob.glob(str(root / "**" / ("*"+ext)), recursive=True)
        paths = sorted({os.path.abspath(p) for p in paths})
        if not paths:
            raise RuntimeError("이미지 파일이 없습니다: %s" % str(root))
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        return self.tf(im)


class EvalImageFolder(Dataset):
    """평가용(증강 없음, 크기 고정)."""
    def __init__(self, root, size=128):
        root = Path(root)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        paths = []
        for ext in exts:
            paths += glob.glob(str(root / "**" / ("*"+ext)), recursive=True)
        self.paths = sorted({os.path.abspath(p) for p in paths})
        if not self.paths:
            raise RuntimeError("이미지 파일이 없습니다: %s" % str(root))
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.size = int(size)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        return self.tf(im)


# =====================
# Teacher / Student
# =====================
def build_teacher_cpu(arch="mnetv3", local_weights=None):
    """
    torchvision 0.11.3:
      - mobilenet_v3_small(pretrained=True, 1024-d) 기본
      - 실패 시 resnet18(pretrained=True, 512-d) 폴백
      - 인터넷/캐시 문제로 pretrained 다운로드 실패하면, local_weights 로드 시도
      - 모두 실패하면 비사전학습 resnet18으로 폴백(성능 저하 경고)
    항상 CPU에서만 실행.
    """
    def try_load_pretrained_mnetv3():
        m = models.mobilenet_v3_small(pretrained=True)
        m.classifier[-1] = nn.Identity()
        return m, 1024

    def try_load_pretrained_resnet18():
        m = models.resnet18(pretrained=True)
        m.fc = nn.Identity()
        return m, 512

    m, tdim = None, None
    if arch == "mnetv3":
        try:
            m, tdim = try_load_pretrained_mnetv3()
            print("[teacher] MobileNetV3-Small(pretrained) 사용 (1024-d)")
        except Exception as e:
            print("[teacher] mobilenet_v3_small pretrained 실패:", repr(e))
    if m is None:
        try:
            m, tdim = try_load_pretrained_resnet18()
            print("[teacher] ResNet18(pretrained) 폴백 (512-d)")
        except Exception as e:
            print("[teacher] resnet18 pretrained 실패:", repr(e))

    # 로컬 가중치 제공 시 로드(사전학습 가중치 파일을 수동으로 넣은 경우)
    if m is None and (local_weights is not None):
        try:
            if arch == "mnetv3":
                m = models.mobilenet_v3_small(pretrained=False)
                m.classifier[-1] = nn.Identity()
                tdim = 1024
            else:
                m = models.resnet18(pretrained=False)
                m.fc = nn.Identity()
                tdim = 512
            state = torch.load(str(local_weights), map_location="cpu")
            m.load_state_dict(state, strict=False)
            print("[teacher] 로컬 가중치 로드 성공:", str(local_weights))
        except Exception as e:
            print("[teacher] 로컬 가중치 로드 실패:", repr(e))

    # 최후 폴백: 무사전학습 resnet18(품질 낮음)
    if m is None:
        m = models.resnet18(pretrained=False)
        m.fc = nn.Identity()
        tdim = 512
        print("[teacher] 경고: 비사전학습 ResNet18 사용(성능 저하 가능)")

    m.eval().to(TEACHER_DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tdim


class StudentWithProj(nn.Module):
    """Student(TinyMobileNet) + teacher feature projection(학습용). 저장 시엔 student만 저장."""
    def __init__(self, out_dim=128, width=0.35, tdim=1024):
        super(StudentWithProj, self).__init__()
        self.student = TinyMobileNet(out_dim=out_dim, width=width, use_depthwise=True, use_bn=True)
        self.tproj = nn.Linear(tdim, out_dim, bias=False)
    def forward(self, x, tfeat):
        s = self.student(x)
        t = self.tproj(tfeat)
        return s, t


def cosine_distill_loss(s, t):
    # s,t: [B,D] → L2 정규화 후 1 - cos 평균
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


# =====================
# 평가(라벨 불필요)
# =====================
def evaluate_student(data_dir, model_student, proj_layer, teacher, device, teacher_device,
                     input_size=128, max_imgs=512, batch_size=32, workers=2):
    """
    라벨 없이 하는 빠른 성능 확인:
      1) mean cos(student, teacher-proj)
      2) Nearest-Neighbor@1 agreement (teacher 공간 vs student 공간)
    """
    ds = EvalImageFolder(data_dir, size=input_size)
    if len(ds) > max_imgs:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(max_imgs)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=(device == "cuda"), drop_last=False)

    model_student.eval()
    teacher.eval()

    cos_list, S_list, T_list = [], [], []
    with torch.no_grad():
        for xb in dl:
            xb_dev = xb.to(device, non_blocking=True)
            t_cpu = teacher(xb.to(teacher_device))           # teacher on CPU
            t = proj_layer(t_cpu.to(device, non_blocking=True))
            s = model_student(xb_dev)
            s = F.normalize(s, dim=1)
            t = F.normalize(t, dim=1)
            cos = (s * t).sum(dim=1)
            cos_list.append(cos.detach().cpu())
            S_list.append(s.detach().cpu())
            T_list.append(t.detach().cpu())

    import numpy as np
    cos_all = torch.cat(cos_list, 0).numpy()
    S = torch.cat(S_list, 0).numpy()
    T = torch.cat(T_list, 0).numpy()

    # 최근접@1 일치율 (self 제외)
    S_sim = S @ S.T
    T_sim = T @ T.T
    np.fill_diagonal(S_sim, -1.0)
    np.fill_diagonal(T_sim, -1.0)
    nn_S = S_sim.argmax(axis=1)
    nn_T = T_sim.argmax(axis=1)
    agree = float((nn_S == nn_T).mean())

    print("[eval] mean cos(student, teacher-proj)=%.4f ± %.4f (n=%d)" %
          (cos_all.mean(), cos_all.std(), cos_all.size))
    print("[eval] NN@1 agreement(student vs teacher)=%.2f%%" % (agree * 100.0))

    # 간단 가이드
    if cos_all.mean() < 0.6:
        print("[eval] note: 정렬이 낮습니다. 이미지 다양성↑, epoch↑, batch↑를 고려하세요.")
    elif agree < 0.5:
        print("[eval] note: 최근접 일치율이 낮습니다. 데이터 수/증강을 늘리거나 학습 파라미터를 조정하세요.")


# =====================
# 메인
# =====================
def main():
    ap = argparse.ArgumentParser()
    # 평가 옵션
    ap.add_argument("--eval_only", action="store_true", help="학습 생략하고 평가만 수행")
    ap.add_argument("--eval_max", type=int, default=512, help="평가에 사용할 최대 이미지 수")
    ap.add_argument("--eval_batch", type=int, default=32, help="평가 배치 크기")

    # 경로/모델/학습 하이퍼파라미터
    ap.add_argument("data", type=str, nargs="?", default=str(DATA_DIR),
                    help="라벨 불필요, 이미지 루트 폴더(재귀 검색)")
    ap.add_argument("--out_pth", type=str, default=str(OUT_PTH))
    ap.add_argument("--size", type=int, default=DEF_INPUT_SIZE)
    ap.add_argument("--out_dim", type=int, default=DEF_EMBED_DIM)
    ap.add_argument("--width", type=float, default=DEF_WIDTH_SCALE)
    ap.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--workers", type=int, default=DEF_WORKERS)
    ap.add_argument("--lr", type=float, default=DEF_LR)
    ap.add_argument("--no_amp", action="store_true")

    # teacher 관련
    ap.add_argument("--teacher", type=str, default="mnetv3", choices=["mnetv3", "resnet18"])
    ap.add_argument("--teacher_weights", type=str, default=None,
                    help="오프라인 환경용 teacher 로컬 가중치(.pth) 경로")

    args = ap.parse_args()

    print("[env] torch=", torch.__version__, "cuda=", torch.cuda.is_available())
    print("[path] ROOT=", str(ROOT))
    print("[path] DATA_DIR=", args.data)
    print("[path] OUT_PTH=", args.out_pth)

    # 데이터
    ds = UnlabeledImageFolder(args.data, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=(STUDENT_DEVICE == "cuda"),
                    drop_last=False)

    # 교사(항상 CPU)
    teacher, tdim = build_teacher_cpu(arch=args.teacher,
                                      local_weights=(Path(args.teacher_weights) if args.teacher_weights else None))
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.size, args.size).to(TEACHER_DEVICE)
        feat = teacher(dummy)
        infer_dim = int(feat.shape[-1])
        print("[teacher] feature_dim=", infer_dim)
        if infer_dim != tdim:
            print("[warn] 예상 tdim(%d) != 실제(%d) → 보정" % (tdim, infer_dim))
            tdim = infer_dim

    # 학생(가능하면 CUDA)
    device = STUDENT_DEVICE
    use_amp_flag = (device == "cuda" and (not args.no_amp) and DEF_USE_AMP)
    model = StudentWithProj(out_dim=args.out_dim, width=args.width, tdim=tdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_flag)

    # 평가만
    if args.eval_only:
        if Path(args.out_pth).exists():
            state = torch.load(str(args.out_pth), map_location=device)
            model.student.load_state_dict(state, strict=False)
            print("[load] student weights ←", args.out_pth)
        else:
            print("[warn] eval_only지만 가중치가 없습니다:", args.out_pth)
        evaluate_student(args.data, model.student, model.tproj, teacher,
                         device=device, teacher_device=TEACHER_DEVICE,
                         input_size=args.size, max_imgs=args.eval_max,
                         batch_size=args.eval_batch, workers=args.workers)
        return

    # 학습
    steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for i, xb in enumerate(dl):
            xb = xb.to(device, non_blocking=True)

            # teacher forward on CPU
            with torch.no_grad():
                tfeat_cpu = teacher(xb.detach().to(TEACHER_DEVICE))
            tfeat = tfeat_cpu.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp_flag):
                s, t = model(xb, tfeat)
                loss = cosine_distill_loss(s, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())
            steps += 1
            if (i + 1) % 50 == 0:
                print("[epoch %d] step %d/%d loss=%.4f" %
                      (epoch, i + 1, len(dl), loss_sum / (i + 1)))

        print("[epoch %d] mean_loss=%.4f | time=%.1fs" %
              (epoch, loss_sum / max(1, len(dl)), time.time() - t0))

    # 저장: student만
    state = model.student.state_dict()
    torch.save(state, str(args.out_pth))
    print("[save] student state_dict →", str(args.out_pth))

    # 빠른 성능 확인
    evaluate_student(args.data, model.student, model.tproj, teacher,
                     device=device, teacher_device=TEACHER_DEVICE,
                     input_size=args.size, max_imgs=args.eval_max,
                     batch_size=args.eval_batch, workers=args.workers)


if __name__ == "__main__":
    main()
