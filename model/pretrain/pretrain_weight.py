# pretrain_weight.py 
# torch 1.10.0 CPU/NO CUDA, torchvision 0.11.3)
# - model/pretrain/img_data 의 ROI 이미지를 사용해 TinyMobileNet(student)을
#   CPU teacher로부터 코사인 증류(pretext) 사전학습
# - 3뷰 파일명 cap_YYYYmmdd_HHMMSS_{c|lm|rm}.jpg 인식, 뷰별 증강/밸런싱 적용
# - 결과: .pth(state_dict) + .onnx 동시 저장 → 런타임 ONNX로 바로 사용
#
# 사용 예:
#   cd model/pretrain
#   python3 pretrain_weight.py \
#     --size 128 --out_dim 128 --width 0.35 --no_dw --no_bn \
#     --out_pth ../weights/tinymnet_emb_128d_w035.pth \
#     --out_onnx ../weights/tinymnet_emb_128d_w035.onnx
#
#   # 평가만:
#   python3 pretrain_weight.py --eval_only --out_pth ../weights/tinymnet_emb_128d_w035.pth

import os
import sys
import glob
import time
import argparse
from pathlib import Path
import copy
import math
import random

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
OUT_PTH  = ROOT / "weights" / "tinymnet_emb_128d_w035.pth"        # 결과 가중치 경로(기본)
OUT_ONNX = ROOT / "weights" / "tinymnet_emb_128d_w035.onnx"       # 결과 ONNX 경로(기본)
OUT_PTH.parent.mkdir(parents=True, exist_ok=True)

# 런타임과 반드시 일치
DEF_INPUT_SIZE   = 128
DEF_EMBED_DIM    = 128
DEF_WIDTH_SCALE  = 0.35
DEF_USE_DW       = False   # ★ 런타임과 맞추세요
DEF_USE_BN       = False   # ★ 런타임과 맞추세요

# 학습 기본값(환경에 맞게 조절 가능)
DEF_EPOCHS   = 5
DEF_BATCH    = 24
DEF_WORKERS  = 2
DEF_LR       = 1e-3
DEF_USE_AMP  = True   # CUDA 사용시에만 활성됨(아래에서 자동 판단)

# 디바이스: 교사=CPU, 학생=CUDA(가능 시) / 여기서는 보통 CPU
STUDENT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_DEVICE = "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# =====================
# TinyMobileNet (학습 전용 정의)
# =====================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, g=1, act=True, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k//2, groups=g, bias=(not use_bn))
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else None
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        x = self.act(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand, use_depthwise=True, use_bn=True):
        super().__init__()
        hidden = int(round(in_ch * expand))
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1.0:
            layers.append(ConvBNAct(in_ch, hidden, k=1, s=1, use_bn=use_bn))
        g = hidden if use_depthwise else 1
        layers += [
            ConvBNAct(hidden, hidden, k=3, s=stride, g=g, use_bn=use_bn),
            ConvBNAct(hidden, out_ch, k=1, s=1, act=False, use_bn=use_bn)
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out

class TinyMobileNet(nn.Module):
    def __init__(self, out_dim=DEF_EMBED_DIM, width=DEF_WIDTH_SCALE,
                 use_depthwise=DEF_USE_DW, use_bn=DEF_USE_BN):
        super().__init__()
        def c(ch): return max(8, int(ch * width))
        self.stem = ConvBNAct(3, c(16), k=3, s=2, use_bn=use_bn)
        self.layer1 = InvertedResidual(c(16),  c(24), 2, 2.0, use_depthwise, use_bn)
        self.layer2 = InvertedResidual(c(24),  c(24), 1, 2.0, use_depthwise, use_bn)
        self.layer3 = InvertedResidual(c(24),  c(40), 2, 2.5, use_depthwise, use_bn)
        self.layer4 = InvertedResidual(c(40),  c(40), 1, 2.5, use_depthwise, use_bn)
        self.layer5 = InvertedResidual(c(40),  c(80), 2, 2.5, use_depthwise, use_bn)
        self.layer6 = InvertedResidual(c(80),  c(80), 1, 2.5, use_depthwise, use_bn)
        self.head   = ConvBNAct(c(80), c(128), k=1, s=1, use_bn=use_bn)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(c(128), out_dim)
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc.weight, 0, 0.01); nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.layer5(x); x = self.layer6(x)
        x = self.head(x); x = self.pool(x).flatten(1)
        return self.fc(x)

# =====================
# 데이터셋 (3-view aware, shrink 미사용)
# =====================
def _infer_view_from_name(p: str) -> str:
    """파일명 접미사로 뷰 추정: '_c', '_lm', '_rm' (없으면 'c'로 간주)"""
    name = os.path.splitext(os.path.basename(p))[0].lower()
    if name.endswith("_lm"): return "lm"
    if name.endswith("_rm"): return "rm"
    if name.endswith("_c"):  return "c"
    # 예전 데이터나 임의 파일명은 center로 취급
    return "c"

def _list_images(root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for ext in exts:
        paths += glob.glob(str(root / "**" / ("*"+ext)), recursive=True)
    # 절대경로 + 유니크 + 정렬
    paths = sorted({os.path.abspath(p) for p in paths})
    return paths

class ThreeViewImageDataset(Dataset):
    """
    - view-aware 증강:
        center: RandomResizedCrop + RandomHorizontalFlip
        mirror(lm, rm): RandomResizedCrop (수평플립 없음)
    - balance_views: True면 c/lm/rm를 오버샘플링으로 균형 맞춤
    """
    def __init__(self, root, size=128, balance_views=True):
        super().__init__()
        root = Path(root)
        if not root.exists():
            raise RuntimeError("DATA_DIR가 존재하지 않습니다: %s" % str(root))
        all_paths = _list_images(root)
        if not all_paths:
            raise RuntimeError("이미지 파일이 없습니다: %s" % str(root))

        # 뷰 분할
        self.view2paths = {"c": [], "lm": [], "rm": []}
        for p in all_paths:
            v = _infer_view_from_name(p)
            if v not in self.view2paths: v = "c"
            self.view2paths[v].append(p)

        # 밸런싱
        self.items = []  # (path, view)
        if balance_views:
            counts = {k: len(vs) for k, vs in self.view2paths.items()}
            maxc = max(counts.values()) if counts else 0
            for v, vs in self.view2paths.items():
                if len(vs) == 0:  # 해당 뷰가 없으면 skip
                    continue
                rep = int(math.ceil(float(maxc) / float(len(vs))))
                tmp = (vs * rep)[:maxc]
                for p in tmp:
                    self.items.append((p, v))
        else:
            for v, vs in self.view2paths.items():
                for p in vs:
                    self.items.append((p, v))

        if not self.items:
            raise RuntimeError("유효한 학습 항목이 없습니다(뷰 분할/밸런싱 이후 비어있음).")

        # 증강(뷰별)
        # NOTE: shrink 미사용, orientation 분포 보존 목적
        self.tf_center = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.tf_mirror = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.95, 1.05)),
            # 거울뷰: 수평 플립은 제거(분포 일치)
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, v = self.items[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        if v == "c":
            return self.tf_center(im)
        else:
            return self.tf_mirror(im)

class EvalImageFolder(Dataset):
    def __init__(self, root, size=128):
        root = Path(root)
        paths = _list_images(root)
        if not paths:
            raise RuntimeError("이미지 파일이 없습니다: %s" % str(root))
        self.paths = paths
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

    if m is None and (local_weights is not None):
        try:
            if arch == "mnetv3":
                m = models.mobilenet_v3_small(pretrained=False)
                m.classifier[-1] = nn.Identity(); tdim = 1024
            else:
                m = models.resnet18(pretrained=False)
                m.fc = nn.Identity(); tdim = 512
            state = torch.load(str(local_weights), map_location="cpu")
            m.load_state_dict(state, strict=False)
            print("[teacher] 로컬 가중치 로드 성공:", str(local_weights))
        except Exception as e:
            print("[teacher] 로컬 가중치 로드 실패:", repr(e))

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
    def __init__(self, out_dim=DEF_EMBED_DIM, width=DEF_WIDTH_SCALE, tdim=1024,
                 use_depthwise=DEF_USE_DW, use_bn=DEF_USE_BN):
        super().__init__()
        self.student = TinyMobileNet(out_dim=out_dim, width=width,
                                     use_depthwise=use_depthwise, use_bn=use_bn)
        self.tproj = nn.Linear(tdim, out_dim, bias=False)
    def forward(self, x, tfeat):
        s = self.student(x)
        t = self.tproj(tfeat)
        return s, t

def cosine_distill_loss(s, t):
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()

# =====================
# 평가(라벨 불필요)
# =====================
def evaluate_student(data_dir, model_student, proj_layer, teacher, device, teacher_device,
                     input_size=128, max_imgs=512, batch_size=32, workers=2):
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
            t_cpu = teacher(xb.to(teacher_device))
            t = proj_layer(t_cpu.to(device, non_blocking=True))
            s = model_student(xb_dev)
            s = F.normalize(s, dim=1)
            t = F.normalize(t, dim=1)
            cos = (s * t).sum(dim=1)
            cos_list.append(cos.detach().cpu()); S_list.append(s.detach().cpu()); T_list.append(t.detach().cpu())

    import numpy as np
    cos_all = torch.cat(cos_list, 0).numpy()
    S = torch.cat(S_list, 0).numpy(); T = torch.cat(T_list, 0).numpy()

    S_sim = S @ S.T; T_sim = T @ T.T
    np.fill_diagonal(S_sim, -1.0); np.fill_diagonal(T_sim, -1.0)
    nn_S = S_sim.argmax(axis=1); nn_T = T_sim.argmax(axis=1)
    agree = float((nn_S == nn_T).mean())

    print("[eval] mean cos(student, teacher-proj)=%.4f ± %.4f (n=%d)" % (cos_all.mean(), cos_all.std(), cos_all.size))
    print("[eval] NN@1 agreement(student vs teacher)=%.2f%%" % (agree * 100.0))
    if cos_all.mean() < 0.6:
        print("[eval] note: 정렬이 낮습니다. 이미지 다양성↑, epoch↑, batch↑를 고려하세요.")
    elif agree < 0.5:
        print("[eval] note: 최근접 일치율이 낮습니다. 데이터 수/증강을 늘리거나 학습 파라미터를 조정하세요.")

# =====================
# ONNX Export (원본 모델 디바이스 보존: deepcopy 사용)
# =====================
def export_student_to_onnx(model_student: nn.Module, onnx_path: Path, input_size: int, out_dim: int):
    # 원본을 건드리지 않기 위해 deepcopy → CPU → eval 후 export
    m = copy.deepcopy(model_student).to("cpu").eval()
    dummy = torch.randn(1, 3, int(input_size), int(input_size), dtype=torch.float32)
    onnx_path = Path(onnx_path); onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m, dummy, str(onnx_path),
        input_names=["input"], output_names=["emb"],
        opset_version=12,
        dynamic_axes={"input": {0: "N"}, "emb": {0: "N"}}
    )
    print("[save] ONNX →", str(onnx_path))

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
    ap.add_argument("--out_onnx", type=str, default=str(OUT_ONNX))
    ap.add_argument("--size", type=int, default=DEF_INPUT_SIZE)
    ap.add_argument("--out_dim", type=int, default=DEF_EMBED_DIM)
    ap.add_argument("--width", type=float, default=DEF_WIDTH_SCALE)
    ap.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--workers", type=int, default=DEF_WORKERS)
    ap.add_argument("--lr", type=float, default=DEF_LR)
    ap.add_argument("--no_amp", action="store_true")

    # 구조 플래그(런타임과 동일해야 함)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dw", dest="use_dw", action="store_true", help="depthwise conv 사용")
    g.add_argument("--no_dw", dest="use_dw", action="store_false", help="depthwise conv 미사용")
    ap.set_defaults(use_dw=DEF_USE_DW)
    g2 = ap.add_mutually_exclusive_group()
    g2.add_argument("--bn", dest="use_bn", action="store_true", help="BatchNorm 사용")
    g2.add_argument("--no_bn", dest="use_bn", action="store_false", help="BatchNorm 미사용")
    ap.set_defaults(use_bn=DEF_USE_BN)

    # teacher 관련
    ap.add_argument("--teacher", type=str, default="mnetv3", choices=["mnetv3", "resnet18"])
    ap.add_argument("--teacher_weights", type=str, default=None,
                    help="오프라인 환경용 teacher 로컬 가중치(.pth) 경로")

    # 3뷰 관련
    ap.add_argument("--no_balance_views", action="store_true",
                    help="뷰(c/lm/rm) 밸런싱 오버샘플링 비활성화")

    # ONNX 내보내기 스위치
    ap.add_argument("--no_export_onnx", action="store_true", help="학습 후 ONNX 저장 생략")

    args = ap.parse_args()

    print("[env] torch=", torch.__version__, "cuda=", torch.cuda.is_available())
    print("[path] ROOT=", str(ROOT))
    print("[path] DATA_DIR=", args.data)
    print("[path] OUT_PTH=", args.out_pth)
    print("[path] OUT_ONNX=", args.out_onnx)
    print("[arch] size=%d out_dim=%d width=%.3f dw=%s bn=%s" %
          (args.size, args.out_dim, args.width, str(args.use_dw), str(args.use_bn)))

    # 데이터
    ds = ThreeViewImageDataset(args.data, size=args.size, balance_views=(not args.no_balance_views))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(STUDENT_DEVICE == "cuda"), drop_last=False)

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
    model = StudentWithProj(out_dim=args.out_dim, width=args.width, tdim=tdim,
                            use_depthwise=args.use_dw, use_bn=args.use_bn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_flag)

    # 평가만
    if args.eval_only:
        pth = Path(args.out_pth)
        if pth.exists():
            state = torch.load(str(pth), map_location=device)
            model.student.load_state_dict(state, strict=False)
            print("[load] student weights ←", str(pth))
        else:
            print("[warn] eval_only지만 가중치가 없습니다:", str(pth))
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

    # ONNX 내보내기 (런타임용) — deepcopy로 원본 디바이스 보존
    if not args.no_export_onnx:
        export_student_to_onnx(model.student, args.out_onnx, input_size=args.size, out_dim=args.out_dim)

    # (안전) 평가 전에 명시적으로 디바이스 정렬
    model.student.to(device).eval()

    # 빠른 성능 확인
    evaluate_student(args.data, model.student, model.tproj, teacher,
                     device=device, teacher_device=TEACHER_DEVICE,
                     input_size=args.size, max_imgs=args.eval_max,
                     batch_size=args.eval_batch, workers=args.workers)

if __name__ == "__main__":
    main()
