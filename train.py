import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import random
import time
from pathlib import Path
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageChops, ImageFilter

import torchvision.models as models

from model import UpscalerTransformer


IMG_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Папка с датасетом не найдена: {folder}")

    files = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTENSIONS:
            files.append(str(p))

    return sorted(files)


def split_dataset(paths, val_split, seed):
    paths = sorted(paths)

    if len(paths) == 0:
        return [], []

    if len(paths) == 1:
        return paths, paths

    rng = random.Random(seed)
    rng.shuffle(paths)

    if val_split <= 0:
        return paths, paths

    val_count = max(1, int(len(paths) * val_split))
    val_paths = paths[:val_count]
    train_paths = paths[val_count:]

    return train_paths, val_paths
    
    
import torch
import torch.nn as nn

def _load_vgg16():
    try:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        return models.vgg16(weights=weights)
    except Exception:
        return models.vgg16(pretrained=True)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss на признаках VGG16.
    Использует признаки до relu3_3.
    """

    def __init__(self):
        super().__init__()

        vgg = _load_vgg16()

        # VGG16 features:
        # 0 conv1_1, 1 relu, 2 conv1_2, 3 relu,
        # 4 maxpool,
        # 5 conv2_1, 6 relu, 7 conv2_2, 8 relu,
        # 9 maxpool,
        # 10 conv3_1, 11 relu, 12 conv3_2, 13 relu, 14 conv3_3, 15 relu
        self.features = nn.Sequential(*list(vgg.features.children())[:16])

        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        # Всегда держим VGG в eval-режиме.
        return super().train(False)

    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        sr_features = self.features(sr)
        hr_features = self.features(hr)

        return F.l1_loss(sr_features, hr_features)


class FFTLoss(nn.Module):
    """
    FFT loss помогает восстанавливать высокие частоты:
    резкость, мелкие детали, текстуру.
    """

    def forward(self, sr, hr):
        sr_fft = torch.fft.rfft2(sr, norm="ortho")
        hr_fft = torch.fft.rfft2(hr, norm="ortho")

        sr_mag = torch.log1p(torch.abs(sr_fft))
        hr_mag = torch.log1p(torch.abs(hr_fft))

        return F.l1_loss(sr_mag, hr_mag)


class SRDataset(Dataset):
    def __init__(
        self,
        image_paths,
        hr_size=256,
        lr_size=128,
        train=True,
    ):
        self.image_paths = image_paths
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_hr = Image.open(self.image_paths[idx]).convert("RGB")
                
        if self.train:
            rnd = random.randint(0, 128)
            area = (rnd, rnd, 1024 - rnd, 1024 - rnd)
            img_hr = img_hr.crop(area)                    
            img_hr = img_hr.resize((self.hr_size, self.hr_size), Image.Resampling.BICUBIC)
            
            if random.randint(0, 1) == 0:
                img_hr = img_hr.transpose(Image.FLIP_LEFT_RIGHT)
                
            if random.randint(0, 1) == 0:
                random_choice = random.randint(0, 2)                
                if random_choice == 0:
                    img_hr = img_hr.transpose(Image.Transpose.ROTATE_90)
                elif random_choice == 1:
                    img_hr = img_hr.transpose(Image.Transpose.ROTATE_180)
                elif random_choice == 2:
                    img_hr = img_hr.transpose(Image.Transpose.ROTATE_270) 
                                  
            if random.randint(0, 1) == 0:
                rnd_x = random.randint(-self.hr_size // 4, self.hr_size // 4)
                rnd_y = random.randint(-self.hr_size // 4, self.hr_size // 4)                                          
                img_hr = ImageChops.offset(img_hr, xoffset=rnd_x, yoffset=rnd_y) 
           
            degradation_type = random.randint(0, 4)
            
            if degradation_type == 0:
                img_lr = img_hr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
                buffer = io.BytesIO()    
                img_lr.save(buffer, format="JPEG", quality=random.randint(10, 90), optimize=True)    
                buffer.seek(0)
                img_lr = Image.open(buffer)
                img_lr.load()
    
            elif degradation_type == 1:
                img_lr = img_hr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
                img_lr = img_lr.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                
            elif degradation_type == 2:        
                intermediate_size = random.randint(self.lr_size // 2, self.lr_size)            
                img_lr = img_hr.resize((intermediate_size, intermediate_size), Image.Resampling.BICUBIC)        
                img_lr = img_lr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
                
            elif degradation_type == 3:
                img_lr = img_hr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
                noise = Image.effect_noise((self.lr_size, self.lr_size), sigma=random.randint(10, 50)).convert("RGB")
                img_lr = Image.blend(img_lr, noise, alpha=random.uniform(0.1, 0.2))
                
            elif degradation_type == 4:       
                img_lr = img_hr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
        else:
            img_hr = img_hr.resize((self.hr_size, self.hr_size), Image.Resampling.BICUBIC)
            img_lr = img_hr.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC) 

        hr = TF.to_tensor(img_hr)          
        lr = TF.to_tensor(img_lr)    

        return lr, hr    

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    args,
    perceptual_loss,
    fft_loss,
):
    model.train()

    total_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)

    for lr, hr in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                sr = model(lr)
                loss = args.w_l1 * F.l1_loss(sr.float(), hr)
        else:
            sr = model(lr)
            loss = args.w_l1 * F.l1_loss(sr, hr)

        sr_f = sr.float()

        if perceptual_loss is not None:
            loss = loss + args.w_perceptual * perceptual_loss(sr_f, hr)

        if fft_loss is not None:
            loss = loss + args.w_fft * fft_loss(sr_f, hr)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * lr.size(0)
        num_samples += lr.size(0)
        
        avg_loss = total_loss / max(1, num_samples)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_psnr = 0.0
    count = 0
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)

    for lr, hr in pbar:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)
        sr = torch.clamp(sr.float(), 0.0, 1.0)

        mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

        total_psnr += psnr.sum().item()
        count += lr.size(0)
        
        # Обновляем прогресс-бар текущим средним PSNR
        avg_psnr = total_psnr / max(1, count)
        pbar.set_postfix(psnr=f"{avg_psnr:.2f} dB")

    return total_psnr / max(1, count)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Папка с HR-изображениями.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs_quality",
        help="Куда сохранять чекпоинты.",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Путь к last.pth или best.pth для возобновления обучения.",
    )

    parser.add_argument(
        "--reset_scheduler",
        action="store_true",
        help="Пересчитать scheduler под новый --epochs при возобновлении.",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Параметры модели.
    # Для качества лучше больше, чем для скорости.
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)

    # Датасет.
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--lr_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.05)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Loss weights.
    parser.add_argument("--w_l1", type=float, default=1.0)
    
    parser.add_argument("--w_perceptual", type=float, default=0.2)
    parser.add_argument("--w_fft", type=float, default=0.05)

    # Если нужно отключить тяжёлые лоссы.
    parser.add_argument(
        "--no_perceptual",
        action="store_true",
        help="Отключить VGG perceptual loss.",
    )
    parser.add_argument(
        "--no_fft",
        action="store_true",
        help="Отключить FFT loss.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True

    paths = find_images(args.data_dir)
    print(f"Найдено изображений: {len(paths)}")

    train_paths, val_paths = split_dataset(paths, args.val_split, args.seed)
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_dataset = SRDataset(
        image_paths=train_paths,
        hr_size=args.hr_size,
        lr_size=args.lr_size,
        train=True,
    )

    val_dataset = SRDataset(
        image_paths=val_paths,
        hr_size=args.hr_size,
        lr_size=args.lr_size,
        train=False,
    )

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = UpscalerTransformer(
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        window_size=args.window,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    perceptual_loss = None
    if not args.no_perceptual:
        try:
            print("Загрузка VGG perceptual loss...")
            perceptual_loss = VGGPerceptualLoss().to(device).eval()
            print("VGG perceptual loss включён.")
        except Exception as e:
            print(f"Не удалось загрузить VGG perceptual loss: {e}")
            print("Продолжаю без perceptual loss.")
            perceptual_loss = None

    fft_loss = None
    if not args.no_fft:
        fft_loss = FFTLoss().to(device)
        print("FFT loss включён.")

        start_epoch = 1
    best_psnr = -float("inf")

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(f"Чекпоинт для возобновления не найден: {resume_path}")

        print(f"Возобновление обучения из: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)

        if isinstance(ckpt, dict):
            # Веса модели
            model.load_state_dict(ckpt.get("model", ckpt))

            # Оптимизатор
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])

            # Scheduler
            if (
                not args.reset_scheduler
                and "scheduler" in ckpt
                and ckpt["scheduler"] is not None
            ):
                scheduler.load_state_dict(ckpt["scheduler"])
            elif "epoch" in ckpt:
                # Если scheduler не был сохранён или сброшен,
                # восстанавливаем его позицию вручную.
                for _ in range(ckpt["epoch"]):
                    scheduler.step()

            # AMP scaler
            if use_amp and "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])

            start_epoch = ckpt.get("epoch", 0) + 1
            best_psnr = ckpt.get("best_psnr", ckpt.get("val_psnr", -float("inf")))

        else:
            # Если это просто state_dict модели
            model.load_state_dict(ckpt)

        print(f"Продолжаю с эпохи: {start_epoch}")
        print(f"Лучший PSNR на данный момент: {best_psnr:.3f} dB")

    config = {
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "window": args.window,
        "hr_size": args.hr_size,
        "lr_size": args.lr_size,
    }

    if start_epoch > args.epochs:
        print("Обучение уже завершено для указанного --epochs.")
        print("Увеличь --epochs, если хочешь дообучить модель.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            args=args,
            perceptual_loss=perceptual_loss,
            fft_loss=fft_loss,
        )

        val_psnr = evaluate(model, val_loader, device)

        scheduler.step()

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "best_psnr": max(best_psnr, val_psnr),
            "config": config,
        }

        torch.save(state, out_dir / "last.pth")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(state, out_dir / "best.pth")

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"loss: {train_loss:.6f} | "
            f"val PSNR: {val_psnr:.3f} dB | "
            f"best PSNR: {best_psnr:.3f} dB | "
            f"time: {dt:.2f}s"
        )


if __name__ == "__main__":
    main()