import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import build_unet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.tensor_ops import crop_like

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def parse_mask_values(raw: str | None):
    if raw is None:
        return None
    values = []
    for value in raw.split(','):
        value = value.strip()
        if value:
            values.append(int(value))
    return values


def parse_float_values(raw: str | None):
    if raw is None:
        return None
    values = []
    for value in raw.split(','):
        value = value.strip()
        if value:
            values.append(float(value))
    return values


def configure_torch_runtime(device: torch.device, deterministic: bool = False):
    if device.type != 'cuda':
        return

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass


class PaperStyleAugmentedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,
            channels: int,
            elastic_alpha: float = 10.0,
            elastic_grid_size: int = 3,
            rotate_degrees: float = 20.0,
            translate_fraction: float = 0.08,
            brightness: float = 0.12,
            contrast: float = 0.12,
    ):
        self.dataset = dataset
        self.channels = channels
        self.elastic_alpha = elastic_alpha
        self.elastic_grid_size = elastic_grid_size
        self.rotate_degrees = rotate_degrees
        self.translate_fraction = translate_fraction
        self.brightness = brightness
        self.contrast = contrast

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        mask = sample['mask']

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            if 'weight' in sample:
                sample['weight'] = TF.hflip(sample['weight'].unsqueeze(0)).squeeze(0)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            if 'weight' in sample:
                sample['weight'] = TF.vflip(sample['weight'].unsqueeze(0)).squeeze(0)

        height, width = mask.shape[-2:]
        angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
        translate = (
            int(random.uniform(-self.translate_fraction, self.translate_fraction) * width),
            int(random.uniform(-self.translate_fraction, self.translate_fraction) * height),
        )
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[0.0] * self.channels,
        )
        mask = TF.affine(
            mask.unsqueeze(0).float(),
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=[0.0],
        ).squeeze(0).long()
        if 'weight' in sample:
            weight = TF.affine(
                sample['weight'].unsqueeze(0).float(),
                angle=angle,
                translate=translate,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=[1.0],
            ).squeeze(0)
        else:
            weight = None

        if self.elastic_alpha > 0:
            coarse = torch.randn(1, 2, self.elastic_grid_size, self.elastic_grid_size) * self.elastic_alpha
            displacement = F.interpolate(coarse, size=(height, width), mode='bicubic', align_corners=True)
            displacement = displacement.permute(0, 2, 3, 1)
            image = TF.elastic_transform(
                image,
                displacement,
                interpolation=InterpolationMode.BILINEAR,
                fill=[0.0] * self.channels,
            )
            mask = TF.elastic_transform(
                mask.unsqueeze(0).float(),
                displacement,
                interpolation=InterpolationMode.NEAREST,
                fill=[0.0],
            ).squeeze(0).long()
            if weight is not None:
                weight = TF.elastic_transform(
                    weight.unsqueeze(0).float(),
                    displacement,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=[1.0],
                ).squeeze(0)

        image = image * random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        image = image + random.uniform(-self.brightness, self.brightness)
        image = image.clamp(0.0, 1.0)

        out = {'image': image.contiguous(), 'mask': mask.contiguous()}
        if weight is not None:
            out['weight'] = weight.clamp_min(0.0).contiguous()
        return out


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        num_workers: int | None = None,
        checkpoint_dir: Path = dir_checkpoint,
        img_dir: Path = dir_img,
        mask_dir: Path = dir_mask,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        save_best: bool = False,
        abort_on_nonfinite: bool = False,
        optimizer_name: str = 'rmsprop',
        evals_per_epoch: int = 5,
        log_histograms: bool = False,
        log_images: bool = False,
        wandb_mode: str | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        seed: int = 0,
        mask_values=None,
        augment: str = 'none',
        channels: int = 3,
        elastic_alpha: float = 10.0,
        elastic_grid_size: int = 3,
        class_weights=None,
        progress: bool = True,
        weight_dir: Path | None = None,
        architecture: str = 'milesial',
        loss_mode: str = 'repo',
):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(img_dir, mask_dir, img_scale, mask_values=mask_values, weight_dir=weight_dir)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(img_dir, mask_dir, img_scale, mask_values=mask_values, weight_dir=weight_dir)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    if augment == 'paper-light':
        train_set = PaperStyleAugmentedDataset(
            train_set,
            channels=channels,
            elastic_alpha=elastic_alpha,
            elastic_grid_size=elastic_grid_size,
        )

    # 3. Create data loaders
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 0)
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == 'cuda')
    if num_workers > 0:
        loader_args['persistent_workers'] = persistent_workers
        loader_args['prefetch_factor'] = prefetch_factor
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    wandb_kwargs = dict(project='U-Net', resume='allow', anonymous='must')
    if wandb_mode:
        wandb_kwargs['mode'] = wandb_mode
    experiment = wandb.init(**wandb_kwargs)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             num_workers=num_workers, evals_per_epoch=evals_per_epoch, optimizer=optimizer_name,
             prefetch_factor=prefetch_factor if num_workers > 0 else None,
             persistent_workers=persistent_workers if num_workers > 0 else False,
             log_histograms=log_histograms, log_images=log_images, seed=seed,
             augment=augment, elastic_alpha=elastic_alpha if augment == 'paper-light' else None,
              elastic_grid_size=elastic_grid_size if augment == 'paper-light' else None,
              class_weights=class_weights, progress=progress, weight_dir=str(weight_dir) if weight_dir else None,
              architecture=architecture, loss_mode=loss_mode)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Optimizer:       {optimizer_name}
        Architecture:    {architecture}
        Loss mode:       {loss_mode}
        Workers:         {num_workers}
        Evals / epoch:   {evals_per_epoch}
        Augmentation:    {augment}
        Class weights:   {class_weights}
        Weight maps:     {weight_dir}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if model.n_classes > 1:
        weight_tensor = None
        if class_weights is not None:
            if len(class_weights) != model.n_classes:
                raise ValueError(f'Expected {model.n_classes} class weights, got {len(class_weights)}')
            weight_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction='none' if weight_dir else 'mean')
    else:
        criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    best_val_score = float('-inf')
    nonfinite_detected = False

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', disable=not progress) as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                non_blocking = device.type == 'cuda'
                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                    non_blocking=non_blocking,
                )
                true_masks = true_masks.to(device=device, dtype=torch.long, non_blocking=non_blocking)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    true_masks = crop_like(true_masks, masks_pred)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        if loss_mode == 'repo':
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        if weight_dir and 'weight' in batch:
                            pixel_loss = criterion(masks_pred, true_masks)
                            weights = batch['weight'].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                            weights = crop_like(weights, masks_pred)
                            loss = (pixel_loss * weights).mean()
                        else:
                            loss = criterion(masks_pred, true_masks)
                        if loss_mode == 'repo':
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                if not torch.isfinite(loss):
                    nonfinite_detected = True
                    logging.error(f'Non-finite loss detected at epoch={epoch}, step={global_step + 1}: {loss.item()}')
                    if abort_on_nonfinite:
                        break

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = 0 if evals_per_epoch <= 0 else max(n_train // (evals_per_epoch * batch_size), 1)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        if log_histograms:
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        if save_best and float(val_score) > best_val_score:
                            best_val_score = float(val_score)
                            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = dataset.mask_values
                            state_dict['architecture'] = architecture
                            torch.save(state_dict, str(checkpoint_dir / 'checkpoint_best.pth'))
                            logging.info(f'Best checkpoint saved! Validation Dice score: {best_val_score}')
                        try:
                            log_payload = {
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            }
                            if log_images:
                                log_payload.update({
                                    'images': wandb.Image(images[0].detach().cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().detach().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().detach().cpu()),
                                    },
                                })
                            experiment.log(log_payload)
                        except:
                            pass

            if nonfinite_detected and abort_on_nonfinite:
                break

        if save_checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            state_dict['architecture'] = architecture
            torch.save(state_dict, str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        if nonfinite_detected and abort_on_nonfinite:
            raise RuntimeError('Aborted training because a non-finite loss was detected')

    if device.type == 'cuda':
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        logging.info(f'Peak CUDA memory allocated: {peak_mib:.2f} MiB')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--architecture', choices=['milesial', 'original'], default='milesial', help='Model architecture')
    parser.add_argument('--loss-mode', choices=['repo', 'paper-ce'], default='repo', help='repo uses CE/BCE plus Dice; paper-ce uses weighted pixel CE only')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', type=int, default=3, help='Number of input image channels')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of DataLoader workers')
    parser.add_argument('--checkpoint-dir', type=Path, default=dir_checkpoint, help='Directory for checkpoints')
    parser.add_argument('--img-dir', type=Path, default=dir_img, help='Directory containing input images')
    parser.add_argument('--mask-dir', type=Path, default=dir_mask, help='Directory containing target masks')
    parser.add_argument('--weight-dir', type=Path, default=None, help='Directory containing per-pixel weight maps named <id>_weight.npy')
    parser.add_argument('--no-save-checkpoint', action='store_false', dest='save_checkpoint', default=True, help='Disable epoch checkpoint saves')
    parser.add_argument('--save-best', action='store_true', default=False, help='Save checkpoint_best.pth on validation improvement')
    parser.add_argument('--abort-on-nonfinite', action='store_true', default=False, help='Stop training when loss becomes NaN or infinite')
    parser.add_argument('--optimizer', choices=['rmsprop', 'adamw', 'sgd'], default='rmsprop', help='Optimizer to use for training')
    parser.add_argument('--weight-decay', type=float, default=1e-8, help='Optimizer weight decay')
    parser.add_argument('--momentum', type=float, default=0.999, help='Optimizer momentum for RMSprop/SGD')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--evals-per-epoch', type=int, default=5, help='Validation rounds per epoch; use 1 for faster full runs, 0 to disable during training')
    parser.add_argument('--log-histograms', action='store_true', default=False, help='Log parameter histograms to W&B during validation')
    parser.add_argument('--log-images', action='store_true', default=False, help='Log sample images to W&B during validation')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default=None, help='Override W&B mode')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch factor when num_workers > 0')
    parser.add_argument('--no-persistent-workers', action='store_false', dest='persistent_workers', default=True, help='Disable persistent DataLoader workers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for split and training setup')
    parser.add_argument('--mask-values', type=str, default=None, help='Comma-separated mask pixel values, e.g. 0,255, to skip startup mask scan')
    parser.add_argument('--class-weights', type=str, default=None, help='Comma-separated CrossEntropy weights, e.g. 1,14 for imbalanced binary masks')
    parser.add_argument('--no-progress', action='store_false', dest='progress', default=True, help='Disable tqdm progress bars for compact logs')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Disable benchmark kernels for stricter reproducibility')
    parser.add_argument('--augment', choices=['none', 'paper-light'], default='none', help='Training-set augmentation mode')
    parser.add_argument('--elastic-alpha', type=float, default=10.0, help='Elastic deformation displacement scale in pixels for paper-light augmentation')
    parser.add_argument('--elastic-grid-size', type=int, default=3, help='Coarse elastic deformation grid size for paper-light augmentation')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configure_torch_runtime(device, deterministic=args.deterministic)
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images, n_channels=1 for grayscale biomedical images
    # n_classes is the number of probabilities you want to get per pixel
    model = build_unet(args.architecture, n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{args.architecture} architecture\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        state_dict.pop('architecture', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            save_checkpoint=args.save_checkpoint,
            amp=args.amp,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            save_best=args.save_best,
            abort_on_nonfinite=args.abort_on_nonfinite,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            evals_per_epoch=args.evals_per_epoch,
            log_histograms=args.log_histograms,
            log_images=args.log_images,
            wandb_mode=args.wandb_mode,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            seed=args.seed,
            mask_values=parse_mask_values(args.mask_values),
            augment=args.augment,
            channels=args.channels,
            elastic_alpha=args.elastic_alpha,
            elastic_grid_size=args.elastic_grid_size,
            class_weights=parse_float_values(args.class_weights),
            progress=args.progress,
            weight_dir=args.weight_dir,
            architecture=args.architecture,
            loss_mode=args.loss_mode,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            save_checkpoint=args.save_checkpoint,
            amp=args.amp,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            save_best=args.save_best,
            abort_on_nonfinite=args.abort_on_nonfinite,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            evals_per_epoch=args.evals_per_epoch,
            log_histograms=args.log_histograms,
            log_images=args.log_images,
            wandb_mode=args.wandb_mode,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            seed=args.seed,
            mask_values=parse_mask_values(args.mask_values),
            augment=args.augment,
            channels=args.channels,
            elastic_alpha=args.elastic_alpha,
            elastic_grid_size=args.elastic_grid_size,
            class_weights=parse_float_values(args.class_weights),
            progress=args.progress,
            weight_dir=args.weight_dir,
            architecture=args.architecture,
            loss_mode=args.loss_mode,
        )
