import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets.folder import default_loader

from nilearn import datasets
from nilearn.plotting import plot_matrix, show
from nilearn.maskers import NiftiMasker

# We will NOT import tqdm since we want no progress bar.

from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as nn_utils  # for spectral_norm
import numpy as np
from PIL import Image

# ───────────────────────────────────────────────────────────────────────────────
# 1. Fetch Haxby stimuli via Nilearn
# ───────────────────────────────────────────────────────────────────────────────
from nilearn.datasets import fetch_haxby

print("1) Fetching Haxby stimuli via Nilearn…")
haxby_bunch  = fetch_haxby(subjects=[2], fetch_stimuli=True)
stimuli_dict = haxby_bunch.stimuli

print("   Classes:", list(stimuli_dict.keys()))
for cls_name, path_list in stimuli_dict.items():
    example_list = list(path_list)
    print(f"    {cls_name} (first 3 of {len(example_list)}):")
    for p in example_list[:3]:
        print(f"      → {p} (exists? {os.path.isfile(p)})")
    print()

# ───────────────────────────────────────────────────────────────────────────────
# 2. Hyperparameters & Configuration
# ───────────────────────────────────────────────────────────────────────────────
print("2) Setting hyperparameters & directories…")

# Target resolution: 64×64
IMAGE_SIZE   = 64
NC           = 3       # RGB
BATCH_SIZE   = 64
NUM_WORKERS  = 8
NUM_EPOCHS   = 1000     # increase later if needed

# Latent dims
NZ           = 100     # noise vector size
EMBED_DIM    = 64      # class embedding dimension

# Architecture capacity
NGF          = 256     # base number of feature maps in G
NDF          = 256     # base number of feature maps in D

# Learning rates & optimizer params
LR_D         = 5e-5
LR_G         = 1e-4
BETA1        = 0.5

# One-sided label smoothing
REAL_LABEL_SMOOTH = 0.9
FAKE_LABEL_VAL    = 0.0

# Random seeds
MANUAL_SEED  = 42
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

# Device
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {DEVICE}")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(MANUAL_SEED)
    print(f"     GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("     (No GPU detected; using CPU)")

# Directories for samples & checkpoints
CHECKPOINTS  = "./checkpoints"
SAMPLES_DIR  = "./samples"
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
print(f"   Checkpoints → {CHECKPOINTS}")
print(f"   Samples     → {SAMPLES_DIR}\n")

# ───────────────────────────────────────────────────────────────────────────────
# 3. Helper: Denormalize a batch from [-1,1] → [0,1]
# ───────────────────────────────────────────────────────────────────────────────
def denormalize_batch(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5) + 0.5

# ───────────────────────────────────────────────────────────────────────────────
# 4. Custom Dataset: wrap Haxby stimuli, filter out invalid filepaths
# ───────────────────────────────────────────────────────────────────────────────

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Fix for truncated images

class HaxbyStimuliDataset(Dataset):
    def __init__(self, stimuli_dict, transform=None):
        super(HaxbyStimuliDataset, self).__init__()
        self.transform    = transform
        self.stimuli_dict = stimuli_dict

        scrambled_stimuli = list(stimuli_dict['controls']['scrambled_bottles']) \
                    + list(stimuli_dict['controls']['scrambled_cats']) \
                    + list(stimuli_dict['controls']['scrambled_chairs']) \
                    + list(stimuli_dict['controls']['scrambled_faces']) \
                    + list(stimuli_dict['controls']['scrambled_houses']) \
                    + list(stimuli_dict['controls']['scrambled_scissors']) \
                    + list(stimuli_dict['controls']['scrambled_shoes'])

        def img_from_stimulus(stimulus):
            if stimulus == "scrambledpix":
                out =  Image.open(np.random.choice(scrambled_stimuli))
            else:
                key = stimulus + "s" if stimulus != "scissors" else stimulus
                out = Image.open(stimuli_dict[key][0])
            return out

        print("3) Building sample list for Dataset…")
        # self.samples = []
        # self.stimuli = []
        func_filename = haxby_bunch.func[0]
        mask_filename = haxby_bunch.mask
        labels_df = pd.read_csv(haxby_bunch.session_target[0], sep=' ')
        y = labels_df['labels'].values
        run = labels_df['chunks'].values

        runs_full = labels_df['chunks'].values

        masker = NiftiMasker(
            mask_img=mask_filename,
            standardize='zscore_sample',
            runs=runs_full,
            smoothing_fwhm=4,
            memory='nilearn_cache',
            memory_level=1
        )
        X_full = masker.fit_transform(func_filename)
        mask_nonrest = labels_df['labels'].values != 'rest'
        self.samples = X_full[mask_nonrest]
        self.classes = np.unique(y[mask_nonrest]).tolist()
        self.stimuli = [img_from_stimulus(i) for i in labels_df['labels'].values[mask_nonrest]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # img_path, cls_name = self.samples[idx]
        # img = default_loader(img_path)          # returns a PIL.Image
        # if self.transform:
        #     img = self.transform(img)           # apply transforms
        # # Convert class name → integer index
        # class_idx = self.classes.index(cls_name)
        # return img, class_idx
        img = self.stimuli[idx]
        if self.transform:
            img = self.transform(img)               # apply transforms
        return img, self.samples[idx]

# ───────────────────────────────────────────────────────────────────────────────
# 5. Build DataLoader + transforms (400×400 → 64×64)
# ───────────────────────────────────────────────────────────────────────────────
print("4) Setting up DataLoader + transforms…")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # convert to RGB
    transforms.Resize(IMAGE_SIZE),        # downsample from 400×400 → 64×64
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),               # [0,1]
    transforms.Normalize((0.5,)*NC, (0.5,)*NC)  # [−1,1]
])

haxby_dataset = HaxbyStimuliDataset(stimuli_dict, transform=transform)
dataloader     = DataLoader(
    haxby_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

print(f"    Dataset size       = {len(haxby_dataset)} images")
print(f"    Number of classes  = {len(haxby_dataset.classes)}\n")

# Dynamically set CLASS_DIM = number of classes
NUM_CLASSES = len(haxby_dataset.classes)
CLASS_DIM   = NUM_CLASSES

# ───────────────────────────────────────────────────────────────────────────────
# 5b. Set up latent space classifier
# ───────────────────────────────────────────────────────────────────────────────

LATENT_CLF_INPUT_DIM = 39912  # 400×400 image flattened
LATENT_CLF_LATENT_DIM = 64

print("5) Setting up latent space classifier (CVAE)...")
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, dropout=0.2):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, 256)
        self.enc_bn1 = nn.BatchNorm1d(256)
        self.enc_fc2 = nn.Linear(256, 128)
        self.enc_bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc_mu    = nn.Linear(128, latent_dim)
        self.fc_logvar= nn.Linear(128, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.dec_fc1 = nn.Linear(latent_dim + num_classes, 128)
        self.dec_bn1 = nn.BatchNorm1d(128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_bn2 = nn.BatchNorm1d(256)
        self.dec_out = nn.Linear(256, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_bn1(self.enc_fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.enc_bn2(self.enc_fc2(h)))
        h = self.dropout(h)
        mu      = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def decode(self, z, y_onehot):
        z_cond = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.dec_bn1(self.dec_fc1(z_cond)))
        h = self.dropout(h)
        h = F.relu(self.dec_bn2(self.dec_fc2(h)))
        h = self.dropout(h)
        return self.dec_out(h)

    def forward(self, x, y_onehot):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y_onehot)
        logits = self.classifier(z)
        return recon_x, logits, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
latent_clf = CVAE(LATENT_CLF_INPUT_DIM, LATENT_CLF_LATENT_DIM, 8)
latent_clf.load_state_dict(torch.load('best_model.pth'))
latent_clf.eval()  # Set to evaluation mode
latent_clf = latent_clf.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ───────────────────────────────────────────────────────────────────────────────
# 6. Conditional Generator & Discriminator (64×64) w/ embeddings & spectral-norm
# ───────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────
# Helper: Adaptive Instance Normalization
# ───────────────────────────────────────────────────────────────────────────────
class AdaIN(nn.Module):
    def __init__(self, num_channels: int, embed_dim: int):
        super(AdaIN, self).__init__()
        # Two separate linear layers to produce scale (gamma) and bias (beta)
        self.fc_gamma = nn.Linear(embed_dim, num_channels)
        self.fc_beta  = nn.Linear(embed_dim, num_channels)

    def forward(self, x: torch.Tensor, class_embed: torch.Tensor) -> torch.Tensor:
        """
        x:            (batch, C, H, W)
        class_embed:  (batch, embed_dim)
        Returns: AdaIN‐normalized and re‐scaled x
        """
        batch, C, H, W = x.size()
        # Compute per‐channel mean & std of x
        x_reshaped = x.view(batch, C, -1)  # (batch, C, H*W)
        mean_x     = x_reshaped.mean(dim=2).view(batch, C, 1, 1)
        std_x      = x_reshaped.std(dim=2).view(batch, C, 1, 1) + 1e-5

        x_norm = (x - mean_x) / std_x  # Normalize

        # Produce gamma & beta from class embedding
        gamma = self.fc_gamma(class_embed).view(batch, C, 1, 1)
        beta  = self.fc_beta(class_embed).view(batch, C, 1, 1)

        return gamma * x_norm + beta

# ───────────────────────────────────────────────────────────────────────────────
# ▪︎ CHANGE: Add class embeddings at every layer via AdaIN
# ───────────────────────────────────────────────────────────────────────────────
class ConditionalGenerator(nn.Module):
    def __init__(self, nz: int, latent_dim: int, ngf: int, nc: int):
        super(ConditionalGenerator, self).__init__()
        self.nz        = nz
        self.latent_dim = latent_dim
        self.ngf       = ngf
        self.nc        = nc

        # Project (z ∥ embed) → (ngf*8) × 4 × 4
        self.project = nn.Sequential(
            nn.Linear(self.nz + self.latent_dim, self.ngf * 8 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # Four upsampling blocks (4→8, 8→16, 16→32, 32→64)
        # Remove BatchNorm and insert AdaIN after each ConvTranspose2d
        self.deconv1 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)
        self.adain1  = AdaIN(self.ngf * 4, self.latent_dim)

        self.deconv2 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)
        self.adain2  = AdaIN(self.ngf * 2, self.latent_dim)

        self.deconv3 = nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False)
        self.adain3  = AdaIN(self.ngf, self.latent_dim)

        self.deconv4 = nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False)
        # No AdaIN on final layer; just Tanh

    def forward(self, noise: torch.Tensor, latent_space: torch.Tensor) -> torch.Tensor:
        """
        noise:      (batch_size, nz)
        class_idxs: (batch_size,) long
        → returns:  (batch_size, nc, 64, 64)
        """

        # 2) Concatenate noise + latent_space → projected feature
        x = torch.cat([noise, latent_space], dim=1)       # (batch, nz + embed_dim)
        x = self.project(x)                               # (batch, ngf*8*4*4)
        x = x.view(-1, self.ngf * 8, 4, 4)                # (batch, ngf*8, 4, 4)

        # 3) First upsample: 4 → 8
        x = self.deconv1(x)                               # (batch, ngf*4, 8, 8)
        x = self.adain1(x, latent_space)                  # AdaIN
        x = nn.ReLU(inplace=True)(x)

        # 4) Second upsample: 8 → 16
        x = self.deconv2(x)                               # (batch, ngf*2, 16, 16)
        x = self.adain2(x, latent_space)
        x = nn.ReLU(inplace=True)(x)

        # 5) Third upsample: 16 → 32
        x = self.deconv3(x)                               # (batch, ngf, 32, 32)
        x = self.adain3(x, latent_space)
        x = nn.ReLU(inplace=True)(x)

        # 6) Fourth upsample: 32 → 64
        x = self.deconv4(x)                               # (batch, nc, 64, 64)
        out = torch.tanh(x)                               # output in [−1, +1]
        return out


class ConditionalDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, latent_dim: int):
        super(ConditionalDiscriminator, self).__init__()
        self.nc        = nc
        self.ndf       = ndf
        self.latent_dim = latent_dim

        # Feature extractor: 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1
        self.feature_extractor = nn.Sequential(
            # 64 → 32
            nn_utils.spectral_norm(nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            # 32 → 16
            nn_utils.spectral_norm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            # 16 → 8
            nn_utils.spectral_norm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            # 8 → 4
            nn_utils.spectral_norm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 → 1 (output has shape (batch, 1, 1, 1))
            nn_utils.spectral_norm(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False))
            # No activation (raw score map)
        )
        # After flatten, we get shape (batch, 1). Then we'll concatenate with the class embedding (embed_dim)
        # to produce (batch, 1 + embed_dim) → final linear → 1 logit.
        self.classifier = nn_utils.spectral_norm(nn.Linear(1 + self.latent_dim, 1))

    def forward(self, img: torch.Tensor, latent_space: torch.Tensor) -> torch.Tensor:
        """
        img:        (batch_size, nc, 64, 64)
        class_idxs: (batch_size,) long
        → returns:  (batch_size,) raw logits
        """
        feat_map = self.feature_extractor(img)          # (batch, 1, 1, 1)
        feat = feat_map.view(feat_map.size(0), -1)      # (batch, 1)
        combined = torch.cat([feat, latent_space], dim=1) # (batch, 1 + latent_dim)
        return self.classifier(combined).view(-1)

# ───────────────────────────────────────────────────────────────────────────────
# 7. Weight Initialization (DCGAN best practice, plus Linear)
# ───────────────────────────────────────────────────────────────────────────────
def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# ───────────────────────────────────────────────────────────────────────────────
# 8. Training Loop (64×64, LSGAN, embeddings, mixed-precision, NO tqdm)
# ───────────────────────────────────────────────────────────────────────────────
def train_conditional_dcgan(
    dataloader, nz, embed_dim, ngf, ndf, nc,
    lr_d, lr_g, beta1, num_epochs, device,
    sample_interval_epochs=5, checkpoint_interval_epochs=10
):
    """
    Train a conditional DCGAN on Haxby stimuli at 64×64 resolution.
    - Mixed precision (torch.cuda.amp)
    - LSGAN (MSELoss) with one-sided label smoothing (real=0.9, fake=0.0)
    - Spectral normalization in D + embedding layers in both G & D
    - Save sample grids & checkpoints every few epochs
    - No tqdm: just print at epoch boundaries
    """

    print("8) Initializing Generator and Discriminator…")
    netG = ConditionalGenerator(nz, embed_dim, ngf, nc).to(device)
    netD = ConditionalDiscriminator(nc, ndf, embed_dim).to(device)

    # Initialize weights
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    # LSGAN criterion
    criterion  = nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    real_label_val = REAL_LABEL_SMOOTH
    fake_label_val = FAKE_LABEL_VAL

    use_amp = (device.type == "cuda")
    scalerD = GradScaler(enabled=use_amp)
    scalerG = GradScaler(enabled=use_amp)

    # Prepare fixed noise + class indices for sample visualization (64 total)
    fixed_noise = torch.randn(64, nz, device=device)
    images, brain_scans = next(iter(dataloader))
    b_size = images.size(0)
    all_indices = np.arange(b_size)
    np.random.shuffle(all_indices)
    fixed_sample_choices = all_indices[:64]

    with torch.no_grad():
        fixed_latent_spaces = []
        for i in fixed_sample_choices:
            # brain_scans[i] has shape [39912]; add a batch dimension → [1, 39912]
            x = brain_scans[i].unsqueeze(0).to(device)
            mu, _ = latent_clf.encode(x)
            fixed_latent_spaces.append(mu.detach())

    fixed_latent_spaces = torch.cat(fixed_latent_spaces, dim=0)  # shape [64, latent_dim]

    print("   Fixed noise & class indices built for sampling.\n")

    for epoch in range(1, num_epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches  = 0

        for real_images, brain_scans in dataloader:
            real_images = real_images.to(device)      # (batch, 3, 64, 64)
            brain_scans  = brain_scans.to(device)       # (batch,)
            b_size      = real_images.size(0)
            num_batches += 1

            # ────────────── Train Discriminator ──────────────
            netD.zero_grad()
            # Labels
            label_real = torch.full((b_size,), real_label_val, device=device)
            label_fake = torch.full((b_size,), fake_label_val, device=device)

            # Convert brain scans to latent space
            with torch.no_grad():
                mu, _ = latent_clf.encode(brain_scans)  # (batch, latent_dim)
                latent_space = mu  # (batch, latent_dim)

            # (a) Real pass
            with autocast(enabled=use_amp):
                real_logits = netD(real_images, latent_space)       # (batch,)
                errD_real   = criterion(real_logits, label_real)
            scalerD.scale(errD_real).backward()

            # (b) Fake pass (detach G)
            noise = torch.randn(b_size, nz, device=device)
            with autocast(enabled=use_amp):
                fake_images = netG(noise, latent_space)             # (batch, 3, 64, 64)
                fake_logits = netD(fake_images.detach(), latent_space)
                errD_fake   = criterion(fake_logits, label_fake)
            scalerD.scale(errD_fake).backward()
            scalerD.step(optimizerD)
            scalerD.update()

            # ────────────── Train Generator ──────────────
            netG.zero_grad()
            with autocast(enabled=use_amp):
                fake_logits_forG = netD(fake_images, latent_space)   # (batch,)
                # For G, we want D(fake) → “real” (0.9)
                label_gen        = torch.full((b_size,), real_label_val, device=device)
                errG             = criterion(fake_logits_forG, label_gen)
            scalerG.scale(errG).backward()
            scalerG.step(optimizerG)
            scalerG.update()

            # Accumulate losses
            epoch_d_loss += (errD_real.item() + errD_fake.item())
            epoch_g_loss += errG.item()

        # ────────────── End of epoch: compute averages ──────────────
        avg_d = epoch_d_loss / num_batches
        avg_g = epoch_g_loss / num_batches
        print(f"[Epoch {epoch:03d}] Avg_Loss_D: {avg_d:.4f}, Avg_Loss_G: {avg_g:.4f}")

        # ────────────── Save sample grid every N epochs ──────────────
        if epoch % sample_interval_epochs == 0 or epoch == 1:
            print(f"   → Saving fake sample grid at epoch {epoch}…")
            with torch.no_grad():
                fake_samples = netG(fixed_noise, fixed_latent_spaces).detach().cpu()
            fake_grid = vutils.make_grid(
                denormalize_batch(fake_samples),
                padding=2, normalize=False, nrow=8
            )
            fake_path = Path(SAMPLES_DIR) / f"fake_epoch_{epoch:04d}.png"
            vutils.save_image(fake_grid, str(fake_path))
            print(f"      • Saved fake samples → {fake_path}")

        # ────────────── Save checkpoints every N epochs ──────────────
        if epoch % checkpoint_interval_epochs == 0:
            print(f"   → Saving model checkpoints at epoch {epoch}…")
            ckpt_G = Path(CHECKPOINTS) / f"netG_epoch_{epoch:04d}.pth"
            ckpt_D = Path(CHECKPOINTS) / f"netD_epoch_{epoch:04d}.pth"
            torch.save(netG.state_dict(), str(ckpt_G))
            torch.save(netD.state_dict(), str(ckpt_D))
            print(f"      • Saved Generator → {ckpt_G}")
            print(f"      • Saved Discriminator → {ckpt_D}")

    print("\n***** Training Complete *****")

# ───────────────────────────────────────────────────────────────────────────────
# 9. Main: Launch Training
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("9) Launching training…\n")
    train_conditional_dcgan(
        dataloader=dataloader,
        nz=NZ,
        embed_dim=EMBED_DIM,
        ngf=NGF,
        ndf=NDF,
        nc=NC,
        lr_d=LR_D,
        lr_g=LR_G,
        beta1=BETA1,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        sample_interval_epochs=25,      # save samples every 25 epochs
        checkpoint_interval_epochs=25   # save checkpoints every 25 epochs
    )