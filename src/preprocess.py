import os
import logging
import requests
import zipfile
import tarfile
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset, Dataset as HFDataset

# Local util imported from train.py to generate identical run names
from .train import make_run_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_CACHE_DIR = os.path.expanduser("~/.cache/zorropp_data")

# -----------------------------------------------------------------------------
# Utility helpers (download & transforms)
# -----------------------------------------------------------------------------

def download_and_extract(url, dest_path, extract_path):
    if os.path.exists(extract_path):
        logging.info(f"Dataset already exists at {extract_path}. Skipping download.")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logging.info(f"Downloading {url} → {dest_path}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(total=total, unit='iB', unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk); bar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        raise SystemExit("DOWNLOAD FAILED")
    logging.info(f"Extracting {dest_path}")
    if dest_path.endswith('.zip'):
        with zipfile.ZipFile(dest_path, 'r') as zf: zf.extractall(extract_path)
    elif dest_path.endswith('.tar.gz'):
        with tarfile.open(dest_path, 'r:gz') as tf_: tf_.extractall(extract_path)
    else:
        raise ValueError("Unsupported archive type")
    os.remove(dest_path)


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _attempt_load_hf_dataset(name: str, **kwargs):
    try:
        return load_dataset(name, **kwargs, cache_dir=DATA_CACHE_DIR)
    except Exception as e:
        raise RuntimeError(f"Could not load dataset '{name}' from the Hub: {e}") from e


# -----------------------------------------------------------------------------
# Minimal dataset wrappers (unchanged)
# -----------------------------------------------------------------------------

class ImageCorruptionDataset(Dataset):
    def __init__(self, data: HFDataset, transform):
        self.data, self.transform = data, transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item.get('image') or item.get('img')
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        label = int(item['label'])
        return self.transform(img), label

class RealisticTTAStream(Dataset):
    def __init__(self, base_dataset, corruption_fn, eta, frames_per_severity=10000):
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.eta = eta
        self.frames_per_sev = int(frames_per_severity * eta)
        self.sev_levels = [1,2,3,4,5]
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        sev = self.sev_levels[min(idx // self.frames_per_sev, len(self.sev_levels)-1)]
        clean, lbl = self.base_dataset[idx]
        return self.corruption_fn(clean, sev), lbl

# -----------------------------------------------------------------------------
# Main factory
# -----------------------------------------------------------------------------

def get_dataloaders(config):
    dataloaders = {}
    transform = get_image_transforms()

    for exp_cfg in config['experiments']:
        # Skip template entries that miss 'corruption'
        if 'corruption' not in exp_cfg['dataset']:
            logging.info("Encountered template config without 'corruption'. Skipping …")
            continue

        try:
            run_name = make_run_name(exp_cfg)
        except ValueError:
            logging.warning("Could not create run-name – treating as template. Skipping.")
            continue

        d_name = exp_cfg['dataset']['name']
        batch_size = exp_cfg['dataloader']['batch_size']

        try:
            # CIFAR10-C placeholder
            if 'cifar10-c' in d_name:
                ds = _attempt_load_hf_dataset('cifar10', split='test')
                if 'img' in ds.column_names:
                    ds = ds.rename_column('img', 'image')
                def _shot_noise(im: Image.Image, severity=5):
                    im_np = np.array(im).astype(np.float32)/255.0
                    lam = severity*10.0
                    noisy = np.random.poisson(im_np*lam)/lam
                    noisy = np.clip(noisy*255.0,0,255).astype(np.uint8)
                    return Image.fromarray(noisy)
                base = ImageCorruptionDataset(ds, transform)
                dataset = RealisticTTAStream(base, _shot_noise, exp_cfg['stream']['eta'])
            # ImageNet placeholder
            elif 'imagenet-c' in d_name:
                ds = _attempt_load_hf_dataset('imagenet-1k', split='validation')
                dataset = ImageCorruptionDataset(ds, transform)
                logging.warning("Smoke-test: using clean ImageNet-val without additional corruptions.")
            elif 'edgehar-c' in d_name:
                # Dummy tensor dataset for CI
                dataset = torch.utils.data.TensorDataset(torch.randn(1000,3,224,224), torch.randint(0,10,(1000,)))
            else:
                raise ValueError(f"Unsupported dataset: {d_name}")
        except Exception as e:
            logging.error(f"Failed preparing dataset {d_name}: {e}")
            raise SystemExit(f"DATASET FAILED: {d_name}")

        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataloaders[run_name] = dl

    return dataloaders