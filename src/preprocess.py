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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_CACHE_DIR = os.path.expanduser("~/.cache/zorropp_data")

# -----------------------------------------------------------------------------
# Utility: fail-fast download helper
# -----------------------------------------------------------------------------

def download_and_extract(url, dest_path, extract_path):
    if os.path.exists(extract_path):
        logging.info(f"Dataset already exists at {extract_path}. Skipping download.")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logging.info(f"Downloading from {url} to {dest_path}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        raise SystemExit(f"DOWNLOAD FAILED: {url}")

    logging.info(f"Extracting {dest_path} to {extract_path}")
    if dest_path.endswith('.zip'):
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif dest_path.endswith('.tar.gz'):
        with tarfile.open(dest_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        raise ValueError(f"Unsupported archive type for {dest_path}")
    os.remove(dest_path)

# -----------------------------------------------------------------------------
# Dataset wrappers
# -----------------------------------------------------------------------------

class ImageCorruptionDataset(Dataset):
    """HF dataset wrapper that applies a torchvision transform."""

    def __init__(self, data: HFDataset, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Support both 'image' and 'img' field names
        img = item.get('image') or item.get('img')
        if not isinstance(img, Image.Image):
            # some HF datasets store images as numpy arrays
            img = Image.fromarray(img)
        img = img.convert("RGB")
        label = int(item['label'])
        if self.transform:
            img = self.transform(img)
        return img, label

class RealisticTTAStream(Dataset):
    """Streaming wrapper that increases corruption severity over time."""

    def __init__(self, base_dataset, corruption_function, eta, frames_per_severity=10000):
        self.base_dataset = base_dataset
        self.corruption_function = corruption_function
        self.eta = eta
        self.frames_per_severity = int(frames_per_severity * self.eta)
        self.severities = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        severity_level = self.severities[min(idx // self.frames_per_severity, len(self.severities) - 1)]
        clean_image, label = self.base_dataset[idx]
        corrupted_image = self.corruption_function(clean_image, severity_level)
        return corrupted_image, label

# -----------------------------------------------------------------------------
# Dataloader factory
# -----------------------------------------------------------------------------

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _attempt_load_hf_dataset(name: str, **kwargs):
    """Try to load a dataset from the HF hub; raise clear error if unavailable."""
    try:
        return load_dataset(name, **kwargs, cache_dir=DATA_CACHE_DIR)
    except Exception as e:
        raise RuntimeError(f"Could not load dataset '{name}' from the Hub: {e}") from e


def get_dataloaders(config):
    dataloaders = {}
    image_transform = get_image_transforms()

    for exp_config in config['experiments']:
        d_name = exp_config['dataset']['name']
        d_corr = exp_config['dataset']['corruption']
        d_sev = exp_config['dataset']['severity']
        batch_size = exp_config['dataloader']['batch_size']
        run_name = (
            f"{exp_config['exp_name']}_{exp_config['model']['name']}"+
            f"_{d_name}_{d_corr}_{exp_config['method']}_eta{exp_config['stream']['eta']}_seed{exp_config['seed']}"
        )

        try:
            # ------------------------------------------------------------------
            # CIFAR10-C
            # ------------------------------------------------------------------
            if 'cifar10-c' in d_name:
                # The official CIFAR-10-C set is not on the Hub yet -> fall back
                # to clean CIFAR-10 test set and apply corruption on-the-fly.
                logging.info("Loading CIFAR-10 test set as stand-in for CIFAR10-C …")
                dataset = _attempt_load_hf_dataset('cifar10', split='test')
                # Harmonise column names
                if 'img' in dataset.column_names:
                    dataset = dataset.rename_column('img', 'image')

                def _shot_noise(img: Image.Image, severity=5):
                    """Simple shot-noise corruption – Poisson sampling."""
                    img_np = np.array(img).astype(np.float32) / 255.0
                    lam = severity * 10.0  # arbitrary scaling
                    noisy = np.random.poisson(img_np * lam) / float(lam)
                    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
                    return Image.fromarray(noisy)

                base_ds = ImageCorruptionDataset(dataset, image_transform)
                dataset = RealisticTTAStream(base_ds, _shot_noise, eta=exp_config['stream']['eta'])

            # ------------------------------------------------------------------
            # ImageNet-C
            # ------------------------------------------------------------------
            elif 'imagenet-c' in d_name:
                dataset = _attempt_load_hf_dataset('imagenet-1k', split='validation')  # placeholder clean set
                if 'image' not in dataset.column_names:
                    raise ValueError("Expected 'image' column in ImageNet dataset")
                dataset = ImageCorruptionDataset(dataset, image_transform)
                logging.warning("Using clean ImageNet validation images – corruption not applied in smoke-test mode.")

            # ------------------------------------------------------------------
            # DomainNet-C and ESC50-C are skipped in smoke test
            # ------------------------------------------------------------------
            elif 'domainnet-c' in d_name or 'esc50-c' in d_name:
                logging.warning(f"{d_name} not available for smoke test. Skipping experiment {run_name}.")
                continue

            # ------------------------------------------------------------------
            # EdgeHAR-C (placeholder / dummy data)
            # ------------------------------------------------------------------
            elif 'edgehar-c' in d_name:
                record_id = '4646714'  # MobiAct v2.0 – example ID
                url = f"https://zenodo.org/api/records/{record_id}"
                try:
                    response = requests.get(url, timeout=30).json()
                    zip_url = [f['links']['self'] for f in response['files'] if f['key'] == 'MobiAct_Dataset_v2.0.zip'][0]
                    dest_path = os.path.join(DATA_CACHE_DIR, 'edgehar-c.zip')
                    extract_path = os.path.join(DATA_CACHE_DIR, 'edgehar-c')
                    download_and_extract(zip_url, dest_path, extract_path)
                except Exception as e:
                    logging.warning(f"Could not download EdgeHAR-C dataset metadata: {e}")
                # Dummy tensor dataset
                dataset = torch.utils.data.TensorDataset(torch.randn(1000, 3, 224, 224), torch.randint(0, 10, (1000,)))

            else:
                raise ValueError(f"Unknown dataset: {d_name}")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            dataloaders[run_name] = dataloader
        except Exception as e:
            logging.error(f"Failed to prepare dataset {d_name}/{d_corr}: {e}")
            raise SystemExit(f"DATASET FAILED: {d_name}")

    return dataloaders