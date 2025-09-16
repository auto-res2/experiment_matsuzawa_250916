import os
import logging
import requests
import tarfile
import hashlib
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
import glob

DATA_DIR = "./data"

def download_and_verify(url: str, dest_path: str, expected_sha256: str):
    if os.path.exists(dest_path):
        logging.info(f"File already exists: {dest_path}. Verifying checksum...")
        sha256_hash = hashlib.sha256()
        with open(dest_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        if sha256_hash.hexdigest() == expected_sha256:
            logging.info("Checksum OK.")
            return
        else:
            logging.warning("Checksum mismatch. Re-downloading.")
            os.remove(dest_path)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logging.info(f"Downloading {url} to {dest_path}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            sha256_hash = hashlib.sha256()
            with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    sha256_hash.update(chunk)
                    bar.update(len(chunk))
        
        if sha256_hash.hexdigest() != expected_sha256:
            os.remove(dest_path)
            raise ValueError(f"Checksum mismatch for {dest_path}. Expected {expected_sha256}, got {sha256_hash.hexdigest()}")
        logging.info("Download complete and verified.")
    except Exception as e:
        logging.error(f"Failed to download or verify {url}: {e}")
        if os.path.exists(dest_path): os.remove(dest_path)
        raise FileNotFoundError(f"Could not obtain required data file from {url}")

def extract_archive(archive_path: str, extract_dir: str):
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        logging.info(f"Data already extracted at {extract_dir}. Skipping extraction.")
        return
    logging.info(f"Extracting {archive_path} to {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

def prepare_all_data(data_manifest: dict):
    for name, info in data_manifest.items():
        archive_name = os.path.basename(info['url'])
        archive_path = os.path.join(DATA_DIR, '_archives', archive_name)
        extract_path = os.path.join(DATA_DIR, name)
        download_and_verify(info['url'], archive_path, info['sha256'])
        extract_path_final = os.path.join(DATA_DIR, info.get('extract_subdir', name))
        if not os.path.exists(extract_path_final) or not os.listdir(extract_path_final):
            extract_archive(archive_path, os.path.dirname(extract_path_final))
        else:
             logging.info(f"Data for {name} already exists at {extract_path_final}")

class ImageCorruptionDataset(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(root_dir, '**', '*.JPEG'), recursive=True) + glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True))
        if not self.image_files:
             raise FileNotFoundError(f"No images found in {root_dir}")
        # Dummy labels for now, as TTA is unsupervised
        self.labels = np.random.randint(0, 1000, len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class SyntheticStream(IterableDataset):
    def __init__(self, num_frames, num_classes, switch_interval):
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.switch_interval = switch_interval
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __iter__(self):
        for i in range(self.num_frames):
            mode = (i // self.switch_interval) % 4
            img_np = np.random.rand(224, 224, 3).astype(np.float32)
            if mode == 1: # Cauchy noise
                img_np += np.random.standard_cauchy(size=(224, 224, 3)) * 0.02
            elif mode == 2: # HSV shift
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np, 'RGB')
                img_hsv = np.array(img_pil.convert('HSV'))
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.5, 0, 255)
                img_pil = Image.fromarray(img_hsv, 'HSV').convert('RGB')
                img_np = np.array(img_pil) / 255.0
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            label = np.random.randint(0, self.num_classes)
            yield self.transform(img_pil), torch.tensor(label).long()

def get_data_stream(config: dict, data_manifest: dict) -> DataLoader:
    d_config = config['dataset']
    d_name = d_config['name']
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if 'exp2_synthetic' in config['experiment']:
        dataset = SyntheticStream(config['stream']['frames'], d_config['num_classes'], d_config['switch_interval'])
    else:
        manifest_key = next((k for k in data_manifest if k.lower() in d_name.lower()), None)
        if not manifest_key:
            raise ValueError(f"No manifest entry found for dataset '{d_name}'")
        base_path = os.path.join(DATA_DIR, data_manifest[manifest_key].get('extract_subdir', manifest_key))
        if 'ImageNet-C' in manifest_key or 'CIFAR10-C' in manifest_key:
            corruption_path = os.path.join(base_path, d_config['corruption'], str(d_config['severity']))
            dataset = ImageCorruptionDataset(corruption_path, transform)
        elif 'DomainNet' in manifest_key:
            domain_path = os.path.join(base_path, d_config['corruption']) # corruption field holds domain name
            dataset = ImageCorruptionDataset(domain_path, transform)
        else: # Basic loader for other datasets
            dataset = ImageCorruptionDataset(base_path, transform)

    return DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=isinstance(dataset, Dataset) and not isinstance(dataset, IterableDataset),
        num_workers=4,
        pin_memory=True
    )