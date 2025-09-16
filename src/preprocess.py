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
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_CACHE_DIR = os.path.expanduser("~/.cache/zorropp_data")

def download_and_extract(url, dest_path, extract_path):
    if os.path.exists(extract_path):
        logging.info(f"Dataset already exists at {extract_path}. Skipping download.")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logging.info(f"Downloading from {url} to {dest_path}")
    try:
        with requests.get(url, stream=True) as r:
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


class ImageCorruptionDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image'].convert("RGB")
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

class RealisticTTAStream(Dataset):
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

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders(config):
    dataloaders = {}
    image_transform = get_image_transforms()
    
    for exp_config in config['experiments']:
        d_name = exp_config['dataset']['name']
        d_corr = exp_config['dataset']['corruption']
        d_sev = exp_config['dataset']['severity']
        batch_size = exp_config['dataloader']['batch_size']
        run_name = f"{exp_config['exp_name']}_{exp_config['model']['name']}_{d_name}_{d_corr}_{exp_config['method']}_eta{exp_config['stream']['eta']}_seed{exp_config['seed']}"

        try:
            if 'cifar10-c' in d_name:
                dataset = load_dataset('hendrycks/cifar10_c', split=d_corr, cache_dir=DATA_CACHE_DIR)
                dataset = dataset.filter(lambda x: x['severity'] == d_sev)
                dataset = ImageCorruptionDataset(dataset, image_transform)
            elif 'imagenet-c' in d_name:
                dataset = load_dataset('hendrycks/imagenet-c', name=d_corr, split='validation', cache_dir=DATA_CACHE_DIR)
                dataset = dataset.filter(lambda x: x['severity'] == d_sev)
                dataset = ImageCorruptionDataset(dataset, image_transform)
            elif 'domainnet-c' in d_name:
                logging.warning("DomainNet-C not available on Hugging Face Hub. Skipping.")
                continue
            elif 'esc50-c' in d_name:
                logging.warning("ESC-50-C not available. Skipping.")
                continue
            elif 'edgehar-c' in d_name:
                # Using MobiAct from Zenodo as a stand-in for EdgeHAR-C
                record_id = '4646714' # MobiAct v2.0
                url = f"https://zenodo.org/api/records/{record_id}"
                response = requests.get(url).json()
                zip_url = [f['links']['self'] for f in response['files'] if f['key'] == 'MobiAct_Dataset_v2.0.zip'][0]
                dest_path = os.path.join(DATA_CACHE_DIR, 'edgehar-c.zip')
                extract_path = os.path.join(DATA_CACHE_DIR, 'edgehar-c')
                download_and_extract(zip_url, dest_path, extract_path)
                # Placeholder for actual dataset loading logic
                logging.warning("EdgeHAR-C (MobiAct) data loading not implemented. Using dummy data.")
                dataset = torch.utils.data.TensorDataset(torch.randn(1000, 3, 224, 224), torch.randint(0, 10, (1000,)))
            else:
                raise ValueError(f"Unknown dataset: {d_name}")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            dataloaders[run_name] = dataloader
        except Exception as e:
            logging.error(f"Failed to load dataset {d_name}/{d_corr}: {e}")
            logging.error("Please ensure you have network access and necessary permissions.")
            raise SystemExit(f"DATASET FAILED: {d_name}")
            
    return dataloaders
