import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise FileNotFoundError(f"Failed to download required file: {url}")

def get_bert_embeddings(texts, model_name, device, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu()) # CLS token
    return torch.cat(embeddings, dim=0)

def compute_structural_features(data, sample_size=5000):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Log-degree for all nodes
    degrees = torch.bincount(edge_index[0], minlength=num_nodes) + torch.bincount(edge_index[1], minlength=num_nodes)
    log_degrees = torch.log(degrees.float() + 1)

    # Jaccard and Hop-distance on a sample of edges for scalability
    num_edges = edge_index.size(1)
    if num_edges > sample_size:
        sample_indices = torch.randperm(num_edges)[:sample_size]
        sampled_edge_index = edge_index[:, sample_indices]
    else:
        sampled_edge_index = edge_index
        sample_indices = torch.arange(num_edges)

    # Build adjacency list for Jaccard
    adj = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        u, v = edge_index[:, i].tolist()
        adj[u].add(v)
        adj[v].add(u)

    jaccard_coeffs = []
    for i in range(sampled_edge_index.size(1)):
        u, v = sampled_edge_index[:, i].tolist()
        intersection = len(adj[u].intersection(adj[v]))
        union = len(adj[u].union(adj[v]))
        jaccard_coeffs.append(intersection / union if union > 0 else 0)

    # Hop-distance using SciPy's shortest_path on the full graph (can be slow)
    graph_sparse = coo_matrix((np.ones(num_edges), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(num_nodes, num_nodes))
    dist_matrix = shortest_path(csgraph=graph_sparse, directed=False, limit=3, unweighted=True)
    
    hop_dists = []
    for i in range(sampled_edge_index.size(1)):
        u, v = sampled_edge_index[:, i].tolist()
        dist = dist_matrix[u, v]
        hop_dists.append(dist if np.isfinite(dist) else 4) # 4 for >3 hops

    s_ij = torch.zeros(num_edges, 4)
    s_ij[:, 0] = log_degrees[edge_index[0]]
    s_ij[:, 1] = log_degrees[edge_index[1]]
    s_ij[sample_indices, 2] = torch.tensor(jaccard_coeffs, dtype=torch.float)
    s_ij[sample_indices, 3] = torch.tensor(hop_dists, dtype=torch.float) / 3.0 # Normalize

    return s_ij

def process_ogbn_papers100m(config, output_dir):
    dataset_name = config['name']
    output_path = os.path.join(output_dir, f"{dataset_name}.pt")
    if os.path.exists(output_path):
        print(f'Processed data for {dataset_name} found.')
        return
    
    print(f'Processing {dataset_name}...')
    dataset = PygNodePropPredDataset(name=dataset_name, root=config['path'])
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, split_idx['train'], True)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, split_idx['valid'], True)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, split_idx['test'], True)
    
    # Normalize features
    data.x = F.normalize(data.x, p=2., dim=-1)
    
    # Add self loops
    data.edge_index, _ = add_self_loops(data.edge_index)
    
    print('Computing structural features for ogbn-papers100M (this may take a while)...')
    data.s_ij = compute_structural_features(data, sample_size=10000)

    torch.save(data, output_path)
    print(f'Saved processed data to {output_path}')

def process_generic_graph(config, output_dir, device):
    dataset_name = config['name']
    output_path = os.path.join(output_dir, f"{dataset_name}.pt")
    if os.path.exists(output_path):
        print(f'Processed data for {dataset_name} found.')
        return

    raw_path = config['path']
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Please download/place it there. This experiment cannot run without it.")

    print(f'Processing {dataset_name}...')
    # This is a generic loader. Real implementation would be specific.
    # Assuming CSV with 'source', 'target', 'text', 'label' columns
    df = pd.read_csv(raw_path)
    
    # Create node mapping
    nodes = pd.concat([df['source'], df['target']]).unique()
    node_map = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # Create edge index
    source = df['source'].map(node_map).values
    target = df['target'].map(node_map).values
    edge_index = torch.tensor([source, target], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Create features
    if 'text' in df.columns:
        node_texts = df.groupby(df['source'].map(node_map))['text'].first().reindex(range(num_nodes), fill_value='').tolist()
        x = get_bert_embeddings(node_texts, config['feature_model'], device)
        x = x.to(torch.float16) # Save memory
    else:
        # Dummy features if no text
        x = torch.randn(num_nodes, 128)
    
    # Create labels and masks
    if 'label' in df.columns:
        labels_map = {label: i for i, label in enumerate(df['label'].unique())}
        labels = torch.full((num_nodes,), -1, dtype=torch.long)
        node_labels = df.drop_duplicates('source').set_index('source')['label'].map(labels_map)
        labels[node_labels.index.map(node_map)] = torch.tensor(node_labels.values, dtype=torch.long)
    else:
        labels = torch.randint(0, 2, (num_nodes,))

    # Create splits
    perm = torch.randperm(num_nodes)
    train_end = int(0.6 * num_nodes)
    val_end = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, perm[:train_end], True)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, perm[train_end:val_end], True)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, perm[val_end:], True)

    data = Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    print(f'Computing structural features for {dataset_name}...')
    data.s_ij = compute_structural_features(data, sample_size=5000)

    torch.save(data, output_path)
    print(f'Saved processed data to {output_path}')

def generate_dummy_psi(output_dir, in_features=4, out_features=256):
    # Create a dummy pre-trained hyper-network Ψ
    psi_path = os.path.join(output_dir, 'psi.pt')
    if not os.path.exists(psi_path):
        print("Generating dummy pre-trained hyper-network psi.pt")
        psi_model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features)
        )
        torch.save(psi_model, psi_path)


def main(config):
    output_dir = config['preprocess']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create the dummy hyper-network needed by META-LEAP
    generate_dummy_psi(output_dir, out_features=config['experiments']['exp1_throughput']['model_params']['hidden_channels'] * 2)

    for ds_config in config['preprocess']['datasets']:
        if ds_config['type'] == 'ogb':
            ds_config['path'] = raw_dir
            process_ogbn_papers100m(ds_config, output_dir)
        elif ds_config['type'] == 'generic':
            # For smoke test, generate a dummy file if it doesn't exist
            if config.get('is_smoke_test', False) and not os.path.exists(ds_config['path']):
                print(f"Smoke test mode: generating dummy data for {ds_config['name']}")
                num_nodes, num_edges = 500, 2000
                src = np.random.randint(0, num_nodes, num_edges)
                dst = np.random.randint(0, num_nodes, num_edges)
                text = ['sample text'] * num_edges
                label = np.random.randint(0, 5, num_edges)
                pd.DataFrame({'source': src, 'target': dst, 'text': text, 'label': label}).to_csv(ds_config['path'], index=False)
            
            process_generic_graph(ds_config, output_dir, device)
