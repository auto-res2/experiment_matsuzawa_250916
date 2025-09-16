import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pykalman import KalmanFilter
from sklearn.linear_model import BayesianRidge
from transformers import AutoModel, AutoTokenizer
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# --- LLM API Client (Simulated) ---
class LLMAPIClient:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        # In a real scenario, you'd check if the key is valid.
        # if not self.api_key:
        #     raise ValueError("API key for OpenAI or HuggingFace must be set in environment variables.")

    def get_score(self, prompt):
        """Simulates getting a helpfulness score, safety score, and token heatmap."""
        # Simulate API latency
        time.sleep(0.1)
        helpfulness = np.random.rand() * (0.8 - 0.2) + 0.2  # Score between 0.2 and 0.8
        severity_cost = np.random.rand() * 0.1 # Low severity cost
        # Simulate a heatmap based on token length
        num_tokens = len(prompt.split())
        heatmap = np.random.rand(1, 1, 32, 32) # Simulate a fixed size heatmap for the CNN
        return {"helpfulness": helpfulness, "severity_cost": severity_cost, "heatmap": torch.tensor(heatmap, dtype=torch.float32)}

    def get_paraphrases(self, prompt, n=5):
        """Simulates getting paraphrases from Vicuna."""
        return [f"{prompt} (paraphrase {i+1})" for i in range(n)]

# --- Token-Risk CNN ---
class TokenRiskCNN(nn.Module):
    def __init__(self):
        super(TokenRiskCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Base Optimizer Class ---
class BaseOptimizer:
    def __init__(self, config, data, llm_client, device):
        self.config = config
        self.prompts = data['optimization']['prompt'].tolist()
        self.llm_client = llm_client
        self.device = device
        self.history = []

    def propose(self):
        raise NotImplementedError

    def update(self, prompt, result):
        self.history.append({"prompt": prompt, **result})

    def run_optimization(self):
        budget = self.config['optimization_params']['api_budget']
        for i in range(budget):
            print(f"  Trial {i+1}/{budget}...")
            prompt = self.propose()
            result = self.llm_client.get_score(prompt)
            self.update(prompt, result)
        return self.history

# --- Baseline Optimizers ---
class RandomOptimizer(BaseOptimizer):
    def propose(self):
        return random.choice(self.prompts)

class ShiftBOOptimizer(BaseOptimizer):
    # Simplified implementation of SHIFT-BO
    def __init__(self, config, data, llm_client, device):
        super().__init__(config, data, llm_client, device)
        self.model = BayesianRidge()
        self.X_hist = []
        self.y_hist = []

    def propose(self):
        if len(self.history) < 5:
            return random.choice(self.prompts)
        else:
            # Simulate UCB acquisition on random features
            X_cand = np.random.rand(len(self.prompts), 10)
            mu, std = self.model.predict(X_cand, return_std=True)
            ucb = mu + self.config['hyperparameters']['bo_beta'][0] * std
            best_prompt_idx = np.argmax(ucb)
            return self.prompts[best_prompt_idx]

    def update(self, prompt, result):
        super().update(prompt, result)
        # Use a dummy feature vector for the prompt
        self.X_hist.append(np.random.rand(1, 10))
        self.y_hist.append(result['helpfulness'])
        if len(self.history) >= 2:
            self.model.fit(np.vstack(self.X_hist), np.array(self.y_hist))

class MetaBOPromptOptimizer(BaseOptimizer):
    # Simplified implementation of MetaBO-Prompt
    def propose(self):
        # Meta-learning aspect is complex to simulate; falls back to random search here
        return random.choice(self.prompts)

# --- REFLECT-BO Implementation ---
class ReflectBOOptimizer(BaseOptimizer):
    def __init__(self, config, data, llm_client, device):
        super().__init__(config, data, llm_client, device)
        self.encoder_name = self.config['model_names']['encoder']
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.encoder_name).to(self.device)
        self.bayesian_head = BayesianRidge()
        self.risk_cnn = TokenRiskCNN().to(self.device)
        self.cnn_optimizer = optim.Adam(self.risk_cnn.parameters(), lr=self.config['hyperparameters']['learning_rate'][0])
        self.kalman_filter = KalmanFilter(n_dim_obs=1, n_dim_state=1)
        self.X_embedded = []
        self.y_helpfulness = []
        self.y_severity = []
        self.residuals = []
        self.bootstrap()

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.no_grad():
            embeddings = self.encoder(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        return embeddings.cpu().numpy()

    def bootstrap(self):
        print("  Running on-the-fly contrastive bootstrapping...")
        bootstrap_budget = self.config['optimization_params']['bootstrap_budget']
        seed_prompts = random.sample(self.prompts, bootstrap_budget)
        positive_pairs = []
        for p in seed_prompts:
            paraphrases = self.llm_client.get_paraphrases(p)
            for para in paraphrases:
                positive_pairs.append((p, para))
        # In a real implementation, you would fine-tune the encoder here.
        # This is a placeholder for the logic.
        print(f"  Generated {len(positive_pairs)} pairs for bootstrapping.")

    def cvar_ei_acquisition(self, mu, std, severity_predictions, cvar_alpha, beta):
        # Simplified CVaR-EI: Penalize EI by predicted severity risk
        ei = mu + beta * std
        cvar_penalty = np.percentile(severity_predictions, cvar_alpha * 100)
        return ei - cvar_penalty

    def propose(self):
        if len(self.history) < 5: # Initial random exploration
            return random.choice(self.prompts)

        # Kalman Filter proactive shift anticipation
        if len(self.residuals) > 10:
            kf = self.kalman_filter.em(self.residuals, n_iter=5)
            (filtered_state_means, _) = kf.filter(self.residuals)
            (pred_state, _) = kf.predict(filtered_state_means[-1])
            if pred_state[0, 0] > self.config['kalman_params']['tau'][0]:
                print("  Kalman filter anticipates shift, performing exploration.")
                return random.choice(self.prompts) # Thompson sampling would be better

        # Standard acquisition
        candidate_embeddings = self.embed(self.prompts)
        mu, std = self.bayesian_head.predict(candidate_embeddings, return_std=True)
        
        # Predict token-wise risk (simulated here with random heatmaps)
        dummy_heatmaps = torch.randn(len(self.prompts), 1, 32, 32).to(self.device)
        with torch.no_grad():
            severity_predictions = self.risk_cnn(dummy_heatmaps).cpu().numpy().flatten()
        
        scores = self.cvar_ei_acquisition(mu, std, severity_predictions, 
                                           self.config['hyperparameters']['cvar_alpha'][0], 
                                           self.config['hyperparameters']['bo_beta'][0])
        best_idx = np.argmax(scores)
        return self.prompts[best_idx]

    def update(self, prompt, result):
        super().update(prompt, result)
        embedding = self.embed([prompt])
        self.X_embedded.append(embedding)
        self.y_helpfulness.append(result['helpfulness'])
        self.y_severity.append(result['severity_cost'])

        # Update Bayesian Head and calculate residual
        if len(self.history) >= 2:
            X = np.vstack(self.X_embedded)
            y = np.array(self.y_helpfulness)
            self.bayesian_head.fit(X, y)
            pred = self.bayesian_head.predict(embedding)
            self.residuals.append(result['helpfulness'] - pred[0])

        # Update Risk CNN
        heatmap = result['heatmap'].to(self.device)
        target = torch.tensor([result['severity_cost']], dtype=torch.float32).to(self.device)
        self.risk_cnn.train()
        self.cnn_optimizer.zero_grad()
        prediction = self.risk_cnn(heatmap)
        loss = nn.MSELoss()(prediction.squeeze(), target)
        loss.backward()
        self.cnn_optimizer.step()

# --- Federated Learning Simulation (Experiment 2) ---
class FederatedClient:
    def __init__(self, client_id, config, data, llm_client, device):
        self.client_id = client_id
        self.optimizer = ReflectBOOptimizer(config, data, llm_client, device)
        self.device = device
        
    def local_update(self):
        print(f"Client {self.client_id}: Starting local episode...")
        self.optimizer.run_optimization()
        return self.optimizer.encoder.state_dict()

    def apply_global_model(self, global_state_dict):
        self.optimizer.encoder.load_state_dict(global_state_dict)

def run_experiment_1(config, data, device):
    print("Running Experiment 1: Bootstrapping in Data-Scarce Domains")
    results = {}
    llm_client = LLMAPIClient(config)
    
    optimizers = {
        "REFLECT-BO": ReflectBOOptimizer,
        "SHIFT-BO": ShiftBOOptimizer,
        "MetaBO-Prompt": MetaBOPromptOptimizer,
        "Random": RandomOptimizer
    }

    for name, opt_class in optimizers.items():
        print(f"\n----- Evaluating {name} -----")
        all_seed_histories = []
        for seed in config['seeds']:
            print(f"Running seed {seed}...")
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            optimizer = opt_class(config, data, llm_client, device)
            history = optimizer.run_optimization()
            all_seed_histories.append(history)
        results[name] = all_seed_histories
    return results

def run_experiment_2(config, data, device):
    print("Running Experiment 2: Continual, Privacy-Preserving Adaptation")
    llm_client = LLMAPIClient(config)
    n_clients = config['federated_params']['n_clients']
    n_episodes = config['federated_params']['episodes']
    results = {method: [] for method in config['experiment_2']['methods']}
    
    # Split data among clients
    client_data_splits = np.array_split(data['optimization'], n_clients)
    
    for method in config['experiment_2']['methods']:
        print(f"\n----- Evaluating {method} -----")
        # Initialize server model
        global_encoder = AutoModel.from_pretrained(config['model_names']['encoder']).to(device)
        
        for episode in range(n_episodes):
            print(f"-- Episode {episode+1}/{n_episodes} --")
            client_gradients = []
            episode_history = []

            # Simulate clients running in parallel
            for i in range(n_clients):
                client_df = pd.DataFrame(client_data_splits[i])
                client_data = {'optimization': client_df}
                client = FederatedClient(i, config, client_data, llm_client, device)
                client.apply_global_model(global_encoder.state_dict())
                
                # Attach PrivacyEngine if DP is enabled
                if method == "REFLECT-BO" or method == "REFLECT-BO-NoShift":
                    if not ModuleValidator.is_valid(client.optimizer.encoder):
                        client.optimizer.encoder = ModuleValidator.fix(client.optimizer.encoder)
                    
                    privacy_engine = PrivacyEngine()
                    encoder_optimizer = optim.Adam(client.optimizer.encoder.parameters(), lr=config['hyperparameters']['learning_rate'][0])

                    client.optimizer.encoder, encoder_optimizer, _ = privacy_engine.make_private(
                        module=client.optimizer.encoder,
                        optimizer=encoder_optimizer,
                        data_loader=[(torch.randn(1, 384),) for _ in range(10)], # Dummy dataloader
                        noise_multiplier=1.1, # Corresponds to eps=~1.0 for this setup
                        max_grad_norm=1.0,
                    )

                client.local_update()
                episode_history.extend(client.optimizer.history)
                # Collect gradients (simplified)
                client_gradients.append({k: v.cpu() for k, v in client.optimizer.encoder.state_dict().items()})

            # FedAvg aggregation
            with torch.no_grad():
                avg_state_dict = global_encoder.state_dict()
                for key in avg_state_dict.keys():
                    avg_tensor = torch.stack([grad[key] for grad in client_gradients]).mean(dim=0)
                    avg_state_dict[key] = avg_tensor
                global_encoder.load_state_dict(avg_state_dict)
            
            results[method].append(episode_history)
            print(f"-- Drift occurs after episode {episode+1} --")

    return results

def run_experiment_3(config, data, device):
    print("Running Experiment 3: Human-in-the-Loop (Headless Mode)")
    # This runs the 'headless' condition of the experiment
    llm_client = LLMAPIClient(config)
    results = {}
    all_seed_histories = []
    for seed in config['seeds']:
        print(f"Running seed {seed}...")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        optimizer = ReflectBOOptimizer(config, data, llm_client, device)
        history = optimizer.run_optimization()
        all_seed_histories.append(history)
    results["REFLECT-BO-headless"] = all_seed_histories
    return results

def run(config, preprocessed_data_path):
    """Main training entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_files = list(Path(preprocessed_data_path).glob("*.parquet"))
    data = {f.stem: pd.read_parquet(f) for f in data_files}

    exp_id = config['experiment_to_run']
    if exp_id == 1:
        results = run_experiment_1(config, data, device)
    elif exp_id == 2:
        results = run_experiment_2(config, data, device)
    elif exp_id == 3:
        results = run_experiment_3(config, data, device)
    else:
        raise ValueError(f"Invalid experiment_id: {exp_id}")

    # Save results
    output_dir = Path(config['paths']['training_output_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    output_file = output_dir / f"training_results_exp{exp_id}_{ts}.json"

    # Convert tensors in history to lists for JSON serialization
    for method, seeds in results.items():
        for seed_idx, history in enumerate(seeds):
            for step_idx, step in enumerate(history):
                if 'heatmap' in step and isinstance(step['heatmap'], torch.Tensor):
                    results[method][seed_idx][step_idx]['heatmap'] = 'removed_for_json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Training results saved to {output_file}")
    return str(output_dir)