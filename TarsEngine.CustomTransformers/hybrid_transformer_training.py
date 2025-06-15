# hybrid_transformer_training.py
# Phase 1: Custom Transformer Training for TARS Non-Euclidean Vector Store

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import math
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np

# ========= Hybrid Projection Heads ========= #

class HyperbolicProjection(nn.Module):
    """Projects embeddings into hyperbolic space (Poincaré disk model)"""
    def __init__(self, input_dim, output_dim, curvature=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.c = curvature  # positive for hyperbolic space
        
    def mobius_add(self, x, y):
        """Möbius addition in Poincaré disk"""
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return num / (denom + 1e-8)
    
    def forward(self, x):
        proj = self.linear(x)
        # Project to Poincaré disk (norm < 1)
        norm = torch.norm(proj, dim=-1, keepdim=True)
        return proj / (1 + norm)  # Ensures ||proj|| < 1

class ProjectiveProjection(nn.Module):
    """Projects embeddings into projective space"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        # Normalize to unit sphere (projective equivalence)
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

class DualQuaternionProjection(nn.Module):
    """Projects embeddings into dual quaternion space"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        assert output_dim % 8 == 0, "Dual quaternion output_dim must be divisible by 8"
        self.real_linear = nn.Linear(input_dim, output_dim // 2)
        self.dual_linear = nn.Linear(input_dim, output_dim // 2)
    
    def forward(self, x):
        real = self.real_linear(x)
        dual = self.dual_linear(x)
        # Normalize real part as quaternion
        real_norm = torch.norm(real.view(-1, real.size(-1)//4, 4), dim=-1, keepdim=True)
        real = real.view(-1, real.size(-1)//4, 4) / (real_norm + 1e-8)
        real = real.view(-1, real.size(-1))
        return torch.cat([real, dual], dim=-1)

# ========= Hybrid Multi-Head Architecture ========= #

class HybridHead(nn.Module):
    """Multi-space embedding head for TARS"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.euclidean = nn.Linear(hidden_dim, output_dim)
        self.hyperbolic = HyperbolicProjection(hidden_dim, output_dim)
        self.projective = ProjectiveProjection(hidden_dim, output_dim)
        self.dual_quaternion = DualQuaternionProjection(hidden_dim, output_dim)
        
    def forward(self, x):
        return {
            'euclidean': self.euclidean(x),
            'hyperbolic': self.hyperbolic(x),
            'projective': self.projective(x),
            'dual_quaternion': self.dual_quaternion(x)
        }

# ========= TARS Custom Transformer ========= #

class TarsCustomTransformer(nn.Module):
    """Custom transformer for TARS with hybrid embedding heads"""
    def __init__(self, backbone_name="microsoft/MiniLM-L12-H384-uncased", 
                 hidden_dim=384, output_dim=128):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.head = HybridHead(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None):
        # Get backbone embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        hidden = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
        hidden = self.dropout(hidden)
        # Project to hybrid spaces
        return self.head(hidden)

# ========= TARS Dataset for .trsx Files ========= #

class TarsDataset(Dataset):
    """Dataset for TARS .trsx and metascript files"""
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entries = self._load_trsx_data(data_dir)
        
    def _load_trsx_data(self, data_dir):
        """Load and parse .trsx files"""
        entries = []
        
        # Look for .trsx files in the data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.trsx'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            entries.append({
                                'text': content,
                                'source': file_path,
                                'type': 'trsx'
                            })
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
        
        # Add some synthetic TARS-style data if no .trsx files found
        if not entries:
            print("No .trsx files found, generating synthetic TARS data...")
            entries = self._generate_synthetic_data()
            
        return entries
    
    def _generate_synthetic_data(self):
        """Generate synthetic TARS-style training data"""
        synthetic_data = [
            {
                'text': 'belief_graph { contradiction_rate: 0.05, coherence: 0.92 }',
                'source': 'synthetic',
                'type': 'belief'
            },
            {
                'text': 'agent_trace { reasoning_path: "hypothesis -> validation -> conclusion", confidence: 0.87 }',
                'source': 'synthetic', 
                'type': 'reasoning'
            },
            {
                'text': 'flux_block { transformation: "sierpinski", iterations: 5, complexity: 0.73 }',
                'source': 'synthetic',
                'type': 'fractal'
            },
            {
                'text': 'vector_similarity { euclidean: 0.23, hyperbolic: 0.45, cosine: 0.78 }',
                'source': 'synthetic',
                'type': 'similarity'
            },
            {
                'text': 'meta_reflection { improvement_suggestion: "reduce contradiction rate", priority: high }',
                'source': 'synthetic',
                'type': 'reflection'
            }
        ]
        return synthetic_data * 100  # Replicate for training
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            entry['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': entry['text'],
            'source': entry['source'],
            'type': entry['type']
        }

# ========= Training Functions ========= #

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Simple reconstruction loss for now (will be replaced with hybrid loss)
        euclidean_emb = outputs['euclidean']
        loss = F.mse_loss(euclidean_emb, torch.randn_like(euclidean_emb))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def save_model(model, tokenizer, save_dir):
    """Save the trained model and tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save model config
    config = {
        'model_type': 'TarsCustomTransformer',
        'hidden_dim': 384,
        'output_dim': 128,
        'spaces': ['euclidean', 'hyperbolic', 'projective', 'dual_quaternion']
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Model saved to {save_dir}")

# ========= Main Training Function ========= #

def train_tars_transformer(data_dir="data", epochs=5, batch_size=8, learning_rate=2e-5):
    """Main training function for TARS custom transformer"""
    print("🌌 Starting TARS Custom Transformer Training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    model = TarsCustomTransformer().to(device)
    
    # Setup dataset and dataloader
    dataset = TarsDataset(data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Loaded {len(dataset)} training examples")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Average loss: {avg_loss:.4f}")
    
    # Save the trained model
    save_model(model, tokenizer, "models/tars_custom_transformer")
    
    print("🎉 TARS Custom Transformer training complete!")
    return model, tokenizer

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_tars_transformer()
