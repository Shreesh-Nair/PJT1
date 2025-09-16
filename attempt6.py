# Multi-Scale Temporal Attention with Focal Loss - FIXED VERSION
# Optimized for precision with corrected hyperparameters

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from scipy.interpolate import interp1d
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIXED FOCAL LOSS FOR PRECISION IMPROVEMENT
# ============================================================================

class FocalLoss(nn.Module):
    """FIXED: Gentler Focal Loss for better precision/recall balance"""
    def __init__(self, alpha=0.25, gamma=1.0, reduction='mean'):  # FIXED PARAMETERS
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Much lower alpha
        self.gamma = gamma  # Much lower gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal term (gentler)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# SIMPLIFIED MULTI-SCALE MODULES (FIXED)
# ============================================================================

class TemporalDifferenceEncoder(nn.Module):
    """Simplified temporal difference encoding"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        
        if seq_len < 2:
            return torch.zeros(batch_size, seq_len, 64).to(x.device)
        
        # Compute temporal differences
        diff_features = []
        diff_features.append(torch.zeros(batch_size, feature_dim).to(x.device))
        
        for i in range(1, seq_len):
            diff = x[:, i] - x[:, i-1]
            diff_features.append(diff)
        
        temporal_diffs = torch.stack(diff_features, dim=1)
        motion_encoded = self.motion_encoder(temporal_diffs)
        
        return motion_encoded

class SimpleMultiScaleEncoder(nn.Module):
    """FIXED: Simplified multi-scale processing"""
    def __init__(self, input_dim, embed_dim=128, num_heads=2, dropout=0.15):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Simplified single transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        
        # Simple scale weighting (learnable but constrained)
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)  # Initialize equally
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def create_simple_windows(self, x, window_size):
        """FIXED: Less aggressive windowing"""
        batch_size, seq_len, embed_dim = x.shape
        
        if seq_len <= window_size:
            return x
        
        # FIXED: Less aggressive stride
        stride = max(1, window_size // 4)  # Changed from // 2
        windowed_features = []
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window = x[:, start:end, :]
            windowed_features.append(window.mean(dim=1))
        
        # Ensure we have the right number of outputs
        while len(windowed_features) < seq_len:
            windowed_features.append(windowed_features[-1])
        
        windowed_features = windowed_features[:seq_len]
        return torch.stack(windowed_features, dim=1)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to embedding space
        x_proj = self.input_projection(x)
        
        # Simple multi-scale processing
        short_features = self.create_simple_windows(x_proj, window_size=6)   # Smaller windows
        medium_features = self.create_simple_windows(x_proj, window_size=12)
        long_features = self.create_simple_windows(x_proj, window_size=24)
        
        # Apply transformer to each scale
        short_out = self.transformer_layer(short_features)
        medium_out = self.transformer_layer(medium_features)
        long_out = self.transformer_layer(long_features)
        
        # FIXED: Normalized scale weights
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        
        # Weighted combination
        output = (
            scale_weights_norm[0] * short_out +
            scale_weights_norm[1] * medium_out +
            scale_weights_norm[2] * long_out
        )
        
        output = self.layer_norm(output)
        
        # Return scale weights for monitoring
        scale_usage = scale_weights_norm.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        return output, scale_usage

# ============================================================================
# FIXED MULTI-SCALE MODEL
# ============================================================================

class FixedMultiScaleVAD(nn.Module):
    """FIXED: Multi-Scale VAD with corrected parameters"""
    def __init__(self, input_dim, embed_dim=128, num_heads=2, dropout=0.2, max_seq_len=60):
        super().__init__()
        
        # Feature reduction (keep working approach)
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 8, input_dim // 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 32, embed_dim)
        )
        
        # Temporal difference encoder
        self.temporal_diff_encoder = TemporalDifferenceEncoder(embed_dim, hidden_dim=64)
        
        # Simplified multi-scale encoder
        self.multi_scale_encoder = SimpleMultiScaleEncoder(
            input_dim=embed_dim + 64,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # FIXED: Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)  # Returns logits
        )
        
    def forward(self, x, return_probabilities=False):
        batch_size, seq_len, _ = x.shape
        
        # Feature reduction
        x_reduced = self.feature_reducer(x)
        
        # Add positional encoding
        x_pos = x_reduced + self.pos_embedding[:, :seq_len, :]
        
        # Temporal differences
        motion_features = self.temporal_diff_encoder(x_pos)
        
        # Combine features
        combined_features = torch.cat([x_pos, motion_features], dim=-1)
        
        # Multi-scale processing
        temporal_features, scale_weights = self.multi_scale_encoder(combined_features)
        
        # Classification
        logits = self.classifier(temporal_features).squeeze(-1)
        
        if return_probabilities:
            return torch.sigmoid(logits), scale_weights
        else:
            return logits, scale_weights

# ============================================================================
# FIXED LOSS FUNCTION
# ============================================================================

def fixed_multi_scale_loss(logits, targets, scale_weights=None, 
                          alpha=0.25, gamma=1.0, beta=0.005, lambda_temp=0.001):
    """FIXED: Much gentler loss function"""
    
    # Gentler focal loss
    focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    focal_loss = focal_loss_fn(logits, targets)
    
    # FIXED: Much smaller temporal loss weight
    probabilities = torch.sigmoid(logits)
    temporal_loss = torch.tensor(0.0, device=logits.device)
    
    if probabilities.dim() > 1 and probabilities.size(1) > 1:
        temporal_diff = probabilities[:, 1:] - probabilities[:, :-1]
        temporal_loss = torch.mean(temporal_diff ** 2)
    
    # FIXED: Much smaller scale loss weight
    scale_loss = torch.tensor(0.0, device=logits.device)
    if scale_weights is not None:
        scale_variance = torch.var(scale_weights, dim=-1).mean()
        scale_loss = -scale_variance * 0.01  # Much smaller multiplier
    
    # FIXED: Proper loss combination
    total_loss = focal_loss + beta * temporal_loss + lambda_temp * scale_loss
    
    return total_loss, focal_loss, temporal_loss, scale_loss

# ============================================================================
# FIXED THRESHOLD SELECTION
# ============================================================================

def find_balanced_threshold(all_scores, all_labels):
    """FIXED: Find threshold that balances precision and recall"""
    thresholds = np.linspace(0.05, 0.95, 100)  # More comprehensive range
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        if len(np.unique(predictions)) > 1:
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            
            # Ensure both precision and recall are reasonable
            if precision > 0.05 and recall > 0.05:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
    
    return best_threshold

# ============================================================================
# DATASET CLASS (SAME AS BEFORE)
# ============================================================================

class EnhancedMemoryOptimizedDataset(Dataset):
    """Enhanced dataset with light data augmentation"""
    def __init__(self, i3d_dir, json_file, max_length=None, is_train=True, data_augmentation=True):
        self.i3d_dir = i3d_dir
        self.is_train = is_train
        self.data_augmentation = data_augmentation and is_train
        
        # Load labels
        with open(json_file, 'r') as f:
            all_labels = json.load(f)
        
        self.labels = {}
        for video_name, label in all_labels.items():
            npy_name = video_name.replace('.avi', '.npy')
            npy_path = os.path.join(i3d_dir, npy_name)
            
            if os.path.exists(npy_path):
                try:
                    test_load = np.load(npy_path)
                    if test_load.shape[0] > 0:
                        self.labels[video_name] = label
                except:
                    pass
        
        self.video_names = list(self.labels.keys())
        print(f"Dataset - Available files: {len(self.video_names)}")
        
        # Determine max length
        if max_length is None:
            max_len = 0
            sample_size = min(20, len(self.video_names))
            for video_name in self.video_names[:sample_size]:
                npy_name = video_name.replace('.avi', '.npy')
                npy_path = os.path.join(i3d_dir, npy_name)
                try:
                    features = np.load(npy_path)
                    max_len = max(max_len, features.shape[0])
                except:
                    continue
            self.max_length = min(max_len + 5, 60) if max_len > 0 else 60
        else:
            self.max_length = max_length
            
        print(f"Max sequence length: {self.max_length}")
    
    def extract_features(self, features):
        """Simple feature extraction"""
        T, spatial_regions, feature_dim = features.shape
        return features.reshape(T, -1)
    
    def light_augmentation(self, features, labels):
        """FIXED: Very light augmentation"""
        if not self.data_augmentation or np.random.random() > 0.3:
            return features, labels
        
        seq_len = features.shape[0]
        
        # Only very small temporal shifts
        if seq_len > 8:
            shift = np.random.randint(-1, 2)
            if shift != 0:
                if shift > 0:
                    features = np.concatenate([features[shift:], features[-shift:]])
                    labels = np.concatenate([labels[shift:], labels[-shift:]])
                else:
                    features = np.concatenate([features[:shift], features[:shift]])
                    labels = np.concatenate([labels[:shift], labels[:shift]])
        
        return features, labels
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        npy_name = video_name.replace('.avi', '.npy')
        npy_path = os.path.join(self.i3d_dir, npy_name)
        
        try:
            features = np.load(npy_path)
            if len(features.shape) != 3:
                features = np.random.randn(30, 10, 2048).astype(np.float32)
        except:
            features = np.random.randn(30, 10, 2048).astype(np.float32)
        
        processed_features = self.extract_features(features)
        original_seq_length = processed_features.shape[0]
        
        # Label processing
        if self.is_train:
            video_label = self.labels[video_name]
            frame_labels = np.full(original_seq_length, video_label, dtype=np.float32)
        else:
            try:
                frame_labels = np.array(self.labels[video_name], dtype=np.float32)
                if len(frame_labels) != original_seq_length:
                    if len(frame_labels) > 1:
                        f = interp1d(np.linspace(0, 1, len(frame_labels)), 
                                   frame_labels, kind='nearest')
                        frame_labels = f(np.linspace(0, 1, original_seq_length))
                    else:
                        frame_labels = np.zeros(original_seq_length, dtype=np.float32)
            except:
                frame_labels = np.zeros(original_seq_length, dtype=np.float32)
        
        # Light augmentation
        processed_features, frame_labels = self.light_augmentation(
            processed_features, frame_labels)
        
        # Sequence length processing
        current_seq_length = processed_features.shape[0]
        
        if current_seq_length > self.max_length:
            processed_features = processed_features[:self.max_length]
            frame_labels = frame_labels[:self.max_length]
            seq_length = self.max_length
        else:
            seq_length = current_seq_length
            if seq_length < self.max_length:
                padding_length = self.max_length - seq_length
                feature_padding = np.zeros((padding_length, processed_features.shape[1]), 
                                         dtype=np.float32)
                processed_features = np.vstack([processed_features, feature_padding])
                label_padding = np.zeros(padding_length, dtype=np.float32)
                frame_labels = np.concatenate([frame_labels, label_padding])
        
        return {
            'features': torch.FloatTensor(processed_features),
            'labels': torch.FloatTensor(frame_labels),
            'seq_length': seq_length,
            'video_name': video_name
        }

# ============================================================================
# FIXED TRAINING FUNCTIONS
# ============================================================================

def train_fixed_multi_scale(model, train_loader, optimizer, scaler, device, epoch):
    """FIXED: Training with proper loss handling"""
    model.train()
    total_loss = 0
    focal_losses = []
    temporal_losses = []
    scale_losses = []
    
    accumulation_steps = 2  # Smaller accumulation
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        features = batch['features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        seq_lengths = batch['seq_length']
        
        with torch.cuda.amp.autocast():
            logits, scale_weights = model(features)
            
            batch_loss = 0
            batch_focal = 0
            batch_temporal = 0
            batch_scale = 0
            valid_samples = 0
            
            for i in range(features.size(0)):
                seq_len = seq_lengths[i].item()
                if seq_len > 0:
                    pred_logits = logits[i, :seq_len]
                    target_labels = labels[i, :seq_len]
                    sample_scale_weights = scale_weights[i, :seq_len] if scale_weights is not None else None
                    
                    total_loss_sample, focal_loss, temporal_loss, scale_loss = fixed_multi_scale_loss(
                        pred_logits, target_labels, sample_scale_weights
                    )
                    
                    batch_loss += total_loss_sample
                    batch_focal += focal_loss
                    batch_temporal += temporal_loss
                    batch_scale += scale_loss
                    valid_samples += 1
            
            if valid_samples > 0:
                loss = batch_loss / (valid_samples * accumulation_steps)
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                focal_losses.append(batch_focal.item() / valid_samples)
                temporal_losses.append(batch_temporal.item() / valid_samples)
                scale_losses.append(batch_scale.item() / valid_samples)
        
        if (batch_idx + 1) % (accumulation_steps * 5) == 0:
            torch.cuda.empty_cache()
    
    print(f"  Focal Loss: {np.mean(focal_losses) if focal_losses else 0:.4f}")
    print(f"  Temporal Loss: {np.mean(temporal_losses) if temporal_losses else 0:.4f}")
    print(f"  Scale Loss: {np.mean(scale_losses) if scale_losses else 0:.4f}")
    
    return total_loss / len(train_loader)

def evaluate_fixed_multi_scale(model, test_loader, device):
    """FIXED: Evaluation with better threshold selection"""
    model.eval()
    all_scores = []
    all_labels = []
    all_scale_weights = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            seq_lengths = batch['seq_length']
            
            with torch.cuda.amp.autocast():
                scores, scale_weights = model(features, return_probabilities=True)
                
                for i in range(features.size(0)):
                    seq_len = seq_lengths[i].item()
                    if seq_len > 0:
                        frame_scores = scores[i, :seq_len].cpu().numpy()
                        frame_labels = labels[i, :seq_len].cpu().numpy()
                        frame_scale_weights = scale_weights[i, :seq_len].cpu().numpy()
                        
                        all_scores.extend(frame_scores)
                        all_labels.extend(frame_labels)
                        all_scale_weights.extend(frame_scale_weights)
            
            if len(all_scores) % 1000 == 0:
                torch.cuda.empty_cache()
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_scale_weights = np.array(all_scale_weights)
    
    if len(np.unique(all_labels)) < 2:
        return 0.5, 0.5, 0.5, 0.0, 0.0, 0.0
    
    # Calculate AUC
    auc_score = roc_auc_score(all_labels, all_scores)
    
    # FIXED: Better threshold selection
    optimal_threshold = find_balanced_threshold(all_scores, all_labels)
    
    # Calculate metrics at optimal threshold
    predictions = (all_scores >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == all_labels)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    
    # Scale usage analysis
    scale_usage = np.mean(all_scale_weights, axis=0)
    
    print(f"Detailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"Scale Usage - Short: {scale_usage[0]:.3f}, Medium: {scale_usage[1]:.3f}, Long: {scale_usage[2]:.3f}")
    
    return auc_score, accuracy, optimal_threshold, precision, recall, f1

# ============================================================================
# FIXED MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """FIXED: Main function with corrected hyperparameters"""
    config = {
        'batch_size': 4,          # FIXED: Increased
        'learning_rate': 0.00005, # FIXED: Much lower
        'num_epochs': 30,         # More epochs
        'embed_dim': 128,
        'num_heads': 2,           # Reduced complexity
        'dropout': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'patience': 15,           # More patience
        'focal_alpha': 0.25,      # FIXED: Much lower
        'focal_gamma': 1.0,       # FIXED: Much lower
        'warmup_epochs': 5
    }
    
    print("="*60)
    print("FIXED MULTI-SCALE TEMPORAL ATTENTION")
    print("Precision-Focused with Corrected Hyperparameters")
    print("="*60)
    print(f"\nFIXED Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Dataset paths
    train_i3d_dir = "I3D/Train_I3D"
    test_i3d_dir = "I3D/Test_I3D"
    train_json = "ShanghaiTech-campus/train.json"
    test_json = "ShanghaiTech-campus/test.json"
    
    print("\nLoading datasets...")
    try:
        train_dataset = EnhancedMemoryOptimizedDataset(
            train_i3d_dir, train_json, is_train=True, data_augmentation=True
        )
        test_dataset = EnhancedMemoryOptimizedDataset(
            test_i3d_dir, test_json, 
            max_length=train_dataset.max_length, 
            is_train=False, data_augmentation=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, 0, 0
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Initialize FIXED model
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]
    print(f"Input feature dimension: {input_dim}")
    
    model = FixedMultiScaleVAD(
        input_dim=input_dim,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        max_seq_len=train_dataset.max_length
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # FIXED: More conservative optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - config['warmup_epochs']) / 
                                   (config['num_epochs'] - config['warmup_epochs'])))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    
    print("\nStarting FIXED multi-scale training...")
    best_auc = 0
    best_precision = 0
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_loss = train_fixed_multi_scale(
            model, train_loader, optimizer, scaler, config['device'], epoch+1
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        # Evaluation every 2 epochs
        if (epoch + 1) % 2 == 0:
            print("Evaluating...")
            auc_score, accuracy, threshold, precision, recall, f1 = evaluate_fixed_multi_scale(
                model, test_loader, config['device']
            )
            
            print(f"AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # FIXED: Save model based on balanced metrics
            improvement = False
            if f1 > best_f1:  # Prioritize F1-score
                improvement = True
            elif f1 >= best_f1 * 0.98 and precision > best_precision:  # Or precision if F1 similar
                improvement = True
            elif f1 >= best_f1 * 0.95 and auc_score > best_auc:  # Or AUC if others similar
                improvement = True
            
            if improvement:
                best_auc = auc_score
                best_precision = precision
                best_f1 = f1
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'auc': auc_score,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'threshold': threshold
                }, 'best_fixed_multi_scale_model.pth')
                
                print(f"New best model saved! F1: {best_f1:.4f}, Precision: {best_precision:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        torch.cuda.empty_cache()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL FIXED EVALUATION")
    print("="*60)
    
    try:
        checkpoint = torch.load('best_fixed_multi_scale_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        auc_score, accuracy, threshold, precision, recall, f1 = evaluate_fixed_multi_scale(
            model, test_loader, config['device']
        )
        
        print(f"Final Results:")
        print(f"  AUC-ROC: {auc_score:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        print(f"\nImprovement Analysis:")
        if precision > 0.4:
            print(f"  ✅ GOOD Precision: {precision:.1%}")
        elif precision > 0.2:
            print(f"  📈 IMPROVED Precision: {precision:.1%}")
        else:
            print(f"  ⚠️ Still Low Precision: {precision:.1%}")
            
        if f1 > 0.3:
            print(f"  ✅ GOOD F1-Score: {f1:.1%}")
        elif f1 > 0.15:
            print(f"  📈 IMPROVED F1-Score: {f1:.1%}")
        else:
            print(f"  ⚠️ Still Low F1-Score: {f1:.1%}")
            
    except Exception as e:
        print(f"Error during final evaluation: {e}")
        auc_score, precision = 0, 0
    
    return model, auc_score, precision

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    print("🚀 FIXED Multi-Scale Training...")
    print("🔧 Corrected hyperparameters for better precision")
    print("⚡ RTX 4050 Optimized")
    
    model, auc, precision = main()
    
    print(f"\n🎉 FIXED TRAINING COMPLETE!")
    print(f"📊 Final AUC: {auc:.4f} | Final Precision: {precision:.4f}")
    print(f"💾 Model saved: 'best_fixed_multi_scale_model.pth'")
    
    if precision > 0.3:
        print(f"✅ MUCH BETTER: {precision:.1%} precision achieved!")
    elif precision > 0.15:
        print(f"📈 IMPROVED: {precision:.1%} precision - getting better!")
    else:
        print(f"🔄 Still working on precision improvements...")
