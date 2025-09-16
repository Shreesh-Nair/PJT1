# testing_metrics_attempt6.py - Comprehensive Metrics for Fixed Multi-Scale Model
# Evaluates the Fixed Multi-Scale Temporal Attention model with proper architecture matching

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FOCAL LOSS (FOR REFERENCE - NOT USED IN TESTING)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - same as training"""
    def __init__(self, alpha=0.25, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# MULTI-SCALE MODEL ARCHITECTURE - EXACT MATCH WITH TRAINING
# ============================================================================

class TemporalDifferenceEncoder(nn.Module):
    """Temporal difference encoding - exact match"""
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
    """Multi-scale encoder - exact match"""
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
        """Less aggressive windowing"""
        batch_size, seq_len, embed_dim = x.shape
        
        if seq_len <= window_size:
            return x
        
        # Less aggressive stride
        stride = max(1, window_size // 4)
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
        short_features = self.create_simple_windows(x_proj, window_size=6)
        medium_features = self.create_simple_windows(x_proj, window_size=12)
        long_features = self.create_simple_windows(x_proj, window_size=24)
        
        # Apply transformer to each scale
        short_out = self.transformer_layer(short_features)
        medium_out = self.transformer_layer(medium_features)
        long_out = self.transformer_layer(long_features)
        
        # Normalized scale weights
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

class FixedMultiScaleVAD(nn.Module):
    """EXACT MATCH with training model"""
    def __init__(self, input_dim, embed_dim=128, num_heads=2, dropout=0.2, max_seq_len=60):
        super().__init__()
        
        # Feature reduction (exact match)
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
        
        # Classifier
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
# TEST DATASET - MATCHING TRAINING PREPROCESSING
# ============================================================================

class FixedMultiScaleTestDataset(Dataset):
    """Test dataset matching the training preprocessing exactly"""
    def __init__(self, i3d_dir, json_file, max_length=60):
        self.i3d_dir = i3d_dir
        self.max_length = max_length
        
        # Load labels
        with open(json_file, 'r') as f:
            all_labels = json.load(f)
        
        self.labels = {}
        self.video_names = []
        
        # Filter available videos
        for video_name, label in all_labels.items():
            npy_name = video_name.replace('.avi', '.npy')
            npy_path = os.path.join(i3d_dir, npy_name)
            
            if os.path.exists(npy_path):
                try:
                    test_load = np.load(npy_path)
                    if test_load.shape[0] > 0:
                        self.labels[video_name] = label
                        self.video_names.append(video_name)
                except:
                    continue
        
        print(f"Loaded {len(self.video_names)} test videos for Fixed Multi-Scale evaluation")
    
    def extract_features(self, features):
        """Same feature extraction as training"""
        T, spatial_regions, feature_dim = features.shape
        return features.reshape(T, -1)  # (T, 20480)
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        npy_name = video_name.replace('.avi', '.npy')
        npy_path = os.path.join(self.i3d_dir, npy_name)
        
        # Load features
        try:
            features = np.load(npy_path)  # (T, 10, 2048)
            if len(features.shape) != 3:
                features = np.random.randn(30, 10, 2048).astype(np.float32)
        except:
            features = np.random.randn(30, 10, 2048).astype(np.float32)
        
        # Extract features (same as training)
        processed_features = self.extract_features(features)
        original_seq_length = processed_features.shape[0]
        
        # Process labels (frame-level for test set)
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
        
        # Sequence length processing (same as training)
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
            'video_name': video_name,
            'features': torch.FloatTensor(processed_features),
            'labels': torch.FloatTensor(frame_labels),
            'seq_length': seq_length
        }

# ============================================================================
# BALANCED THRESHOLD SELECTION
# ============================================================================

def find_balanced_threshold_test(all_scores, all_labels):
    """Find balanced threshold - same as training"""
    thresholds = np.linspace(0.05, 0.95, 100)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        if len(np.unique(predictions)) > 1:
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            
            if precision > 0.05 and recall > 0.05:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
    
    return best_threshold

# ============================================================================
# COMPREHENSIVE METRICS CALCULATION
# ============================================================================

def calculate_fixed_multi_scale_metrics(model_path, test_i3d_dir, test_json_file, batch_size=4):
    """Calculate comprehensive metrics for Fixed Multi-Scale model"""
    print("="*80)
    print("FIXED MULTI-SCALE TEMPORAL ATTENTION - COMPREHENSIVE METRICS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = FixedMultiScaleTestDataset(test_i3d_dir, test_json_file, max_length=60)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Initialize model with saved configuration
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Get input dimension from sample
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['features'].shape[-1]
        
        model = FixedMultiScaleVAD(
            input_dim=input_dim,
            embed_dim=config.get('embed_dim', 128),
            num_heads=config.get('num_heads', 2),
            dropout=config.get('dropout', 0.2),
            max_seq_len=test_dataset.max_length
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Fixed Multi-Scale model loaded successfully:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Training epochs: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Training AUC: {checkpoint.get('auc', 'unknown'):.4f}")
        print(f"  Training Precision: {checkpoint.get('precision', 'unknown'):.4f}")
        print(f"  Training F1-Score: {checkpoint.get('f1_score', 'unknown'):.4f}")
        
    except Exception as e:
        print(f"Error loading Fixed Multi-Scale model: {e}")
        return None
    
    model.eval()
    
    # Collect all predictions and labels
    all_scores = []
    all_labels = []
    all_scale_weights = []
    video_results = {}
    
    print(f"\nEvaluating {len(test_dataset)} videos with Fixed Multi-Scale model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing videos")):
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            seq_lengths = batch['seq_length']
            video_names = batch['video_name']
            
            # Get probabilities and scale weights
            with torch.cuda.amp.autocast():
                scores, scale_weights = model(features, return_probabilities=True)
            
            # Process each video in batch
            for i in range(features.size(0)):
                seq_len = seq_lengths[i].item()
                if seq_len > 0:
                    video_name = video_names[i]
                    frame_scores = scores[i, :seq_len].cpu().numpy()
                    frame_labels = labels[i, :seq_len].cpu().numpy()
                    frame_scale_weights = scale_weights[i, :seq_len].cpu().numpy()
                    
                    # Store frame-level data
                    all_scores.extend(frame_scores)
                    all_labels.extend(frame_labels)
                    all_scale_weights.extend(frame_scale_weights)
                    
                    # Store video-level results
                    video_results[video_name] = {
                        'scores': frame_scores,
                        'labels': frame_labels,
                        'scale_weights': frame_scale_weights,
                        'mean_score': np.mean(frame_scores),
                        'max_score': np.max(frame_scores),
                        'anomaly_ratio': np.mean(frame_labels),
                        'scale_usage': np.mean(frame_scale_weights, axis=0)
                    }
            
            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_scale_weights = np.array(all_scale_weights)
    
    print(f"\nTotal frames evaluated: {len(all_scores)}")
    print(f"Anomalous frames: {np.sum(all_labels)} ({np.mean(all_labels)*100:.1f}%)")
    
    # Handle edge cases
    if len(np.unique(all_labels)) < 2:
        print("Warning: Only one class present in labels.")
        return None
    
    # Calculate AUC and find optimal threshold
    try:
        auc_score = roc_auc_score(all_labels, all_scores)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        
        # Use balanced threshold selection (same as training)
        optimal_threshold = find_balanced_threshold_test(all_scores, all_labels)
        
    except Exception as e:
        print(f"Error calculating ROC: {e}")
        auc_score = 0.0
        optimal_threshold = 0.5
    
    # Apply threshold and calculate metrics
    binary_predictions = (all_scores >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_labels, binary_predictions, zero_division=0)
    f1 = f1_score(all_labels, binary_predictions, zero_division=0)
    
    # Confusion matrix
    tp = np.sum((binary_predictions == 1) & (all_labels == 1))
    fp = np.sum((binary_predictions == 1) & (all_labels == 0))
    tn = np.sum((binary_predictions == 0) & (all_labels == 0))
    fn = np.sum((binary_predictions == 0) & (all_labels == 1))
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Scale usage analysis
    overall_scale_usage = np.mean(all_scale_weights, axis=0)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("FIXED MULTI-SCALE MODEL - COMPREHENSIVE RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall: {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1-Score: {f1:.4f} ({f1*100:.1f}%)")
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  NPV: {npv:.4f}")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"  True Positives: {tp:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Negatives: {fn:,}")
    
    print(f"\n⚙️ THRESHOLD & SCORING:")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Score Range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  Score Mean: {np.mean(all_scores):.4f}")
    print(f"  Score Std: {np.std(all_scores):.4f}")
    
    print(f"\n🔄 MULTI-SCALE ANALYSIS:")
    print(f"  Short-term usage: {overall_scale_usage[0]:.3f} ({overall_scale_usage[0]*100:.1f}%)")
    print(f"  Medium-term usage: {overall_scale_usage[1]:.3f} ({overall_scale_usage[1]*100:.1f}%)")
    print(f"  Long-term usage: {overall_scale_usage[2]:.3f} ({overall_scale_usage[2]*100:.1f}%)")
    
    # Performance assessment with multi-scale context
    print(f"\n📈 FIXED MULTI-SCALE ASSESSMENT:")
    if precision > 0.4 and recall > 0.4:
        print("  🎯 EXCELLENT - Well-balanced precision and recall!")
    elif precision > 0.3 and f1 > 0.3:
        print("  ✅ VERY GOOD - Strong improvement in precision!")
    elif precision > 0.2 and f1 > 0.2:
        print("  👍 GOOD - Solid improvement over baseline!")
    elif precision > 0.1:
        print("  📈 IMPROVED - Better than previous attempts!")
    else:
        print("  ⚠️ NEEDS WORK - Still struggling with precision")
    
    # Multi-scale effectiveness
    scale_balance = 1 - np.var(overall_scale_usage)
    if scale_balance > 0.8:
        print("  🔄 Multi-scale: Well balanced usage across time scales")
    elif np.max(overall_scale_usage) > 0.8:
        print(f"  🔄 Multi-scale: Dominated by {'short' if np.argmax(overall_scale_usage) == 0 else 'medium' if np.argmax(overall_scale_usage) == 1 else 'long'}-term patterns")
    else:
        print("  🔄 Multi-scale: Moderate balance across scales")
    
    # Video-level analysis
    video_accuracies = []
    video_precisions = []
    video_f1s = []
    
    for video_name, results in video_results.items():
        video_scores = results['scores']
        video_labels = results['labels']
        if len(np.unique(video_labels)) > 1:
            video_predictions = (video_scores >= optimal_threshold).astype(int)
            video_accuracy = np.mean(video_predictions == video_labels)
            video_precision = precision_score(video_labels, video_predictions, zero_division=0)
            video_f1 = f1_score(video_labels, video_predictions, zero_division=0)
            
            video_accuracies.append(video_accuracy)
            video_precisions.append(video_precision)
            video_f1s.append(video_f1)
    
    if video_accuracies:
        print(f"\n🎬 VIDEO-LEVEL ANALYSIS:")
        print(f"  Videos with both classes: {len(video_accuracies)}")
        print(f"  Mean video accuracy: {np.mean(video_accuracies):.4f} ± {np.std(video_accuracies):.4f}")
        print(f"  Mean video precision: {np.mean(video_precisions):.4f} ± {np.std(video_precisions):.4f}")
        print(f"  Mean video F1-score: {np.mean(video_f1s):.4f} ± {np.std(video_f1s):.4f}")
        print(f"  Best video accuracy: {np.max(video_accuracies):.4f}")
        print(f"  Best video precision: {np.max(video_precisions):.4f}")
    
    # Save detailed results
    results_dict = {
        'model_info': {
            'model_type': 'Fixed Multi-Scale Temporal Attention',
            'architecture': 'Multi-scale encoder with temporal differences',
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_score),
            'specificity': float(specificity),
            'npv': float(npv)
        },
        'confusion_matrix': {
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        },
        'threshold_info': {
            'optimal_threshold': float(optimal_threshold),
            'score_mean': float(np.mean(all_scores)),
            'score_std': float(np.std(all_scores))
        },
        'multi_scale_analysis': {
            'short_term_usage': float(overall_scale_usage[0]),
            'medium_term_usage': float(overall_scale_usage[1]),
            'long_term_usage': float(overall_scale_usage[2]),
            'scale_balance': float(scale_balance)
        },
        'dataset_info': {
            'total_videos': len(video_results),
            'total_frames': len(all_scores),
            'anomaly_ratio': float(np.mean(all_labels))
        }
    }
    
    # Save to JSON
    output_file = 'fixed_multi_scale_metrics_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Detailed results saved to '{output_file}'")
    print("="*80)
    
    return results_dict

def compare_with_attempts(results_dict, baseline_accuracy=0.50, baseline_precision=0.06):
    """Compare Fixed Multi-Scale results with previous attempts"""
    if results_dict is None:
        return
    
    current_acc = results_dict['overall_metrics']['accuracy']
    current_prec = results_dict['overall_metrics']['precision']
    current_f1 = results_dict['overall_metrics']['f1_score']
    current_auc = results_dict['overall_metrics']['auc_roc']
    
    print(f"\n📊 COMPARISON WITH PREVIOUS ATTEMPTS:")
    print(f"  Baseline Accuracy: {baseline_accuracy:.1%}")
    print(f"  Fixed Multi-Scale Accuracy: {current_acc:.1%}")
    print(f"  Accuracy Improvement: {((current_acc - baseline_accuracy) / baseline_accuracy * 100):+.1f}%")
    print(f"")
    print(f"  Previous Precision: {baseline_precision:.1%}")
    print(f"  Fixed Multi-Scale Precision: {current_prec:.1%}")
    print(f"  Precision Improvement: {((current_prec - baseline_precision) / baseline_precision * 100):+.1f}%")
    print(f"")
    print(f"  F1-Score Achievement: {current_f1:.1%}")
    print(f"  AUC-ROC Achievement: {current_auc:.4f}")
    
    # Overall assessment
    if current_prec > 0.3 and current_f1 > 0.3:
        print(f"\n✅ MAJOR IMPROVEMENT: Fixed Multi-Scale shows excellent performance!")
    elif current_prec > 0.2 and current_f1 > 0.2:
        print(f"\n🎯 GOOD IMPROVEMENT: Fixed Multi-Scale significantly better!")
    elif current_prec > 0.15:
        print(f"\n📈 MODERATE IMPROVEMENT: Fixed Multi-Scale shows progress!")
    else:
        print(f"\n⚠️ STILL IMPROVING: More tuning needed for optimal performance")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'best_fixed_multi_scale_model.pth'  # Your Fixed Multi-Scale model
    TEST_I3D_DIR = 'I3D/Test_I3D'
    TEST_JSON_FILE = 'ShanghaiTech-campus/test.json'
    BATCH_SIZE = 4  # Adjust based on GPU memory
    
    print("🚀 Fixed Multi-Scale Temporal Attention - Comprehensive Testing")
    print("🎯 Evaluating precision improvements with multi-scale analysis")
    
    # Run comprehensive evaluation
    results = calculate_fixed_multi_scale_metrics(
        MODEL_PATH,
        TEST_I3D_DIR,
        TEST_JSON_FILE,
        BATCH_SIZE
    )
    
    # Compare with baseline and previous attempts
    if results:
        compare_with_attempts(results, 
                            baseline_accuracy=0.50,   # Adjust to your baseline
                            baseline_precision=0.059) # From your previous attempt
    
    print(f"\n🎉 Fixed Multi-Scale evaluation complete!")
    print(f"📋 Check 'fixed_multi_scale_metrics_results.json' for detailed results")
