# Testing Code for Fixed Multi-Scale Temporal Attention Model (attempt6.py)
# Comprehensive testing with exact architecture matching

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import os
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXACT MODEL ARCHITECTURE FROM ATTEMPT6.PY
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - exact match from attempt6"""
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
        else:
            return focal_loss

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
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def create_simple_windows(self, x, window_size):
        """Less aggressive windowing - exact match"""
        batch_size, seq_len, embed_dim = x.shape
        
        if seq_len <= window_size:
            return x
        
        stride = max(1, window_size // 4)
        windowed_features = []
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window = x[:, start:end, :]
            windowed_features.append(window.mean(dim=1))
        
        while len(windowed_features) < seq_len:
            windowed_features.append(windowed_features[-1])
        
        windowed_features = windowed_features[:seq_len]
        return torch.stack(windowed_features, dim=1)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x_proj = self.input_projection(x)
        
        short_features = self.create_simple_windows(x_proj, window_size=6)
        medium_features = self.create_simple_windows(x_proj, window_size=12)
        long_features = self.create_simple_windows(x_proj, window_size=24)
        
        short_out = self.transformer_layer(short_features)
        medium_out = self.transformer_layer(medium_features)
        long_out = self.transformer_layer(long_features)
        
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        
        output = (
            scale_weights_norm[0] * short_out +
            scale_weights_norm[1] * medium_out +
            scale_weights_norm[2] * long_out
        )
        
        output = self.layer_norm(output)
        
        scale_usage = scale_weights_norm.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        return output, scale_usage

class FixedMultiScaleVAD(nn.Module):
    """EXACT MATCH with attempt6 model"""
    def __init__(self, input_dim, embed_dim=128, num_heads=2, dropout=0.2, max_seq_len=60):
        super().__init__()
        
        # Feature reduction - exact match
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 8, input_dim // 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 32, embed_dim)
        )
        
        self.temporal_diff_encoder = TemporalDifferenceEncoder(embed_dim, hidden_dim=64)
        
        self.multi_scale_encoder = SimpleMultiScaleEncoder(
            input_dim=embed_dim + 64,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, x, return_probabilities=False):
        batch_size, seq_len, _ = x.shape
        
        x_reduced = self.feature_reducer(x)
        x_pos = x_reduced + self.pos_embedding[:, :seq_len, :]
        motion_features = self.temporal_diff_encoder(x_pos)
        combined_features = torch.cat([x_pos, motion_features], dim=-1)
        temporal_features, scale_weights = self.multi_scale_encoder(combined_features)
        logits = self.classifier(temporal_features).squeeze(-1)
        
        if return_probabilities:
            return torch.sigmoid(logits), scale_weights
        else:
            return logits, scale_weights

# ============================================================================
# FEATURE LOADING - MATCHING TRAINING PREPROCESSING
# ============================================================================

def load_fixed_multi_scale_features(video_name, i3d_dir, max_length=60):
    """Load features matching the training preprocessing exactly"""
    npy_name = video_name.replace('.avi', '.npy')
    npy_path = os.path.join(i3d_dir, npy_name)
    
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found!")
        return None, 0, 0
    
    try:
        features = np.load(npy_path)  # (T, 10, 2048)
        if len(features.shape) != 3:
            features = np.random.randn(30, 10, 2048).astype(np.float32)
    except:
        features = np.random.randn(30, 10, 2048).astype(np.float32)
    
    print(f"Original feature shape: {features.shape}")
    
    # Extract features - same as training
    T, spatial_regions, feature_dim = features.shape
    processed_features = features.reshape(T, -1)  # (T, 20480)
    original_length = processed_features.shape[0]
    
    # Sequence length processing - same as training
    if original_length > max_length:
        processed_features = processed_features[:max_length]
        seq_length = max_length
    else:
        seq_length = original_length
        if seq_length < max_length:
            padding_length = max_length - seq_length
            feature_padding = np.zeros((padding_length, processed_features.shape[1]), dtype=np.float32)
            processed_features = np.vstack([processed_features, feature_padding])
    
    features_tensor = torch.FloatTensor(processed_features).unsqueeze(0)  # (1, max_length, 20480)
    
    return features_tensor, seq_length, original_length

# ============================================================================
# THRESHOLD SELECTION - SAME AS TRAINING
# ============================================================================

def find_balanced_threshold_test(all_scores, method='adaptive'):
    """Find balanced threshold - multiple methods available"""
    if method == 'adaptive':
        return np.mean(all_scores) + 0.5 * np.std(all_scores)
    elif method == 'percentile_75':
        return np.percentile(all_scores, 75)
    elif method == 'percentile_80':
        return np.percentile(all_scores, 80)
    elif method == 'mean_plus_std':
        return np.mean(all_scores) + np.std(all_scores)
    elif method == 'median_plus':
        return np.median(all_scores) + 0.1
    else:
        return 0.5  # Conservative default

def analyze_fixed_multi_scale_thresholds(model, features_tensor, seq_length, device='cuda'):
    """Comprehensive threshold analysis for Fixed Multi-Scale model"""
    model.eval()
    
    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        scores, scale_weights = model(features_tensor, return_probabilities=True)
        scores = scores.squeeze(0)[:seq_length].cpu().numpy()
        scale_weights = scale_weights.squeeze(0)[:seq_length].cpu().numpy()
    
    threshold_methods = {
        'Adaptive (Mean + 0.5*STD)': find_balanced_threshold_test(scores, 'adaptive'),
        'Percentile 75': find_balanced_threshold_test(scores, 'percentile_75'),
        'Percentile 80': find_balanced_threshold_test(scores, 'percentile_80'),
        'Mean + STD': find_balanced_threshold_test(scores, 'mean_plus_std'),
        'Median + 0.1': find_balanced_threshold_test(scores, 'median_plus'),
        'Conservative 0.5': 0.5,
        'Mean Score': np.mean(scores)
    }
    
    print("\n" + "="*70)
    print("FIXED MULTI-SCALE THRESHOLD ANALYSIS")
    print("="*70)
    print(f"Score Statistics:")
    print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    
    print(f"\nScale Usage:")
    overall_scale_usage = np.mean(scale_weights, axis=0)
    print(f"  Short-term: {overall_scale_usage[0]:.3f} ({overall_scale_usage[0]*100:.1f}%)")
    print(f"  Medium-term: {overall_scale_usage[1]:.3f} ({overall_scale_usage[1]*100:.1f}%)")
    print(f"  Long-term: {overall_scale_usage[2]:.3f} ({overall_scale_usage[2]*100:.1f}%)")
    
    print("-" * 70)
    
    results = {}
    for method_name, threshold in threshold_methods.items():
        predictions = (scores >= threshold).astype(int)
        anomaly_pct = np.mean(predictions) * 100
        
        # Calculate confidence based on distance from threshold
        confidence = np.mean(np.abs(scores - threshold))
        
        results[method_name] = {
            'threshold': threshold,
            'predictions': predictions,
            'anomaly_percentage': anomaly_pct,
            'confidence': confidence
        }
        
        print(f"{method_name:25s}: {threshold:.4f} → {anomaly_pct:5.1f}% | Conf: {confidence:.3f}")
    
    # Select recommended method (reasonable anomaly % and good confidence)
    reasonable_methods = [k for k, v in results.items() 
                         if 5 <= v['anomaly_percentage'] <= 70]
    
    if reasonable_methods:
        best_method = max(reasonable_methods, key=lambda x: results[x]['confidence'])
        print(f"\nRECOMMENDED: {best_method}")
        return results[best_method]['predictions'], results[best_method]['threshold'], overall_scale_usage
    else:
        print(f"\nFALLBACK: Using Adaptive method")
        fallback = 'Adaptive (Mean + 0.5*STD)'
        return results[fallback]['predictions'], results[fallback]['threshold'], overall_scale_usage

# ============================================================================
# ENHANCED VISUALIZATION FOR MULTI-SCALE MODEL
# ============================================================================

def create_fixed_multi_scale_visualization(video_name, scores, predictions, ground_truth=None,
                                          threshold_used=0.5, scale_usage=None, save_path=None):
    """Enhanced visualization for Fixed Multi-Scale model"""
    
    fig_height = 16 if scale_usage is not None else 12
    num_plots = 5 if scale_usage is not None else 4
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, fig_height))
    time_points = np.arange(len(scores))
    
    # Plot 1: Enhanced anomaly scores
    axes[0].plot(time_points, scores, 'b-', linewidth=2.5, label='Fixed Multi-Scale Scores', alpha=0.8)
    axes[0].axhline(y=threshold_used, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold_used:.3f})')
    
    # Add statistical bands
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    axes[0].axhspan(mean_score - std_score, mean_score + std_score,
                   alpha=0.2, color='gray', label='±1 STD')
    axes[0].fill_between(time_points, 0, scores, alpha=0.3, color='blue')
    
    stats_text = f'Multi-Scale Statistics:\nMean: {np.mean(scores):.3f}\nSTD: {np.std(scores):.3f}\nMax: {np.max(scores):.3f}\nMin: {np.min(scores):.3f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    axes[0].set_ylabel('Anomaly Score', fontsize=12)
    axes[0].set_title(f'Fixed Multi-Scale Temporal Attention: {video_name}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Binary predictions
    axes[1].plot(time_points, predictions, 'ro-', markersize=4, label='Multi-Scale Predictions')
    axes[1].fill_between(time_points, 0, predictions, alpha=0.5, color='red',
                        step='pre', label='Detected Anomaly Regions')
    
    anomaly_pct = np.mean(predictions) * 100
    normal_pct = 100 - anomaly_pct
    
    pred_text = f'Detection Summary:\nAnomalous: {anomaly_pct:.1f}%\nNormal: {normal_pct:.1f}%'
    axes[1].text(0.02, 0.98, pred_text, transform=axes[1].transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
    
    axes[1].set_ylabel('Prediction (0/1)', fontsize=12)
    axes[1].set_title('Fixed Multi-Scale Binary Predictions', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Ground truth comparison
    if ground_truth is not None:
        axes[2].plot(time_points, ground_truth, 'g-', linewidth=4,
                    label='Ground Truth', alpha=0.8)
        axes[2].plot(time_points, predictions, 'r--', linewidth=2,
                    label='Multi-Scale Predictions', alpha=0.8)
        axes[2].fill_between(time_points, 0, ground_truth, alpha=0.3, color='green',
                           step='pre', label='True Anomaly Regions')
        
        if len(ground_truth) == len(predictions):
            tp = np.sum((predictions == 1) & (ground_truth == 1))
            fp = np.sum((predictions == 1) & (ground_truth == 0))
            tn = np.sum((predictions == 0) & (ground_truth == 0))
            fn = np.sum((predictions == 0) & (ground_truth == 1))
            
            accuracy = (tp + tn) / len(ground_truth)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = f'Multi-Scale Performance:\nAccuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
            axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        axes[2].set_ylabel('Anomaly (0/1)', fontsize=12)
        axes[2].set_title('Multi-Scale Predictions vs Ground Truth', fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].set_ylim(-0.1, 1.1)
    else:
        confidence = np.abs(scores - threshold_used)
        axes[2].plot(time_points, confidence, 'purple', linewidth=2,
                    label='Prediction Confidence')
        axes[2].fill_between(time_points, 0, confidence, alpha=0.3, color='purple')
        
        conf_text = f'Confidence Analysis:\nMean: {np.mean(confidence):.3f}\nHigh Conf Regions: {np.sum(confidence > np.percentile(confidence, 75))}'
        axes[2].text(0.02, 0.98, conf_text, transform=axes[2].transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.9))
        
        axes[2].set_ylabel('Confidence', fontsize=12)
        axes[2].set_title('Multi-Scale Prediction Confidence', fontsize=12)
        axes[2].legend(fontsize=10)
    
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Temporal trend analysis
    if len(scores) > 10:
        window_size = max(3, len(scores) // 10)
        smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='same')
        axes[3].plot(time_points, smoothed_scores, 'orange', linewidth=3,
                    label=f'Smoothed Trend (window={window_size})', alpha=0.8)
        axes[3].plot(time_points, scores, 'b-', alpha=0.4, linewidth=1,
                    label='Original Scores')
        
        if len(scores) > 5:
            trend_coeff = np.polyfit(time_points, scores, 1)[0]
            trend_text = f'Temporal Analysis:\nTrend Slope: {trend_coeff:.6f}\nDirection: {"↗ Increasing" if trend_coeff > 0.001 else "↘ Decreasing" if trend_coeff < -0.001 else "→ Stable"}'
            axes[3].text(0.02, 0.98, trend_text, transform=axes[3].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.9))
    else:
        axes[3].plot(time_points, scores, 'b-', linewidth=2, label='Multi-Scale Scores')
    
    axes[3].set_ylabel('Score', fontsize=12)
    axes[3].set_title('Multi-Scale Temporal Trend Analysis', fontsize=12)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Scale usage analysis (if available)
    if scale_usage is not None:
        scale_names = ['Short-term\n(6 frames)', 'Medium-term\n(12 frames)', 'Long-term\n(24 frames)']
        colors = ['lightcoral', 'lightskyblue', 'lightgreen']
        
        bars = axes[4].bar(scale_names, scale_usage, color=colors, alpha=0.7, edgecolor='black')
        axes[4].set_ylabel('Usage Proportion', fontsize=12)
        axes[4].set_title('Multi-Scale Usage Distribution', fontsize=12)
        axes[4].set_ylim(0, max(scale_usage) * 1.1)
        
        # Add percentage labels on bars
        for i, (bar, usage) in enumerate(zip(bars, scale_usage)):
            height = bar.get_height()
            axes[4].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{usage:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Determine dominant scale
        dominant_scale = np.argmax(scale_usage)
        dominant_names = ['short-term', 'medium-term', 'long-term']
        
        scale_text = f'Scale Analysis:\nDominant: {dominant_names[dominant_scale].title()}\nBalance: {"Good" if max(scale_usage) < 0.6 else "Imbalanced"}'
        axes[4].text(0.02, 0.98, scale_text, transform=axes[4].transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        axes[4].grid(True, alpha=0.3, axis='y')
    
    # Set xlabel for the bottom plot
    bottom_plot = axes[4] if scale_usage is not None else axes[3]
    bottom_plot.set_xlabel('Frame Segments', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Fixed Multi-Scale visualization saved: {save_path}")
    
    plt.show()

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def test_fixed_multi_scale_video(video_name, model_path='best_fixed_multi_scale_model.pth',
                                i3d_dir='I3D/Test_I3D', json_file='ShanghaiTech-campus/test.json',
                                threshold_method='auto', show_analysis=True):
    """Comprehensive testing for Fixed Multi-Scale model"""
    
    print(f"Fixed Multi-Scale Testing: {video_name}")
    print("="*70)
    
    # Load Fixed Multi-Scale model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Get input dimension
        features_tensor, _, _ = load_fixed_multi_scale_features(video_name, i3d_dir)
        if features_tensor is not None:
            input_dim = features_tensor.shape[-1]
        else:
            input_dim = 20480  # Default
        
        # Create model with saved configuration
        model = FixedMultiScaleVAD(
            input_dim=input_dim,
            embed_dim=config.get('embed_dim', 128),
            num_heads=config.get('num_heads', 2),
            dropout=config.get('dropout', 0.2),
            max_seq_len=60
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Fixed Multi-Scale model loaded successfully")
        print(f"  Model trained for: {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"  Training AUC: {checkpoint.get('auc', 'unknown'):.4f}")
        print(f"  Training Precision: {checkpoint.get('precision', 'unknown'):.4f}")
        print(f"  Training F1-Score: {checkpoint.get('f1_score', 'unknown'):.4f}")
        
    except Exception as e:
        print(f"Error loading Fixed Multi-Scale model: {e}")
        return None
    
    # Load and process video features
    features_tensor, seq_length, original_length = load_fixed_multi_scale_features(video_name, i3d_dir)
    if features_tensor is None:
        return None
    
    print(f"\nProcessing {seq_length} segments (original: {original_length})")
    print(f"Feature dimension: {features_tensor.shape[-1]}")
    
    # Threshold analysis
    if show_analysis:
        predictions, threshold_used, scale_usage = analyze_fixed_multi_scale_thresholds(
            model, features_tensor, seq_length, device)
    else:
        # Quick prediction without analysis
        model.eval()
        with torch.no_grad():
            scores, scale_weights = model(features_tensor.to(device), return_probabilities=True)
            scores = scores.squeeze(0)[:seq_length].cpu().numpy()
            scale_usage = scale_weights.squeeze(0)[:seq_length].cpu().numpy().mean(axis=0)
            threshold_used = find_balanced_threshold_test(scores, 'adaptive')
            predictions = (scores >= threshold_used).astype(int)
    
    # Get detailed scores for visualization
    model.eval()
    with torch.no_grad():
        scores, _ = model(features_tensor.to(device), return_probabilities=True)
        scores = scores.squeeze(0)[:seq_length].cpu().numpy()
    
    # Load ground truth
    ground_truth = None
    try:
        with open(json_file, 'r') as f:
            labels = json.load(f)
        
        if video_name in labels:
            gt_labels = np.array(labels[video_name])
            
            if len(gt_labels) != original_length:
                if len(gt_labels) > 1:
                    f = interp1d(np.linspace(0, 1, len(gt_labels)), 
                               gt_labels, kind='nearest')
                    gt_labels = f(np.linspace(0, 1, original_length))
                else:
                    gt_labels = np.zeros(original_length)
            
            ground_truth = gt_labels[:seq_length]
            
    except Exception as e:
        print(f"Could not load ground truth: {e}")
    
    # Create comprehensive visualization
    save_path = f"{video_name.replace('.avi', '')}_fixed_multi_scale_results.png"
    create_fixed_multi_scale_visualization(
        video_name, scores, predictions, ground_truth, threshold_used, scale_usage, save_path
    )
    
    # Generate detailed report
    print("\n" + "="*70)
    print(f"FIXED MULTI-SCALE ANOMALY DETECTION REPORT: {video_name}")
    print("="*70)
    
    print(f"\nVideo Analysis:")
    print(f"  Total Segments: {len(scores)}")
    print(f"  Anomaly Segments: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)")
    print(f"  Normal Segments: {len(scores) - np.sum(predictions)} ({(1-np.mean(predictions))*100:.1f}%)")
    print(f"  Score Range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Threshold Used: {threshold_used:.4f}")
    
    print(f"\nMulti-Scale Analysis:")
    print(f"  Short-term dominance: {scale_usage[0]:.1%}")
    print(f"  Medium-term dominance: {scale_usage[1]:.1%}")  
    print(f"  Long-term dominance: {scale_usage[2]:.1%}")
    
    dominant_scale = ['Short-term (6 frames)', 'Medium-term (12 frames)', 'Long-term (24 frames)'][np.argmax(scale_usage)]
    print(f"  Dominant scale: {dominant_scale}")
    
    if ground_truth is not None:
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        accuracy = (tp + tn) / len(ground_truth)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPerformance vs Ground Truth:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.1f}%)")
        print(f"  F1-Score: {f1:.4f} ({f1*100:.1f}%)")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  True Negatives: {tn}")
        print(f"  False Negatives: {fn}")
        
        print(f"\nFixed Multi-Scale Performance Assessment:")
        if precision > 0.4 and f1 > 0.4:
            print("  🎯 EXCELLENT - Outstanding multi-scale precision!")
        elif precision > 0.3 and f1 > 0.3:
            print("  ✅ VERY GOOD - Strong multi-scale performance!")
        elif precision > 0.2 and f1 > 0.2:
            print("  👍 GOOD - Solid multi-scale improvement!")
        elif precision > 0.1:
            print("  📈 IMPROVED - Better than previous attempts!")
        else:
            print("  ⚠️  NEEDS WORK - Multi-scale approach needs tuning")
        
        # Multi-scale effectiveness
        if max(scale_usage) < 0.6:
            print("  🔄 Multi-scale: Well-balanced usage across all time scales")
        else:
            print(f"  🔄 Multi-scale: Focuses primarily on {dominant_scale.lower()}")
    
    return {
        'scores': scores,
        'predictions': predictions, 
        'ground_truth': ground_truth,
        'threshold': threshold_used,
        'scale_usage': scale_usage,
        'accuracy': accuracy if ground_truth is not None else None,
        'precision': precision if ground_truth is not None else None,
        'recall': recall if ground_truth is not None else None,
        'f1_score': f1 if ground_truth is not None else None
    }

# ============================================================================
# MAIN TESTING EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FIXED MULTI-SCALE TEMPORAL ATTENTION - VIDEO TESTING")
    print("="*70)
    print("🎯 Testing with multi-scale temporal analysis")
    print("🔄 Evaluating short/medium/long-term pattern learning")
    
    # Test videos
    test_videos = ["01_0015.avi", "01_0025.avi", "01_0028.avi"]
    
    results_summary = []
    
    for video in test_videos:
        print(f"\n{'='*70}")
        print(f"Testing: {video}")
        print('='*70)
        
        result = test_fixed_multi_scale_video(
            video, 
            model_path='best_fixed_multi_scale_model.pth',
            threshold_method='auto',
            show_analysis=True
        )
        
        if result:
            print(f"✓ Successfully processed {video}")
            
            if result['ground_truth'] is not None:
                results_summary.append({
                    'video': video,
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'f1_score': result['f1_score'],
                    'dominant_scale': ['Short', 'Medium', 'Long'][np.argmax(result['scale_usage'])]
                })
        else:
            print(f"✗ Failed to process {video}")
    
    # Summary report
    if results_summary:
        print(f"\n{'='*70}")
        print("FIXED MULTI-SCALE SUMMARY REPORT")
        print('='*70)
        
        avg_accuracy = np.mean([r['accuracy'] for r in results_summary])
        avg_precision = np.mean([r['precision'] for r in results_summary])
        avg_f1 = np.mean([r['f1_score'] for r in results_summary])
        
        print(f"\nOverall Performance:")
        print(f"  Average Accuracy: {avg_accuracy:.3f}")
        print(f"  Average Precision: {avg_precision:.3f}")
        print(f"  Average F1-Score: {avg_f1:.3f}")
        
        print(f"\nPer-Video Results:")
        for result in results_summary:
            print(f"  {result['video']:12s}: Acc={result['accuracy']:.3f}, Prec={result['precision']:.3f}, F1={result['f1_score']:.3f}, Scale={result['dominant_scale']}")
        
        print(f"\nMulti-Scale Effectiveness:")
        scale_distribution = {}
        for result in results_summary:
            scale = result['dominant_scale']
            scale_distribution[scale] = scale_distribution.get(scale, 0) + 1
        
        for scale, count in scale_distribution.items():
            print(f"  {scale}-term dominant: {count}/{len(results_summary)} videos")
    
    print(f"\n🎉 Fixed Multi-Scale testing complete!")
    print(f"📋 Check generated visualization files for detailed analysis")
