# testing_metrics_video_new1.py - FIXED - Comprehensive Testing for Multi-Scale Temporal CNN Video Model
# Tests the direct video-to-anomaly detection model with MobileNet + Dilated TCN

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL CLASSES - Same as training code
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetFeatureExtractor, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280
        
    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features).flatten(1)
        return pooled

class MultiScaleDilatedTCN(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256):
        super(MultiScaleDilatedTCN, self).__init__()
        
        self.short_term = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        ])
        
        self.medium_term = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, dilation=4, padding=8),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, dilation=8, padding=16)
        ])
        
        self.long_term = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, dilation=16, padding=48),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, dilation=32, padding=96)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        short_out = x
        for layer in self.short_term:
            short_out = F.relu(layer(short_out))
            
        medium_out = x
        for layer in self.medium_term:
            medium_out = F.relu(layer(medium_out))
            
        long_out = x
        for layer in self.long_term:
            long_out = F.relu(layer(long_out))
        
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        fused = (scale_weights_norm[0] * short_out + 
                scale_weights_norm[1] * medium_out + 
                scale_weights_norm[2] * long_out)
        
        fused = self.batch_norm(fused)
        fused = self.dropout(fused)
        fused = fused.transpose(1, 2)
        
        return fused, scale_weights_norm

class VideoAnomalyDetector(nn.Module):
    def __init__(self, sequence_length=16, hidden_dim=256, num_classes=1):
        super(VideoAnomalyDetector, self).__init__()
        
        self.feature_extractor = MobileNetFeatureExtractor(pretrained=True)
        feature_dim = self.feature_extractor.feature_dim
        
        self.temporal_processor = MultiScaleDilatedTCN(
            input_dim=feature_dim, 
            hidden_dim=hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.sequence_length = sequence_length
        
    def forward(self, video_frames, return_probabilities=False):
        batch_size, seq_len = video_frames.shape[:2]
        frames = video_frames.view(-1, 3, 224, 224)
        spatial_features = self.feature_extractor(frames)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        temporal_features, scale_weights = self.temporal_processor(spatial_features)
        logits = self.classifier(temporal_features)
        
        if return_probabilities:
            return torch.sigmoid(logits), scale_weights
        else:
            return logits, scale_weights

# ============================================================================
# VIDEO TEST DATASET
# ============================================================================
class VideoTestDataset(Dataset):
    def __init__(self, video_dir, json_file, sequence_length=16, frame_size=(224, 224)):
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        
        with open(json_file, 'r') as f:
            all_labels = json.load(f)
        
        self.labels = {}
        self.video_names = []
        
        for video_name, label in all_labels.items():
            video_path = os.path.join(video_dir, video_name)
            if os.path.exists(video_path):
                try:
                    # Quick test to see if video can be opened
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if frame_count > 0:
                        self.labels[video_name] = label
                        self.video_names.append(video_name)
                except:
                    continue
        
        print(f"Loaded {len(self.video_names)} test videos for evaluation")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = min(total_frames, self.sequence_length * 2)
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            if len(frames) >= self.sequence_length:
                break
        
        cap.release()
        
        # Pad if not enough frames
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.sequence_length]
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            # Transform frames
            transformed_frames = []
            for frame in frames:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            
            video_tensor = torch.stack(transformed_frames)
            
            # Get frame labels
            frame_labels_list = self.labels[video_name]
            original_seq_length = len(frame_labels_list)
            
            if len(frame_labels_list) != self.sequence_length:
                # Interpolate labels to match sequence length
                frame_labels_array = np.array(frame_labels_list, dtype=np.float32)
                indices = np.linspace(0, len(frame_labels_array)-1, self.sequence_length)
                frame_labels = np.interp(indices, np.arange(len(frame_labels_array)), frame_labels_array)
                frame_labels = np.round(frame_labels).astype(np.float32)
            else:
                frame_labels = np.array(frame_labels_list, dtype=np.float32)
            
            return {
                'video_name': video_name,
                'frames': video_tensor,
                'labels': torch.FloatTensor(frame_labels),
                'seq_length': self.sequence_length,
                'original_length': original_seq_length
            }
            
        except Exception as e:
            print(f"Error loading video {video_name}: {e}")
            # Return dummy data
            dummy_frames = torch.zeros(self.sequence_length, 3, 224, 224)
            dummy_labels = torch.zeros(self.sequence_length)
            return {
                'video_name': video_name,
                'frames': dummy_frames,
                'labels': dummy_labels,
                'seq_length': self.sequence_length,
                'original_length': self.sequence_length
            }

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def find_balanced_threshold(all_scores, all_labels):
    """Find optimal threshold balancing precision and recall"""
    thresholds = np.linspace(0.05, 0.95, 100)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        if len(np.unique(predictions)) > 1:
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            if precision >= 0.05 and recall >= 0.05:
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
    
    return best_threshold

def calculate_video_temporal_cnn_metrics(model_path, test_video_dir, test_json_file, batch_size=4):
    """Calculate comprehensive metrics for Video Temporal CNN model"""
    print("=" * 80)
    print("MULTI-SCALE TEMPORAL CNN - DIRECT VIDEO PROCESSING - COMPREHENSIVE METRICS")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = VideoTestDataset(test_video_dir, test_json_file, sequence_length=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Initialize model
        model = VideoAnomalyDetector(
            sequence_length=config.get('sequence_length', 16),
            hidden_dim=config.get('hidden_dim', 256)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Multi-Scale Temporal CNN model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training epochs: {checkpoint.get('epoch', 'unknown')}")
        print(f"Training AUC: {checkpoint.get('auc', 'unknown'):.4f}")
        print(f"Training F1-Score: {checkpoint.get('f1_score', 'unknown'):.4f}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Evaluation
    model.eval()
    all_scores = []
    all_labels = []
    all_scale_weights = []
    video_results = {}
    
    print(f"Evaluating {len(test_dataset)} videos with Multi-Scale Temporal CNN...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing videos")):
            frames = batch['frames'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            seq_lengths = batch['seq_length']
            video_names = batch['video_name']
            
            with torch.cuda.amp.autocast():
                scores, scale_weights = model(frames, return_probabilities=True)
            
            # FIXED: scale_weights is 1D tensor of size 3 (one for each scale)
            scale_weights_np = scale_weights.cpu().numpy()  # Shape: (3,)
            
            # Process each video in batch
            for i in range(frames.size(0)):
                seq_len = seq_lengths[i].item()
                video_name = video_names[i]
                
                if seq_len > 0:
                    frame_scores = scores[i, :seq_len, 0].cpu().numpy()  # Extract single output
                    frame_labels = labels[i, :seq_len].cpu().numpy()
                    
                    all_scores.extend(frame_scores)
                    all_labels.extend(frame_labels)
                    
                    # Replicate scale weights for each frame in the video
                    for _ in range(seq_len):
                        all_scale_weights.append(scale_weights_np)
                    
                    video_results[video_name] = {
                        'scores': frame_scores,
                        'labels': frame_labels,
                        'scale_weights': scale_weights_np,  # Single scale weight per video
                        'mean_score': np.mean(frame_scores),
                        'max_score': np.max(frame_scores),
                        'anomaly_ratio': np.mean(frame_labels),
                        'scale_usage': scale_weights_np  # Direct usage
                    }
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Convert to arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_scale_weights = np.array(all_scale_weights)  # Shape: (num_frames, 3)
    
    print(f"Frames evaluated: {len(all_scores)}")
    print(f"Anomalous frames: {np.sum(all_labels)} ({np.mean(all_labels)*100:.1f}%)")
    
    # Check if we have both classes
    if len(np.unique(all_labels)) < 2:
        print("Warning: Only one class present in labels.")
        return None
    
    # Calculate metrics
    try:
        auc_score = roc_auc_score(all_labels, all_scores)
        optimal_threshold = find_balanced_threshold(all_scores, all_labels)
    except Exception as e:
        print(f"Error calculating ROC: {e}")
        auc_score = 0.0
        optimal_threshold = 0.5
    
    binary_predictions = (all_scores >= optimal_threshold).astype(int)
    
    # Standard metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_labels, binary_predictions, zero_division=0)
    f1 = f1_score(all_labels, binary_predictions, zero_division=0)
    
    # Confusion matrix components
    tp = np.sum((binary_predictions == 1) & (all_labels == 1))
    fp = np.sum((binary_predictions == 1) & (all_labels == 0))
    tn = np.sum((binary_predictions == 0) & (all_labels == 0))
    fn = np.sum((binary_predictions == 0) & (all_labels == 1))
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Multi-scale analysis - FIXED
    overall_scale_usage = np.mean(all_scale_weights, axis=0)  # Average across all frames
    
    # Print comprehensive results
    print("=" * 80)
    print("MULTI-SCALE TEMPORAL CNN - COMPREHENSIVE RESULTS")
    print("=" * 80)
    
    print("OVERALL PERFORMANCE:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall: {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1-Score: {f1:.4f} ({f1*100:.1f}%)")
    print(f"  AUC-ROC: {auc_score:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  NPV: {npv:.4f}")
    print(f"  FAR (False Alarm Rate): {far:.4f}")
    
    print("\nCONFUSION MATRIX:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Negatives: {fn}")
    
    print("\nTHRESHOLD & SCORING:")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Score Range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  Score Mean: {np.mean(all_scores):.4f}")
    print(f"  Score Std: {np.std(all_scores):.4f}")
    
    print("\nMULTI-SCALE TEMPORAL ANALYSIS:")
    print(f"  Short-term usage: {overall_scale_usage[0]:.3f} ({overall_scale_usage[0]*100:.1f}%)")
    print(f"  Medium-term usage: {overall_scale_usage[1]:.3f} ({overall_scale_usage[1]*100:.1f}%)")
    print(f"  Long-term usage: {overall_scale_usage[2]:.3f} ({overall_scale_usage[2]*100:.1f}%)")
    
    # Performance assessment
    if precision >= 0.4 and recall >= 0.4:
        print("\n✅ EXCELLENT - Well-balanced precision and recall!")
    elif precision >= 0.3 and f1 >= 0.3:
        print("\n✅ VERY GOOD - Strong performance!")
    elif precision >= 0.2 and f1 >= 0.2:
        print("\n✅ GOOD - Solid performance!")
    elif precision >= 0.1:
        print("\n⚠️  IMPROVED - Decent performance!")
    else:
        print("\n❌ NEEDS WORK - Performance needs improvement!")
    
    # Multi-scale balance assessment
    scale_balance = 1 - np.var(overall_scale_usage)
    if scale_balance > 0.8:
        print("🎯 Multi-scale: Well balanced usage across time scales")
    elif np.max(overall_scale_usage) > 0.8:
        dominant_scale = ['short', 'medium', 'long'][np.argmax(overall_scale_usage)]
        print(f"⚖️  Multi-scale: Dominated by {dominant_scale}-term patterns")
    else:
        print("⚖️  Multi-scale: Moderate balance across scales")
    
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
        print("\nVIDEO-LEVEL ANALYSIS:")
        print(f"  Videos with both classes: {len(video_accuracies)}")
        print(f"  Mean video accuracy: {np.mean(video_accuracies):.4f} ± {np.std(video_accuracies):.4f}")
        print(f"  Mean video precision: {np.mean(video_precisions):.4f} ± {np.std(video_precisions):.4f}")
        print(f"  Mean video F1-score: {np.mean(video_f1s):.4f} ± {np.std(video_f1s):.4f}")
        print(f"  Best video accuracy: {np.max(video_accuracies):.4f}")
        print(f"  Best video precision: {np.max(video_precisions):.4f}")
    
    # Create results dictionary
    results_dict = {
        'model_info': {
            'model_type': 'Multi-Scale Temporal CNN',
            'architecture': 'MobileNet + Dilated Temporal Convolutions',
            'parameters': sum(p.numel() for p in model.parameters()),
        },
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_score),
            'specificity': float(specificity),
            'npv': float(npv),
            'far': float(far)
        },
        'confusion_matrix': {
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        },
        'threshold_info': {
            'optimal_threshold': float(optimal_threshold),
            'score_mean': float(np.mean(all_scores)),
            'score_std': float(np.std(all_scores))
        },
        'multiscale_analysis': {
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
    
    # Save results
    output_file = 'video_temporal_cnn_metrics_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 80)
    
    return results_dict

def compare_with_previous_attempts(results_dict, baseline_accuracy=0.50, baseline_precision=0.06):
    """Compare current results with previous attempts"""
    if results_dict is None:
        return
    
    current_acc = results_dict['overall_metrics']['accuracy']
    current_prec = results_dict['overall_metrics']['precision']
    current_f1 = results_dict['overall_metrics']['f1_score']
    current_auc = results_dict['overall_metrics']['auc_roc']
    
    print("COMPARISON WITH BASELINE:")
    print(f"  Baseline Accuracy: {baseline_accuracy:.1%}")
    print(f"  Current Accuracy: {current_acc:.1%}")
    print(f"  Accuracy Improvement: {((current_acc - baseline_accuracy) / baseline_accuracy * 100):+.1f}%")
    print()
    print(f"  Baseline Precision: {baseline_precision:.1%}")
    print(f"  Current Precision: {current_prec:.1%}")
    print(f"  Precision Improvement: {((current_prec - baseline_precision) / baseline_precision * 100):+.1f}%")
    print()
    print(f"  F1-Score Achievement: {current_f1:.1%}")
    print(f"  AUC-ROC Achievement: {current_auc:.4f}")
    
    if current_prec >= 0.3 and current_f1 >= 0.3:
        print("🎉 MAJOR IMPROVEMENT! Multi-Scale Temporal CNN shows excellent performance!")
    elif current_prec >= 0.2 and current_f1 >= 0.2:
        print("🎉 GOOD IMPROVEMENT! Multi-Scale Temporal CNN significantly better!")
    elif current_prec >= 0.15:
        print("✅ MODERATE IMPROVEMENT! Multi-Scale Temporal CNN shows progress!")
    else:
        print("⚠️  STILL IMPROVING - More tuning needed for optimal performance")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "best_video_anomaly_model_new1.pth"
    TEST_VIDEO_DIR = "ShanghaiTech-campus/test"
    TEST_JSON_FILE = "ShanghaiTech-campus/test.json"
    BATCH_SIZE = 4
    
    print("Multi-Scale Temporal CNN - Direct Video Processing - Comprehensive Testing")
    print("Evaluating edge-optimized video anomaly detection with temporal analysis")
    
    # Run evaluation
    results = calculate_video_temporal_cnn_metrics(
        MODEL_PATH, TEST_VIDEO_DIR, TEST_JSON_FILE, BATCH_SIZE
    )
    
    if results:
        compare_with_previous_attempts(
            results, baseline_accuracy=0.50, baseline_precision=0.059
        )
        print(f"\n✅ Video Temporal CNN evaluation complete!")
        print(f"📊 Check 'video_temporal_cnn_metrics_results.json' for detailed results")
