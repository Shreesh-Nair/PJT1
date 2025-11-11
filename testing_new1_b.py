# Updated Video Anomaly Detection - Single Video Testing for MobileNet+TCN Model
# Based on the visualization style from I3D testing code
# Testing video: 01_0015.avi (or change to your desired video)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import json
import os
from scipy.interpolate import interp1d
from torchvision import transforms, models
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL CLASSES (MobileNet + Dilated TCN)
# ============================================================================
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetFeatureExtractor, self).__init__()
        try:
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
        except:
            print("Warning: Could not load pretrained weights, using random initialization")
            mobilenet = models.mobilenet_v2(pretrained=False)
        
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
        
        self.feature_extractor = MobileNetFeatureExtractor(pretrained=False)  # Changed to False for loading
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
# VIDEO PROCESSING FUNCTIONS
# ============================================================================
def load_and_preprocess_video(video_name, video_dir, sequence_length=16):
    """
    Load and preprocess video frames for MobileNet+TCN model
    """
    video_path = os.path.join(video_dir, video_name)
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return None, 0, 0
    
    print(f"Loading video: {video_path}")
    
    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, 0, 0
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.1f} FPS")
    
    # Sample frames uniformly
    target_frames = min(total_frames, sequence_length * 3)  # Sample more for better coverage
    frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if len(frames) >= sequence_length:
            break
    
    cap.release()
    
    original_length = len(frames)
    
    # Pad if needed
    while len(frames) < sequence_length:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Transform frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    for frame in frames[:sequence_length]:
        processed_frame = transform(frame)
        processed_frames.append(processed_frame)
    
    # Stack frames and add batch dimension
    video_tensor = torch.stack(processed_frames).unsqueeze(0)  # (1, sequence_length, 3, 224, 224)
    
    print(f"Processed {sequence_length} frames (original: {original_length})")
    print(f"Video tensor shape: {video_tensor.shape}")
    
    return video_tensor, sequence_length, original_length

def predict_video_anomalies(model, video_tensor, seq_length, device='cuda',
                           threshold_method='adaptive'):
    """
    Get anomaly predictions with different threshold methods
    """
    model.eval()
    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        
        # Get predictions
        scores, _ = model(video_tensor, return_probabilities=True)
        scores = scores.squeeze().cpu().numpy()  # (sequence_length,)
    
    # Different thresholding methods
    if threshold_method == 'original':
        threshold = 0.5  # Default threshold
    elif threshold_method == 'median':
        threshold = np.median(scores)
    elif threshold_method == 'adaptive':
        # Use mean + 0.5 * std as threshold (works well for individual videos)
        threshold = np.mean(scores) + 0.5 * np.std(scores)
    elif threshold_method == 'conservative':
        threshold = 0.7  # Conservative threshold
    else:
        threshold = float(threshold_method)  # Use as direct threshold value
    
    # Apply threshold for binary predictions
    predictions = (scores >= threshold).astype(int)
    
    print(f"Threshold method: {threshold_method}")
    print(f"Applied threshold: {threshold:.4f}")
    print(f"Score range: {scores.min():.4f} to {scores.max():.4f}")
    print(f"Predicted anomaly percentage: {np.mean(predictions)*100:.1f}%")
    
    return scores, predictions, threshold

def analyze_multiple_thresholds(model, video_tensor, seq_length, device='cuda'):
    """
    Test multiple threshold methods to find the best one
    """
    model.eval()
    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        scores, _ = model(video_tensor, return_probabilities=True)
        scores = scores.squeeze().cpu().numpy()
    
    threshold_methods = {
        'Conservative (0.7)': 0.7,
        'Default (0.5)': 0.5,
        'Mean': np.mean(scores),
        'Median': np.median(scores),
        'Mean + 0.5*STD': np.mean(scores) + 0.5 * np.std(scores),
        'Mean + STD': np.mean(scores) + np.std(scores),
        '75th Percentile': np.percentile(scores, 75),
        '90th Percentile': np.percentile(scores, 90)
    }
    
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    results = {}
    
    for method_name, threshold in threshold_methods.items():
        predictions = (scores >= threshold).astype(int)
        anomaly_pct = np.mean(predictions) * 100
        results[method_name] = {
            'threshold': threshold,
            'predictions': predictions,
            'anomaly_percentage': anomaly_pct
        }
        print(f"{method_name:20s}: {threshold:.4f} → {anomaly_pct:5.1f}% anomalies")
    
    # Suggest best threshold
    reasonable_methods = [k for k, v in results.items() 
                         if 5 <= v['anomaly_percentage'] <= 50]  # Reasonable range
    
    if reasonable_methods:
        suggested = reasonable_methods[0]
        print(f"\nSUGGESTED METHOD: {suggested}")
        return results[suggested]['predictions'], results[suggested]['threshold']
    else:
        print(f"\nUSING DEFAULT: All methods give extreme results, using 0.5")
        return results['Default (0.5)']['predictions'], 0.5

def visualize_results(video_name, scores, predictions, ground_truth=None, 
                     threshold_used=0.5, save_path=None):
    """
    Create comprehensive visualization with threshold info (same style as I3D version)
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    time_points = np.arange(len(scores))
    
    # Plot 1: Anomaly Scores with threshold
    axes[0].plot(time_points, scores, 'b-', linewidth=2, label='Anomaly Score')
    axes[0].axhline(y=threshold_used, color='r', linestyle='--', 
                   label=f'Threshold ({threshold_used:.4f})')
    axes[0].fill_between(time_points, 0, scores, alpha=0.3, color='blue')
    
    # Add score statistics
    axes[0].text(0.02, 0.98, f'Mean: {np.mean(scores):.3f}\nSTD: {np.std(scores):.3f}',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title(f'Video: {video_name} - Anomaly Detection Results')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Binary Predictions
    axes[1].plot(time_points, predictions, 'ro-', markersize=4, label='Predicted Anomalies')
    axes[1].fill_between(time_points, 0, predictions, alpha=0.5, color='red',
                        step='pre', label='Anomaly Regions')
    
    anomaly_pct = np.mean(predictions) * 100
    axes[1].text(0.02, 0.98, f'Anomaly: {anomaly_pct:.1f}%',
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    axes[1].set_ylabel('Anomaly (0/1)')
    axes[1].set_title('Binary Anomaly Predictions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Comparison with Ground Truth (if available)
    if ground_truth is not None:
        axes[2].plot(time_points, ground_truth, 'g-', linewidth=4,
                    label='Ground Truth', alpha=0.8)
        axes[2].plot(time_points, predictions, 'r--', linewidth=3,
                    label='Predictions', alpha=0.8)
        axes[2].fill_between(time_points, 0, ground_truth, alpha=0.3, color='green',
                           step='pre', label='True Anomalies')
        
        # Calculate and show accuracy
        if len(ground_truth) == len(predictions):
            accuracy = np.mean(predictions == ground_truth)
            axes[2].text(0.02, 0.98, f'Accuracy: {accuracy:.3f}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        axes[2].set_ylabel('Anomaly (0/1)')
        axes[2].set_title('Predictions vs Ground Truth')
        axes[2].legend()
        axes[2].set_ylim(-0.1, 1.1)
    else:
        # If no ground truth, show confidence intervals
        confidence = np.abs(scores - threshold_used)  # Distance from threshold
        axes[2].plot(time_points, confidence, 'purple', linewidth=2,
                    label='Distance from Threshold')
        axes[2].fill_between(time_points, 0, confidence, alpha=0.3, color='purple')
        axes[2].set_ylabel('Confidence')
        axes[2].set_title('Prediction Confidence (Distance from Threshold)')
        axes[2].legend()
    
    axes[2].set_xlabel('Frame Segments')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    
    plt.show()

def generate_text_report(video_name, scores, predictions, ground_truth=None, threshold_used=0.5):
    """
    Generate a clear text summary with better metrics (same style as I3D version)
    """
    print("\n" + "="*60)
    print(f"ANOMALY DETECTION REPORT: {video_name}")
    print("="*60)
    
    # Basic statistics
    total_segments = len(scores)
    anomaly_segments = np.sum(predictions)
    normal_segments = total_segments - anomaly_segments
    
    print(f"Video Length: {total_segments} segments")
    print(f"Threshold Used: {threshold_used:.4f}")
    print(f"Normal Segments: {normal_segments} ({normal_segments/total_segments*100:.1f}%)")
    print(f"Anomaly Segments: {anomaly_segments} ({anomaly_segments/total_segments*100:.1f}%)")
    
    # Score statistics
    print(f"\nAnomaly Score Statistics:")
    print(f" Average Score: {np.mean(scores):.4f}")
    print(f" Maximum Score: {np.max(scores):.4f}")
    print(f" Minimum Score: {np.min(scores):.4f}")
    print(f" Standard Deviation: {np.std(scores):.4f}")
    print(f" Median Score: {np.median(scores):.4f}")
    
    # Find anomaly regions
    if anomaly_segments > 0:
        print(f"\nDetected Anomaly Regions:")
        in_anomaly = False
        start_segment = 0
        
        for i, pred in enumerate(predictions):
            if pred == 1 and not in_anomaly:
                start_segment = i
                in_anomaly = True
            elif pred == 0 and in_anomaly:
                avg_score = np.mean(scores[start_segment:i])
                max_score = np.max(scores[start_segment:i])
                print(f" Segments {start_segment:2d}-{i-1:2d} (Avg: {avg_score:.3f}, Max: {max_score:.3f})")
                in_anomaly = False
        
        # Handle case where anomaly continues to end
        if in_anomaly:
            avg_score = np.mean(scores[start_segment:])
            max_score = np.max(scores[start_segment:])
            print(f" Segments {start_segment:2d}-{len(predictions)-1:2d} (Avg: {avg_score:.3f}, Max: {max_score:.3f})")
    else:
        print(f"\nNo anomalies detected with current threshold.")
    
    # Comparison with ground truth if available
    if ground_truth is not None:
        gt_anomalies = np.sum(ground_truth)
        
        # Calculate accuracy metrics
        correct_predictions = np.sum(predictions == ground_truth)
        accuracy = correct_predictions / len(ground_truth)
        
        # True/False positives/negatives
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n" + "-"*40)
        print(f"PERFORMANCE vs GROUND TRUTH:")
        print(f"-"*40)
        print(f"Ground Truth Anomalies: {gt_anomalies:.0f} segments ({gt_anomalies/len(ground_truth)*100:.1f}%)")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives: {tn}")
        print(f"False Negatives: {fn}")
        
        # Interpretation
        print(f"\nINTERPRETATION:")
        if accuracy > 0.8:
            print("✓ Excellent performance!")
        elif accuracy > 0.6:
            print("✓ Good performance")
        elif accuracy > 0.4:
            print("⚠ Moderate performance")
        else:
            print("✗ Poor performance - consider different threshold")
        
        return {
            'total_segments': total_segments,
            'anomaly_segments': anomaly_segments,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    return {
        'total_segments': total_segments,
        'anomaly_segments': anomaly_segments,
        'accuracy': None,
        'precision': None,
        'recall': None
    }

def test_single_video(video_name, model_path='best_video_anomaly_model_new1.pth',
                     video_dir='ShanghaiTech-campus/test', json_file='ShanghaiTech-campus/test.json',
                     threshold_method='adaptive', analyze_thresholds=True):
    """
    Complete pipeline to test a single video with MobileNet+TCN model
    """
    print(f"Testing video: {video_name}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VideoAnomalyDetector(sequence_length=16, hidden_dim=256).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None, None, None
    
    # Load video
    video_tensor, seq_length, original_length = load_and_preprocess_video(
        video_name, video_dir)
    
    if video_tensor is None:
        return None, None, None, None
    
    # Analyze multiple thresholds first
    if analyze_thresholds:
        print("\nAnalyzing different threshold methods...")
        suggested_predictions, suggested_threshold = analyze_multiple_thresholds(
            model, video_tensor, seq_length, device)
    
    # Get predictions with specified method
    scores, predictions, threshold_used = predict_video_anomalies(
        model, video_tensor, seq_length, device=device,
        threshold_method=threshold_method)
    
    # Load ground truth if available
    ground_truth = None
    try:
        with open(json_file, 'r') as f:
            labels = json.load(f)
        
        if video_name in labels:
            gt_labels = np.array(labels[video_name])
            
            # Align ground truth with model output
            if len(gt_labels) != seq_length:
                if len(gt_labels) > 1:
                    f = interp1d(np.linspace(0, 1, len(gt_labels)), 
                               gt_labels, kind='nearest')
                    gt_labels = f(np.linspace(0, 1, seq_length))
                else:
                    gt_labels = np.zeros(seq_length)
            
            ground_truth = gt_labels.astype(int)
            
    except Exception as e:
        print(f"Could not load ground truth: {e}")
    
    # Generate results
    save_path = f"{video_name.replace('.avi', '')}_results.png"
    visualize_results(video_name, scores, predictions, ground_truth,
                     threshold_used, save_path=save_path)
    
    report = generate_text_report(video_name, scores, predictions, ground_truth, threshold_used)
    
    return scores, predictions, ground_truth, report

# Main testing function
if __name__ == "__main__":
    print("="*60)
    print("VIDEO ANOMALY DETECTION TEST - MobileNet+TCN Model")
    print("="*60)
    
    # Test single video with improved thresholding
    test_video = "03_0033.avi"  # CORRECTED: Changed to match your requirement 01_0015, 01_0027, 03_0033
    
    scores, predictions, ground_truth, report = test_single_video(
        video_name=test_video,
        model_path='best_video_anomaly_model_new1.pth',  # Your model path
        video_dir='ShanghaiTech-campus/test',            # Your video directory
        json_file='ShanghaiTech-campus/test.json',       # Your labels file
        threshold_method='adaptive',  # Try: 'adaptive', 'conservative', 'median', or a number like 0.3
        analyze_thresholds=True      # Set to True to see all threshold options
    )
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
