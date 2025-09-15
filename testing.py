# UPDATED Video Anomaly Detection - Single Video Testing
# Fixed threshold issues and improved result interpretation

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import json
import os
from scipy.interpolate import interp1d

# Load your trained model (same architecture as before)
class AnomalyLSTM(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128, num_layers=1, dropout=0.2):
        super(AnomalyLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.hidden_dim)
        scores_flat = self.fc(lstm_out_reshaped).squeeze(-1)
        scores = scores_flat.view(x.size(0), x.size(1))
        return scores

def load_and_preprocess_video_features(video_name, i3d_dir, max_length=87):
    """
    Load and preprocess I3D features for a single video
    """
    npy_name = video_name.replace('.avi', '.npy')
    npy_path = os.path.join(i3d_dir, npy_name)
    
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found!")
        return None, 0, 0
    
    # Load I3D features
    features = np.load(npy_path)  # Shape: (T, 10, 2048)
    print(f"Original feature shape: {features.shape}")
    
    # Spatial pooling (same as training)
    pooled_features = features.mean(axis=1)  # Shape: (T, 2048)
    original_length = pooled_features.shape[0]
    
    # Pad/truncate to match training (same as training preprocessing)
    if original_length > max_length:
        pooled_features = pooled_features[:max_length]
        seq_length = max_length
    else:
        seq_length = original_length
        if seq_length < max_length:
            padding = np.zeros((max_length - seq_length, 2048), dtype=np.float32)
            pooled_features = np.vstack([pooled_features, padding])
    
    # Convert to tensor and add batch dimension
    features_tensor = torch.FloatTensor(pooled_features).unsqueeze(0)  # (1, max_length, 2048)
    
    return features_tensor, seq_length, original_length

def predict_video_anomalies(model, features_tensor, seq_length, device='cuda', 
                          threshold_method='adaptive'):
    """
    UPDATED: Get anomaly predictions with better threshold handling
    """
    model.eval()
    
    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        
        # Get predictions
        scores = model(features_tensor)  # (1, max_length)
        scores = scores.squeeze(0)[:seq_length].cpu().numpy()  # Only actual sequence length
        
        # Different thresholding methods
        if threshold_method == 'original':
            threshold = 0.0294  # Original training threshold
        elif threshold_method == 'median':
            threshold = np.median(scores)
        elif threshold_method == 'adaptive':
            # Use mean + 0.5 * std as threshold (works well for individual videos)
            threshold = np.mean(scores) + 0.5 * np.std(scores)
        elif threshold_method == 'conservative':
            threshold = 0.5  # Conservative threshold
        else:
            threshold = float(threshold_method)  # Use as direct threshold value
        
        # Apply threshold for binary predictions
        predictions = (scores >= threshold).astype(int)
        
        print(f"Threshold method: {threshold_method}")
        print(f"Applied threshold: {threshold:.4f}")
        print(f"Score range: {scores.min():.4f} to {scores.max():.4f}")
        print(f"Predicted anomaly percentage: {np.mean(predictions)*100:.1f}%")
        
    return scores, predictions, threshold

def analyze_multiple_thresholds(model, features_tensor, seq_length, device='cuda'):
    """
    NEW: Test multiple threshold methods to find the best one
    """
    model.eval()
    
    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        scores = model(features_tensor).squeeze(0)[:seq_length].cpu().numpy()
    
    threshold_methods = {
        'Original (0.0294)': 0.0294,
        'Conservative (0.5)': 0.5,
        'Mean': np.mean(scores),
        'Median': np.median(scores),
        'Mean + 0.5*STD': np.mean(scores) + 0.5 * np.std(scores),
        'Mean + STD': np.mean(scores) + np.std(scores),
        '75th Percentile': np.percentile(scores, 75)
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
        print(f"\nUSING CONSERVATIVE: All methods give extreme results, using 0.5")
        return results['Conservative (0.5)']['predictions'], 0.5

def visualize_results(video_name, scores, predictions, ground_truth=None, 
                     threshold_used=0.5, save_path=None):
    """
    UPDATED: Create comprehensive visualization with threshold info
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
    UPDATED: Generate a clear text summary with better metrics
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
    print(f"  Average Score: {np.mean(scores):.4f}")
    print(f"  Maximum Score: {np.max(scores):.4f}")
    print(f"  Minimum Score: {np.min(scores):.4f}")
    print(f"  Standard Deviation: {np.std(scores):.4f}")
    print(f"  Median Score: {np.median(scores):.4f}")
    
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
                print(f"  Segments {start_segment:2d}-{i-1:2d} (Avg: {avg_score:.3f}, Max: {max_score:.3f})")
                in_anomaly = False
        
        # Handle case where anomaly continues to end
        if in_anomaly:
            avg_score = np.mean(scores[start_segment:])
            max_score = np.max(scores[start_segment:])
            print(f"  Segments {start_segment:2d}-{len(predictions)-1:2d} (Avg: {avg_score:.3f}, Max: {max_score:.3f})")
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
        print(f"\nINTERPRETation:")
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
        'accuracy': accuracy if ground_truth is not None else None,
        'precision': precision if ground_truth is not None else None,
        'recall': recall if ground_truth is not None else None
    }

def test_single_video(video_name, model_path='best_anomaly_model.pth', 
                     i3d_dir='I3D/Test_I3D', json_file='ShanghaiTech-campus/test.json',
                     threshold_method='adaptive', analyze_thresholds=True):
    """
    UPDATED: Complete pipeline to test a single video with better threshold handling
    """
    print(f"Testing video: {video_name}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AnomalyLSTM().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None, None
    
    # Load video features
    features_tensor, seq_length, original_length = load_and_preprocess_video_features(
        video_name, i3d_dir)
    
    if features_tensor is None:
        return None, None, None
    
    print(f"Processing {seq_length} segments (original: {original_length})")
    
    # Analyze multiple thresholds first
    if analyze_thresholds:
        print("\nAnalyzing different threshold methods...")
        suggested_predictions, suggested_threshold = analyze_multiple_thresholds(
            model, features_tensor, seq_length, device)
    
    # Get predictions with specified method
    scores, predictions, threshold_used = predict_video_anomalies(
        model, features_tensor, seq_length, device=device, 
        threshold_method=threshold_method)
    
    # Load ground truth if available
    ground_truth = None
    try:
        with open(json_file, 'r') as f:
            labels = json.load(f)
        
        if video_name in labels:
            gt_labels = np.array(labels[video_name])
            
            # Align ground truth with features (same as in training)
            if len(gt_labels) != original_length:
                if len(gt_labels) > 1:
                    f = interp1d(np.linspace(0, 1, len(gt_labels)), 
                               gt_labels, kind='nearest')
                    gt_labels = f(np.linspace(0, 1, original_length))
                else:
                    gt_labels = np.zeros(original_length)
            
            ground_truth = gt_labels[:seq_length]  # Match sequence length
            
    except Exception as e:
        print(f"Could not load ground truth: {e}")
    
    # Generate results
    save_path = f"{video_name.replace('.avi', '')}_results_fixed.png"
    visualize_results(video_name, scores, predictions, ground_truth, 
                     threshold_used, save_path=save_path)
    
    report = generate_text_report(video_name, scores, predictions, ground_truth, threshold_used)
    
    return scores, predictions, ground_truth, report

# UPDATED: Main testing function with multiple video support
def test_multiple_videos(video_list, model_path='best_anomaly_model.pth', 
                        i3d_dir='I3D/Test_I3D', json_file='ShanghaiTech-campus/test.json'):
    """
    NEW: Test multiple videos and summarize results
    """
    print("="*60)
    print("MULTIPLE VIDEO TESTING")
    print("="*60)
    
    results_summary = []
    
    for video_name in video_list:
        print(f"\n{'='*60}")
        print(f"Testing: {video_name}")
        print('='*60)
        
        scores, predictions, ground_truth, report = test_single_video(
            video_name, model_path, i3d_dir, json_file, 
            threshold_method='adaptive', analyze_thresholds=False)
        
        if report is not None:
            results_summary.append({
                'video': video_name,
                'accuracy': report.get('accuracy', 'N/A'),
                'anomaly_pct': report['anomaly_segments'] / report['total_segments'] * 100
            })
    
    # Summary report
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print('='*60)
    
    for result in results_summary:
        acc_str = f"{result['accuracy']:.3f}" if result['accuracy'] != 'N/A' else 'N/A'
        print(f"{result['video']:15s}: Accuracy={acc_str:>6s}, Anomaly={result['anomaly_pct']:5.1f}%")

# Example usage - UPDATED
if __name__ == "__main__":
    print("="*60)
    print("UPDATED VIDEO ANOMALY DETECTION TEST")
    print("="*60)
    
    # Test single video with improved thresholding
    test_video = "01_0025.avi"  # Change this to any video you want to test
    
    scores, predictions, ground_truth, report = test_single_video(
        video_name=test_video,
        model_path='best_anomaly_model.pth',
        i3d_dir='I3D/Test_I3D',
        json_file='ShanghaiTech-campus/test.json',
        threshold_method='adaptive',  # Try: 'adaptive', 'conservative', 'median', or a number like 0.3
        analyze_thresholds=True  # Set to True to see all threshold options
    )
    
    # Optionally test multiple videos
    # video_list = ["01_001.avi", "01_002.avi", "01_003.avi"]
    # test_multiple_videos(video_list)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
