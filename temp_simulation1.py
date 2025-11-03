# realtime_edge_simulation.py - FULLY FIXED
# Real-Time Anomaly Detection for Edge Devices

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
from collections import deque
from torchvision import transforms, models
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "best_enhanced_model_new7.pth"
VIDEO_NAME = "01_0015.avi"
VIDEO_DIR = "ShanghaiTech-campus/test"

DETECTION_THRESHOLD = 0.5
BUFFER_SIZE = 16
DISPLAY_FPS = 30
ALERT_PERSISTENCE = 5

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

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
# REAL-TIME DETECTOR
# ============================================================================

class RealTimeAnomalyDetector:
    def __init__(self, model_path, device='cuda', buffer_size=16, threshold=0.5):
        self.device = device
        self.buffer_size = buffer_size
        self.threshold = threshold
        
        print("\n" + "="*70)
        print("REAL-TIME EDGE ANOMALY DETECTION SYSTEM")
        print("="*70)
        
        print(f"\nLoading model from: {model_path}")
        self.model = VideoAnomalyDetector(
            sequence_length=buffer_size,
            hidden_dim=256
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  AUC: {checkpoint.get('auc', 0):.4f}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_count = 0
        self.total_anomalies = 0
        self.detection_times = []
        self.fps_history = deque(maxlen=30)
        self.alert_counter = 0
        
    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb)
    
    def detect_anomaly(self):
        if len(self.frame_buffer) < self.buffer_size:
            return 0.0, False
        
        frames_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            scores, _ = self.model(frames_tensor, return_probabilities=True)
        
        inference_time = (time.time() - start_time) * 1000
        self.detection_times.append(inference_time)
        
        current_score = scores[0, -1, 0].cpu().item()
        is_anomaly = current_score >= self.threshold
        
        if is_anomaly:
            self.total_anomalies += 1
            self.alert_counter = ALERT_PERSISTENCE
        
        return current_score, is_anomaly
    
    def draw_overlay(self, frame, score, is_anomaly, fps, inference_time):
        """Draw status overlay on frame - FIXED font constants"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Top status bar
        cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Title - FIXED: use FONT_HERSHEY_SIMPLEX with bold effect via thickness
        cv2.putText(frame, "EDGE ANOMALY DETECTION", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        
        # Status indicator
        status_color = (0, 255, 0) if not is_anomaly else (0, 0, 255)
        status_text = "NORMAL" if not is_anomaly else "ALERT"
        cv2.circle(frame, (w - 100, 30), 15, status_color, -1)
        cv2.putText(frame, status_text, (w - 200, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Performance metrics
        metrics = f"Frame: {self.frame_count}  |  FPS: {fps:.1f}  |  Latency: {inference_time:.1f}ms"
        cv2.putText(frame, metrics, (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Anomaly score bar
        bar_y, bar_h = 78, 16
        bar_w = w - 30
        
        # Background
        cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Score fill
        fill_w = int(bar_w * min(score, 1.0))
        if score < self.threshold:
            bar_color = (0, 255, 0)
        elif score < 0.7:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 0, 255)
        cv2.rectangle(frame, (15, bar_y), (15 + fill_w, bar_y + bar_h), bar_color, -1)
        
        # Threshold line
        threshold_x = int(15 + bar_w * self.threshold)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_h), (255, 255, 255), 2)
        
        # Score text
        cv2.putText(frame, f"Score: {score:.3f}", (w - 150, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Anomaly alert overlay
        if self.alert_counter > 0:
            # Flashing red border
            thickness = 10 if self.alert_counter % 2 == 0 else 6
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thickness)
            
            # Alert box
            alert_y = h - 120
            alert_h = 90
            
            cv2.rectangle(overlay, (10, alert_y), (w - 10, alert_y + alert_h), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Alert text
            alert_text = "ANOMALY DETECTED"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = alert_y + 55
            
            # Warning symbols
            cv2.putText(frame, "!", (text_x - 60, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
            cv2.putText(frame, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
            cv2.putText(frame, "!", (text_x + text_size[0] + 20, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
            
            self.alert_counter -= 1
        
        # Statistics panel (bottom right)
        stat_x = w - 260
        stat_y = h - 110
        
        cv2.rectangle(overlay, (stat_x, stat_y), (w - 10, h - 10), (30, 30, 30), -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        
        stats_lines = [
            f"Total Frames: {self.frame_count}",
            f"Anomalies: {self.total_anomalies}",
            f"Rate: {(self.total_anomalies/max(self.frame_count,1)*100):.1f}%",
            f"Avg Latency: {np.mean(self.detection_times[-30:]):.1f}ms" if self.detection_times else "Avg Latency: N/A"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(frame, line, (stat_x + 10, stat_y + 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        return frame
    
    def process_video(self, video_path, display_fps=30):
        """Process video with real-time detection"""
        print(f"\n{'='*70}")
        print(f"Video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {total_frames}")
        print(f"  FPS: {video_fps:.1f}")
        print(f"\nControls: q=quit, p=pause, s=screenshot, +/- threshold")
        print(f"{'='*70}\n")
        
        frame_delay = int(1000 / display_fps)
        paused = False
        last_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Preprocess and buffer
                frame_tensor = self.preprocess_frame(frame)
                self.frame_buffer.append(frame_tensor)
                
                # Detect
                score, is_anomaly = self.detect_anomaly()
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / max(current_time - last_time, 0.001)
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history)
                last_time = current_time
                
                avg_inference = np.mean(self.detection_times[-30:]) if self.detection_times else 0
                
                # Draw overlays
                display_frame = self.draw_overlay(frame, score, is_anomaly, avg_fps, avg_inference)
                
                # Resize if too large
                if width > 1280:
                    scale = 1280 / width
                    display_frame = cv2.resize(display_frame, (1280, int(height * scale)))
                
                cv2.imshow('Real-Time Edge Detection', display_frame)
                
                # Progress
                if self.frame_count % 50 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Anomalies: {self.total_anomalies}")
            
            # Handle keys
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸ PAUSED" if paused else "▶ RESUMED")
            elif key == ord('s'):
                filename = f"frame_{self.frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Saved: {filename}")
            elif key == ord('+') or key == ord('='):
                self.threshold = min(self.threshold + 0.05, 0.95)
                print(f"Threshold: {self.threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.threshold = max(self.threshold - 0.05, 0.05)
                print(f"Threshold: {self.threshold:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        print(f"\n{'='*70}")
        print("DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Anomalies Detected: {self.total_anomalies} ({self.total_anomalies/max(self.frame_count,1)*100:.1f}%)")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}")
        print(f"Average Latency: {np.mean(self.detection_times):.2f}ms")
        print(f"Final Threshold: {self.threshold:.3f}")
        print(f"{'='*70}\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("REAL-TIME EDGE DEVICE ANOMALY DETECTION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nChecking files...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Video Dir: {VIDEO_DIR}")
    print(f"  Video: {VIDEO_NAME}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found!")
        print(f"   Path: {os.path.abspath(MODEL_PATH)}")
        sys.exit(1)
    
    video_path = os.path.join(VIDEO_DIR, VIDEO_NAME)
    if not os.path.exists(video_path):
        print(f"\n❌ ERROR: Video not found!")
        print(f"   Path: {os.path.abspath(video_path)}")
        sys.exit(1)
    
    print(f"\n✓ All files found!")
    
    try:
        detector = RealTimeAnomalyDetector(
            model_path=MODEL_PATH,
            device=device,
            buffer_size=BUFFER_SIZE,
            threshold=DETECTION_THRESHOLD
        )
        
        detector.process_video(video_path, display_fps=DISPLAY_FPS)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
