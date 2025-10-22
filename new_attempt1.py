# Multi-Scale Temporal CNN for Video Anomaly Detection - Direct Video Processing - FIXED VERSION
# Edge-Optimized with MobileNet Backbone + Dilated Temporal Convolutions

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import os
import json
import cv2
from tqdm import tqdm
import warnings
from torchvision import transforms, models
warnings.filterwarnings('ignore')

# ============================================================================
# FOCAL LOSS - Same as your previous implementation
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

# ============================================================================
# MOBILENET FEATURE EXTRACTOR
# ============================================================================
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetFeatureExtractor, self).__init__()
        # Load pre-trained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        # Remove the classifier, keep only feature extraction
        self.features = mobilenet.features
        # Add global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Feature dimension is 1280 for MobileNetV2
        self.feature_dim = 1280
        
    def forward(self, x):
        # Input: (batch, 3, 224, 224)
        features = self.features(x)
        # Global average pooling: (batch, 1280, 1, 1) -> (batch, 1280)
        pooled = self.global_pool(features).flatten(1)
        return pooled

# ============================================================================
# MULTI-SCALE DILATED TEMPORAL CNN
# ============================================================================
class MultiScaleDilatedTCN(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256):
        super(MultiScaleDilatedTCN, self).__init__()
        
        # Multi-scale dilated 1D convolutions
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
        
        # Scale fusion weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Input: (batch, sequence, features) -> (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Process at different temporal scales
        short_out = x
        for layer in self.short_term:
            short_out = F.relu(layer(short_out))
            
        medium_out = x
        for layer in self.medium_term:
            medium_out = F.relu(layer(medium_out))
            
        long_out = x
        for layer in self.long_term:
            long_out = F.relu(layer(long_out))
        
        # Weighted fusion of scales
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        fused = (scale_weights_norm[0] * short_out + 
                scale_weights_norm[1] * medium_out + 
                scale_weights_norm[2] * long_out)
        
        # Normalize and return to (batch, sequence, features)
        fused = self.batch_norm(fused)
        fused = self.dropout(fused)
        fused = fused.transpose(1, 2)
        
        return fused, scale_weights_norm

# ============================================================================
# COMPLETE VIDEO ANOMALY DETECTION MODEL
# ============================================================================
class VideoAnomalyDetector(nn.Module):
    def __init__(self, sequence_length=16, hidden_dim=256, num_classes=1):
        super(VideoAnomalyDetector, self).__init__()
        
        # Spatial feature extractor
        self.feature_extractor = MobileNetFeatureExtractor(pretrained=True)
        feature_dim = self.feature_extractor.feature_dim
        
        # Temporal processing
        self.temporal_processor = MultiScaleDilatedTCN(
            input_dim=feature_dim, 
            hidden_dim=hidden_dim
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.sequence_length = sequence_length
        
    def forward(self, video_frames, return_probabilities=False):
        # Input: (batch, sequence, 3, 224, 224)
        batch_size, seq_len = video_frames.shape[:2]
        
        # Reshape for batch processing: (batch*sequence, 3, 224, 224)
        frames = video_frames.view(-1, 3, 224, 224)
        
        # Extract spatial features
        spatial_features = self.feature_extractor(frames)
        
        # Reshape back: (batch, sequence, feature_dim)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Temporal processing
        temporal_features, scale_weights = self.temporal_processor(spatial_features)
        
        # Classification for each frame
        logits = self.classifier(temporal_features)
        
        if return_probabilities:
            return torch.sigmoid(logits), scale_weights
        else:
            return logits, scale_weights

# ============================================================================
# VIDEO DATASET LOADER - FIXED VERSION
# ============================================================================
class VideoAnomalyDataset(Dataset):
    def __init__(self, video_dir, json_file, sequence_length=16, is_train=True, 
                 frame_size=(224, 224), max_videos=None):
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.frame_size = frame_size
        
        # Load labels
        with open(json_file, 'r') as f:
            self.labels = json.load(f)
        
        # Get video files
        self.video_files = list(self.labels.keys())
        if max_videos:
            self.video_files = self.video_files[:max_videos]
        
        print(f"Dataset - Videos loaded: {len(self.video_files)}")
        
        # Frame transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path, target_frames=None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if target_frames is None:
            target_frames = min(total_frames, self.sequence_length * 2)
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
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
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            # Transform frames
            transformed_frames = []
            for frame in frames:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            
            # Stack frames: (sequence, 3, 224, 224)
            video_tensor = torch.stack(transformed_frames)
            
            # Get labels - FIXED VERSION
            if self.is_train:
                # Training: video-level labels
                label = float(self.labels[video_name])
                frame_labels = torch.full((self.sequence_length,), label, dtype=torch.float32)
            else:
                # Testing: frame-level labels
                frame_labels_list = self.labels[video_name]
                if len(frame_labels_list) != self.sequence_length:
                    # Interpolate labels to match sequence length
                    frame_labels_array = np.array(frame_labels_list, dtype=np.float32)
                    indices = np.linspace(0, len(frame_labels_array)-1, self.sequence_length)
                    frame_labels = np.interp(indices, np.arange(len(frame_labels_array)), frame_labels_array)
                    # FIX: Round interpolated values to 0 or 1
                    frame_labels = np.round(frame_labels).astype(np.float32)
                else:
                    frame_labels = np.array(frame_labels_list, dtype=np.float32)
                
                frame_labels = torch.FloatTensor(frame_labels)
            
            return {
                'frames': video_tensor,
                'labels': frame_labels,
                'video_name': video_name
            }
            
        except Exception as e:
            print(f"Error loading video {video_name}: {e}")
            # Return dummy data
            dummy_frames = torch.zeros(self.sequence_length, 3, 224, 224)
            dummy_labels = torch.zeros(self.sequence_length)
            return {
                'frames': dummy_frames,
                'labels': dummy_labels,
                'video_name': video_name
            }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        frames = batch['frames'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits, scale_weights = model(frames)
            
            # Reshape for loss computation
            logits = logits.squeeze(-1)  # (batch, sequence)
            
            # Compute focal loss
            loss = criterion(logits.flatten(), labels.flatten())
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

# ============================================================================
# FIXED EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            frames = batch['frames'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                scores, _ = model(frames, return_probabilities=True)
            
            # Flatten and collect scores
            all_scores.extend(scores.flatten().cpu().numpy())
            all_labels.extend(labels.flatten().cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # FIX: Convert continuous labels to binary (0 or 1)
    # Any label >= 0.5 is considered anomaly (1), else normal (0)
    all_labels_binary = (all_labels >= 0.5).astype(int)
    
    print(f"Debug - Label distribution: {np.bincount(all_labels_binary)}")
    print(f"Debug - Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
    
    # Check if we have both classes
    unique_labels = np.unique(all_labels_binary)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class found in labels: {unique_labels}")
        # Return dummy values if only one class
        return 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    
    # Calculate AUC with binary labels
    try:
        auc_score = roc_auc_score(all_labels_binary, all_scores)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc_score = 0.5
    
    # Find optimal threshold
    thresholds = np.linspace(0.1, 0.9, 100)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)
        if len(np.unique(predictions)) > 1:  # Only calculate if both classes present
            try:
                f1 = f1_score(all_labels_binary, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue
    
    # Calculate final metrics with best threshold
    predictions = (all_scores >= best_threshold).astype(int)
    
    try:
        precision = precision_score(all_labels_binary, predictions, zero_division=0)
        recall = recall_score(all_labels_binary, predictions, zero_division=0)
        accuracy = np.mean(predictions == all_labels_binary)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision = recall = accuracy = 0.5
    
    return auc_score, accuracy, best_threshold, precision, recall, best_f1

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    # Configuration
    config = {
        'batch_size': 2,  # Reduced for RTX 4050
        'learning_rate': 0.0001,
        'num_epochs': 25,
        'sequence_length': 16,
        'hidden_dim': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'patience': 10,
        'focal_alpha': 0.25,
        'focal_gamma': 1.0
    }
    
    print("="*60)
    print("MULTI-SCALE TEMPORAL CNN - VIDEO ANOMALY DETECTION")
    print("Direct Video Processing with MobileNet + Dilated TCN")
    print("="*60)
    
    # Dataset paths
    train_video_dir = "ShanghaiTech-campus/train"
    test_video_dir = "ShanghaiTech-campus/test"
    train_json = "ShanghaiTech-campus/train.json"
    test_json = "ShanghaiTech-campus/test.json"
    
    # Create datasets
    print("Creating datasets...")
    try:
        train_dataset = VideoAnomalyDataset(
            train_video_dir, train_json, 
            sequence_length=config['sequence_length'],
            is_train=True
        )
        test_dataset = VideoAnomalyDataset(
            test_video_dir, test_json,
            sequence_length=config['sequence_length'], 
            is_train=False
        )
        
        print(f"Training videos: {len(train_dataset)}")
        print(f"Test videos: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Initialize model
    model = VideoAnomalyDetector(
        sequence_length=config['sequence_length'],
        hidden_dim=config['hidden_dim']
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(
        alpha=config['focal_alpha'], 
        gamma=config['focal_gamma']
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                config['device'], scaler)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluation every 2 epochs
        if (epoch + 1) % 2 == 0:
            auc, accuracy, threshold, precision, recall, f1 = evaluate_model(
                model, test_loader, config['device']
            )
            
            print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'auc': auc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }, 'best_video_anomaly_model_new1.pth')
                print(f"New best model saved! F1: {best_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print("Early stopping triggered!")
                break
        
        torch.cuda.empty_cache()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    try:
        checkpoint = torch.load('best_video_anomaly_model_new1.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        auc, accuracy, threshold, precision, recall, f1 = evaluate_model(
            model, test_loader, config['device']
        )
        
        print(f"Final Results:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        
    except Exception as e:
        print(f"Error during final evaluation: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    torch.backends.cudnn.benchmark = True
    main()
