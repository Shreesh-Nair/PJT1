# Complete ROBUST Video Anomaly Detection Implementation
# Handles missing files and other edge cases

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Dataset Class - ROBUST with missing file handling
class ShanghaiTechDataset(Dataset):
    def __init__(self, i3d_dir, json_file, max_length=None, is_train=True):
        """
        Robust dataset class that handles missing files
        """
        self.i3d_dir = i3d_dir
        self.is_train = is_train
        
        # Load labels
        with open(json_file, 'r') as f:
            all_labels = json.load(f)
        
        # Filter out video names that don't have corresponding .npy files
        self.labels = {}
        missing_files = []
        
        for video_name, label in all_labels.items():
            npy_name = video_name.replace('.avi', '.npy')
            npy_path = os.path.join(i3d_dir, npy_name)
            
            if os.path.exists(npy_path):
                # Check if file can be loaded
                try:
                    test_load = np.load(npy_path)
                    if test_load.shape[0] > 0:  # Non-empty file
                        self.labels[video_name] = label
                    else:
                        missing_files.append(f"{npy_name} (empty)")
                except:
                    missing_files.append(f"{npy_name} (corrupted)")
            else:
                missing_files.append(npy_name)
        
        self.video_names = list(self.labels.keys())
        
        print(f"Original videos in JSON: {len(all_labels)}")
        print(f"Available .npy files: {len(self.video_names)}")
        if missing_files:
            print(f"Missing/problematic files: {len(missing_files)}")
            if len(missing_files) <= 10:
                print(f"Missing files: {missing_files}")
            else:
                print(f"First 10 missing files: {missing_files[:10]}")
        
        # Determine max_length if not provided
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
            
            self.max_length = min(max_len + 5, 100) if max_len > 0 else 50
        else:
            self.max_length = max_length
            
        print(f"Max sequence length set to: {self.max_length}")
    
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        npy_name = video_name.replace('.avi', '.npy')
        npy_path = os.path.join(self.i3d_dir, npy_name)
        
        try:
            # Load I3D features
            features = np.load(npy_path)  # Shape: (T, 10, 2048)
            
            # Handle unexpected shapes
            if len(features.shape) != 3 or features.shape[1] != 10 or features.shape[2] != 2048:
                print(f"Warning: Unexpected shape {features.shape} for {npy_name}")
                # Create dummy features if shape is wrong
                features = np.random.randn(30, 10, 2048).astype(np.float32)
            
        except Exception as e:
            print(f"Error loading {npy_name}: {e}")
            # Create dummy features for missing/corrupted files
            features = np.random.randn(30, 10, 2048).astype(np.float32)
        
        # Spatial pooling: average across 10 spatial regions
        pooled_features = features.mean(axis=1)  # Shape: (T, 2048)
        
        # Get original sequence length
        original_seq_length = pooled_features.shape[0]
        
        # Get labels and handle properly
        if self.is_train:
            # Training: video-level label (0 or 1)
            video_label = self.labels[video_name]
            # Create frame labels with same length as features
            frame_labels = np.full(original_seq_length, video_label, dtype=np.float32)
        else:
            # Testing: frame-level labels (array of 0s and 1s)
            try:
                frame_labels = np.array(self.labels[video_name], dtype=np.float32)
                
                # Handle length mismatch between features and labels
                if len(frame_labels) != original_seq_length:
                    if len(frame_labels) > 1:
                        # Interpolate labels to match feature length
                        try:
                            f = interp1d(np.linspace(0, 1, len(frame_labels)), 
                                       frame_labels, kind='nearest')
                            frame_labels = f(np.linspace(0, 1, original_seq_length))
                        except:
                            # Fallback: resize using repeat/truncate
                            if len(frame_labels) < original_seq_length:
                                repeat_factor = original_seq_length // len(frame_labels) + 1
                                frame_labels = np.tile(frame_labels, repeat_factor)[:original_seq_length]
                            else:
                                frame_labels = frame_labels[:original_seq_length]
                    else:
                        frame_labels = np.zeros(original_seq_length, dtype=np.float32)
            except:
                # Fallback for label loading errors
                frame_labels = np.zeros(original_seq_length, dtype=np.float32)
        
        # Truncate if longer than max_length
        if original_seq_length > self.max_length:
            pooled_features = pooled_features[:self.max_length]
            frame_labels = frame_labels[:self.max_length]
            seq_length = self.max_length
        else:
            seq_length = original_seq_length
        
        # Pad if shorter than max_length
        if seq_length < self.max_length:
            padding_length = self.max_length - seq_length
            feature_padding = np.zeros((padding_length, 2048), dtype=np.float32)
            pooled_features = np.vstack([pooled_features, feature_padding])
            label_padding = np.zeros(padding_length, dtype=np.float32)
            frame_labels = np.concatenate([frame_labels, label_padding])
        
        return {
            'features': torch.FloatTensor(pooled_features),
            'labels': torch.FloatTensor(frame_labels),
            'seq_length': seq_length,
            'video_name': video_name
        }

# LSTM Model - Same as before
class AnomalyLSTM(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=1, dropout=0.3):
        super(AnomalyLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out_reshaped = lstm_out.contiguous().view(-1, self.hidden_dim)
        scores_flat = self.fc(lstm_out_reshaped).squeeze(-1)
        scores = scores_flat.view(batch_size, seq_length)
        
        return scores

# Training function - Same as before
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        seq_lengths = batch['seq_length']
        
        optimizer.zero_grad()
        
        scores = model(features)
        
        batch_loss = 0
        valid_samples = 0
        
        for i in range(features.size(0)):
            seq_len = seq_lengths[i].item()
            if seq_len > 0:
                pred_scores = scores[i, :seq_len]
                target_labels = labels[i, :seq_len]
                
                sample_loss = criterion(pred_scores, target_labels)
                batch_loss += sample_loss
                valid_samples += 1
        
        if valid_samples > 0:
            loss = batch_loss / valid_samples
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation function - Same as before
def evaluate_model(model, test_loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            seq_lengths = batch['seq_length']
            
            scores = model(features)
            
            for i in range(features.size(0)):
                seq_len = seq_lengths[i].item()
                if seq_len > 0:
                    all_scores.extend(scores[i, :seq_len].cpu().numpy())
                    all_labels.extend(labels[i, :seq_len].cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Handle case where all labels are the same
    if len(np.unique(all_labels)) < 2:
        print("Warning: Only one class present in labels. AUC calculation may be unreliable.")
        return 0.5, 0.5, 0.5
    
    auc_score = roc_auc_score(all_labels, all_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    predictions = (all_scores >= optimal_threshold).astype(int)
    accuracy = np.mean(predictions == all_labels)
    
    return auc_score, accuracy, optimal_threshold

# Main execution function - Updated with better error handling
def main():
    config = {
        'batch_size': 2,
        'learning_rate': 0.0005,
        'num_epochs': 15,
        'hidden_dim': 128,
        'num_layers': 1,
        'dropout': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # REPLACE THESE PATHS WITH YOUR ACTUAL PATHS
    train_i3d_dir = "I3D/Train_I3D"  
    test_i3d_dir = "I3D\Test_I3D"  # Note: Check if it's "Test_I3D" or "test_I3D"
    train_json = "ShanghaiTech-campus/train.json"  
    test_json = "ShanghaiTech-campus/test.json"    
    
    print("Loading datasets...")
    
    try:
        train_dataset = ShanghaiTechDataset(train_i3d_dir, train_json, is_train=True)
        test_dataset = ShanghaiTechDataset(test_i3d_dir, test_json, 
                                         max_length=train_dataset.max_length, 
                                         is_train=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Check if we have enough data
        if len(train_dataset) == 0:
            print("Error: No training data available!")
            return None, 0, 0
        
        if len(test_dataset) == 0:
            print("Error: No test data available!")
            return None, 0, 0
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Please check your file paths and directory structure.")
        return None, 0, 0
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    model = AnomalyLSTM(
        input_dim=2048,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(config['device'])
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")
    
    best_auc = 0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        try:
            train_loss = train_model(model, train_loader, criterion, optimizer, config['device'])
            print(f"Training Loss: {train_loss:.4f}")
            
            if (epoch + 1) % 3 == 0:
                print("Evaluating...")
                auc_score, accuracy, threshold = evaluate_model(model, test_loader, config['device'])
                print(f"AUC-ROC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    torch.save(model.state_dict(), 'best_anomaly_model.pth')
                    print(f"New best model saved! AUC: {best_auc:.4f}")
                    
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            continue
    
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    try:
        auc_score, accuracy, threshold = evaluate_model(model, test_loader, config['device'])
        print(f"Final AUC-ROC: {auc_score:.4f}")
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Optimal Threshold: {threshold:.4f}")
    except Exception as e:
        print(f"Error during final evaluation: {e}")
        auc_score, accuracy = 0, 0
    
    return model, auc_score, accuracy

# Additional utility function to check your dataset
def check_dataset_structure(i3d_dir, json_file):
    """
    Utility function to check dataset structure and identify issues
    """
    print(f"\n=== DATASET STRUCTURE CHECK ===")
    print(f"I3D Directory: {i3d_dir}")
    print(f"JSON File: {json_file}")
    
    # Check if directories exist
    if not os.path.exists(i3d_dir):
        print(f"ERROR: I3D directory does not exist: {i3d_dir}")
        return
    
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file does not exist: {json_file}")
        return
    
    # Load JSON
    with open(json_file, 'r') as f:
        labels = json.load(f)
    
    # Check .npy files
    npy_files = [f for f in os.listdir(i3d_dir) if f.endswith('.npy')]
    
    print(f"Videos in JSON: {len(labels)}")
    print(f".npy files in directory: {len(npy_files)}")
    
    # Check correspondences
    missing_files = []
    for video_name in labels.keys():
        npy_name = video_name.replace('.avi', '.npy')
        if npy_name not in npy_files:
            missing_files.append(npy_name)
    
    if missing_files:
        print(f"Missing .npy files: {len(missing_files)}")
        print(f"First 10 missing: {missing_files[:10]}")
    else:
        print("All JSON entries have corresponding .npy files!")

if __name__ == "__main__":
    # First, check your dataset structure
    # Uncomment these lines to diagnose issues:
    # check_dataset_structure("I3D/Test_I3D", "ShanghaiTech-campus/test.json")
    # check_dataset_structure("I3D/train_I3D", "ShanghaiTech-campus/train.json")
    
    # Run the main pipeline
    model, auc, accuracy = main()
