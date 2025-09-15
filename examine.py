import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def examine_i3d_files(directory_path):
    """
    Comprehensive examination of I3D .npy files in a directory
    """
    print("=" * 60)
    print("I3D FILES EXAMINATION")
    print("=" * 60)
    
    # Get all .npy files
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    
    print(f"Directory: {directory_path}")
    print(f"Total .npy files found: {len(npy_files)}")
    print(f"First 10 files: {npy_files[:10]}")
    print("-" * 60)
    
    # Statistics collection
    shapes = []
    dtypes = []
    file_sizes = []
    feature_stats = defaultdict(list)
    
    # Examine first few files in detail
    print("DETAILED EXAMINATION OF SAMPLE FILES:")
    print("-" * 60)
    
    sample_files = npy_files[:5]  # Examine first 5 files
    
    for i, filename in enumerate(sample_files):
        filepath = os.path.join(directory_path, filename)
        
        try:
            # Load the file
            data = np.load(filepath)
            
            print(f"\nFile {i+1}: {filename}")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  File size: {os.path.getsize(filepath)} bytes")
            print(f"  Min value: {data.min():.6f}")
            print(f"  Max value: {data.max():.6f}")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  Std: {data.std():.6f}")
            print(f"  Contains NaN: {np.isnan(data).any()}")
            print(f"  Contains Inf: {np.isinf(data).any()}")
            
            # Store statistics
            shapes.append(data.shape)
            dtypes.append(str(data.dtype))
            file_sizes.append(os.path.getsize(filepath))
            
            # Store feature statistics
            feature_stats['min'].append(data.min())
            feature_stats['max'].append(data.max())
            feature_stats['mean'].append(data.mean())
            feature_stats['std'].append(data.std())
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS:")
    print("=" * 60)
    
    # Shape analysis
    unique_shapes = list(set(shapes))
    print(f"Unique shapes found: {unique_shapes}")
    
    for shape in unique_shapes:
        count = shapes.count(shape)
        print(f"  Shape {shape}: {count} files")
    
    # Data type analysis
    unique_dtypes = list(set(dtypes))
    print(f"\nData types: {unique_dtypes}")
    
    # File size analysis
    if file_sizes:
        print(f"\nFile sizes:")
        print(f"  Min: {min(file_sizes)} bytes")
        print(f"  Max: {max(file_sizes)} bytes")
        print(f"  Average: {np.mean(file_sizes):.0f} bytes")
    
    # Feature value analysis
    if feature_stats['min']:
        print(f"\nFeature value ranges across sampled files:")
        print(f"  Min values: {min(feature_stats['min']):.6f} to {max(feature_stats['min']):.6f}")
        print(f"  Max values: {min(feature_stats['max']):.6f} to {max(feature_stats['max']):.6f}")
        print(f"  Mean values: {min(feature_stats['mean']):.6f} to {max(feature_stats['mean']):.6f}")
        print(f"  Std values: {min(feature_stats['std']):.6f} to {max(feature_stats['std']):.6f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    # Provide recommendations based on findings
    if len(unique_shapes) == 1:
        print("✓ All files have consistent shapes - Good for batch processing")
    else:
        print("⚠ Files have different shapes - Need to handle variable lengths")
    
    if len(unique_dtypes) == 1:
        print("✓ All files have consistent data types")
    else:
        print("⚠ Mixed data types found - May need type conversion")
    
    # Check if this looks like typical I3D features
    if shapes and len(shapes[0]) == 2 and shapes[0][1] in [1024, 2048]:
        print("✓ Looks like standard I3D features (1024 or 2048 dimensions)")
    
    return {
        'total_files': len(npy_files),
        'sample_shapes': shapes,
        'sample_dtypes': dtypes,
        'file_sizes': file_sizes,
        'feature_stats': dict(feature_stats)
    }

# Usage:
# Replace with your actual I3D directory path
dir_path = "I3D\Test_I3D"  # or test_I3D
results = examine_i3d_files(dir_path)
