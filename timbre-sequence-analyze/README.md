# Timbre Sequence Analysis

A specialized tool for timbre sequence analysis using fixed 0.5s segmentation and multi-feature fusion for machine learning and detailed audio analysis.

## Features

### 🎯 Core Functionality
- **Fixed Duration Segmentation**: 0.5-second fixed segments for consistent analysis
- **Multi-feature Extraction**: MFCC, spectral, rhythm, and energy features
- **Curve Preprocessing**: Smoothing and denoising using Savitzky-Golay filtering
- **Time Normalization**: Interpolation-based time alignment
- **Feature Fusion**: Weighted combination or PCA-based dimensionality reduction

### 🔧 Technical Features
- **Audio Preprocessing**: Resampling, pre-emphasis, loudness normalization
- **Robust Processing**: Handles various audio formats and lengths
- **Flexible Configuration**: Customizable parameters for different use cases
- **Multiple Output Formats**: JSON, CSV, and NumPy formats
- **GUI Support**: Cross-platform file selection dialogs

## Installation

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Install from Source
```bash
# Clone or download the project
cd timbre-sequence-analyze

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Dependencies
- numpy>=1.23
- scipy>=1.10
- librosa>=0.10.1
- soundfile>=0.12
- scikit-learn>=1.3
- pandas>=2.0
- matplotlib>=3.7
- seaborn>=0.12
- tqdm>=4.66

## Usage

### Basic Usage

#### GUI Mode (Recommended)
```bash
timbre-sequence-analyze --gui
```

#### Command Line
```bash
# Analyze specific files
timbre-sequence-analyze --files audio1.wav audio2.wav --out_dir ./output

# Analyze all files in a directory
timbre-sequence-analyze --in_dir ./audio_folder --out_dir ./output

# With glob patterns
timbre-sequence-analyze --in_dir ./audio_folder --include_glob "*.wav" "*.flac" --out_dir ./output
```

### Advanced Configuration

#### Custom Segment Duration
```bash
timbre-sequence-analyze --files audio.wav --out_dir ./output --segment_duration 0.3
```

#### Adjust Feature Points
```bash
timbre-sequence-analyze --files audio.wav --out_dir ./output --target_points 150
```

#### Use PCA Fusion
```bash
timbre-sequence-analyze --files audio.wav --out_dir ./output --fusion_method pca
```

#### Custom Smoothing
```bash
timbre-sequence-analyze --files audio.wav --out_dir ./output --curve_smooth 7
```

### Complete Example
```bash
timbre-sequence-analyze \
    --files audio1.wav audio2.wav \
    --out_dir ./analysis_results \
    --segment_duration 0.5 \
    --target_points 100 \
    --fusion_method weighted \
    --curve_smooth 5
```

## Output Structure

```
output/
├── fine_analysis_summary.json  # Analysis summary and configuration
├── fine_features.csv          # Fused feature data (all tracks)
├── fine_segments_info.csv     # Segment boundary information
├── fused_features.npy         # Global fused feature matrix
└── [track_name]/              # Individual track results
    ├── fused_features.npy     # Track-specific fused features
    └── segment_times.npy      # Segment timing information
```

### Output Files Description

#### 1. fine_analysis_summary.json
Contains analysis configuration and summary statistics:
```json
{
  "config": {
    "segment_duration": 0.5,
    "curve_smooth_window": 5,
    "target_points": 100,
    "fusion_method": "weighted"
  },
  "tracks": {
    "track1": {
      "num_segments": 120,
      "total_duration": 60.0,
      "feature_dimension": 1900
    }
  },
  "global": {
    "total_tracks": 2,
    "total_segments": 240,
    "fused_feature_dimension": 1900
  }
}
```

#### 2. fine_features.csv
Tabular data with fused features for each segment:
- `track_name`: Track identifier
- `segment_index`: Segment number within track
- `segment_time`: Start time of segment (seconds)
- `feature_0` to `feature_N`: Fused feature vector components

#### 3. fine_segments_info.csv
Segment boundary information:
- `track_name`: Track identifier
- `segment_index`: Segment number
- `start_frame`: Starting frame number
- `end_frame`: Ending frame number
- `start_time`: Start time (seconds)
- `duration`: Segment duration (seconds)

## Use Cases

### Machine Learning
```python
import numpy as np
import pandas as pd

# Load fused features
features = np.load('output/fused_features.npy')
print(f"Feature matrix shape: {features.shape}")

# Load as DataFrame for ML
df = pd.read_csv('output/fine_features.csv')
X = df.filter(regex='feature_').values
y = df['track_name'].values  # or your labels
```

### Audio Analysis
```python
import json

# Load analysis summary
with open('output/fine_analysis_summary.json', 'r') as f:
    summary = json.load(f)

print(f"Total segments: {summary['global']['total_segments']}")
print(f"Feature dimension: {summary['global']['fused_feature_dimension']}")
```

### Research Applications
- **Music Information Retrieval**: Feature extraction for MIR tasks
- **Audio Classification**: Preparing features for classification models
- **Temporal Analysis**: Studying audio characteristics over time
- **Comparative Studies**: Analyzing differences between audio tracks

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--segment_duration` | 0.5 | Fixed segment duration in seconds |
| `--target_points` | 100 | Target number of feature points per segment |
| `--curve_smooth` | 5 | Savitzky-Golay smoothing window size |
| `--fusion_method` | weighted | Feature fusion method (weighted/pca) |
| `--time_normalize` | interpolate | Time normalization method |
| `--sr` | 22050 | Target sample rate |
| `--pre_emphasis` | 0.97 | Pre-emphasis filter coefficient |

## Feature Types

The tool extracts and fuses multiple audio features:

1. **MFCC Features** (30% weight): Mel-frequency cepstral coefficients
2. **Spectral Centroid** (15% weight): Brightness measure
3. **Spectral Rolloff** (15% weight): Frequency rolloff point
4. **Spectral Bandwidth** (15% weight): Frequency spread
5. **Tempo** (10% weight): Beat tracking tempo
6. **Zero Crossing Rate** (10% weight): Noise/roughness measure
7. **RMS Energy** (5% weight): Loudness measure

## Performance Notes

- **Processing Time**: Approximately 1-2 seconds per minute of audio
- **Memory Usage**: ~100MB per hour of audio (depending on parameters)
- **Output Size**: ~1-5MB per minute of audio (depending on target_points)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Audio Format Issues**
   ```bash
   # Check if librosa can read your files
   python -c "import librosa; print(librosa.get_duration('your_file.wav'))"
   ```

3. **Memory Issues**
   ```bash
   # Use smaller target_points for large files
   timbre-sequence-analyze --files large_file.wav --target_points 50
   ```

### Getting Help

- Check the output logs for detailed error messages
- Ensure audio files are not corrupted
- Verify sufficient disk space for output files
- Use `--gui` mode for easier file selection

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Changelog

### v0.1.0
- Initial release
- Fixed 0.5s segmentation
- Multi-feature extraction and fusion
- GUI and CLI support
- Multiple output formats

