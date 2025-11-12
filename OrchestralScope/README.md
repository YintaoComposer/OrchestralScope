# OrchestralScope

A comprehensive suite of tools for audio timbre analysis, including timbre structure analysis, complexity analysis, and timbre sequence analysis.

## Overview

This repository contains two independent analysis tools:

1. **Timbre Structure Analysis** (`timbre-analyze`) - Intelligent timbre segmentation, clustering, and complexity analysis
2. **Timbre Sequence Analysis** (`timbre-sequence-analyze`) - Fine-grained sequence analysis with fixed-duration segmentation

## Quick Start

```

### Paper Quick Commands

For **Timbre Structure Analysis and Complexity Analysis** (as used in the paper):

```bash
python -m timbre_analyze.cli \
  --gui \
  --k_policy per_track \
  --k_select feature_weighted \
  --k_min 4 --k_max 12
```

For **Timbre Sequence Analysis** (as used in the paper):

```bash
timbre-sequence-analyze --gui --use_dtw
```

> **Note**: Changing filenames may cause command-line errors. Use the commands above as specified.

## Project 1: Timbre Structure Analysis (`timbre-analyze`)

### Features

- **Intelligent Segmentation**: Novelty curve-based adaptive segmentation
- **Clustering**: GMM clustering with automatic K selection
- **Complexity Analysis**: Local and global complexity metrics
- **Visualization**: 2D/3D embedding plots, complexity timelines
- **Distance Metrics**: JSD, Levenshtein, Markov, and combined distances

### Basic Usage

```bash
# GUI mode (recommended)
timbre-analyze --gui

# Command line
timbre-analyze --files audio1.wav audio2.wav --out_dir ./output

# With custom parameters
timbre-analyze \
  --files audio.wav \
  --out_dir ./output \
  --k_policy per_track \
  --k_select feature_weighted \
  --k_min 4 --k_max 12
```

### Key Parameters

- `--k_policy`: K selection policy (`auto`, `per_track`, `fixed`)
- `--k_select`: K selection method (`bic`, `aic`, `silhouette`, `feature_weighted`, `conservative`)
- `--k_min`, `--k_max`: K value range
- `--embed_dim`: Embedding dimension (2 or 3)

### Output Files

```
output/
├── segments/           # Segment boundaries
├── sequences/          # Label sequences
├── stats/              # Distribution and Markov matrices
├── distance/           # Distance matrices
├── embedding/          # Dimensionality reduction results
├── complexity/         # Complexity analysis
└── figs/              # Visualization plots
```

## Project 2: Timbre Sequence Analysis (`timbre-sequence-analyze`)

### Features

- **Fixed Segmentation**: 0.5-second fixed segments for consistent analysis
- **Multi-feature Extraction**: MFCC, spectral, rhythm, and energy features
- **Feature Fusion**: Weighted combination or PCA-based fusion
- **DTW Alignment**: Dynamic Time Warping for sequence comparison
- **Visualization**: Feature trends, similarity matrices, distributions

### Basic Usage

```bash
# GUI mode (recommended)
timbre-sequence-analyze --gui

# With DTW alignment (as in paper)
timbre-sequence-analyze --gui --use_dtw

# Command line
timbre-sequence-analyze --files audio1.wav audio2.wav --out_dir ./output
```

### Key Parameters

- `--segment_duration`: Fixed segment duration (default: 0.5 seconds)
- `--target_points`: Target number of feature points per segment
- `--fusion_method`: Feature fusion method (`weighted` or `pca`)
- `--use_dtw`: Enable DTW alignment for similarity calculation
- `--no_viz`: Skip visualization generation

### Output Files

```
output/
├── fine_analysis_summary.json  # Analysis summary
├── fine_features.csv          # Fused feature data
├── fine_segments_info.csv     # Segment information
├── fused_features.npy         # Global fused features
├── [track_name]/              # Per-track results
└── visualizations/            # Visualization plots
    ├── feature_trends.png
    ├── similarity_matrix.png
    ├── feature_distributions.png
    └── segment_analysis.png
```

## Feature Comparison

| Feature | Timbre Structure Analysis | Timbre Sequence Analysis |
|---------|--------------------------|-----------------------------------|
| **Segmentation** | Novelty curve-based adaptive | Fixed 0.5-second segments |
| **Clustering** | GMM with automatic K selection | Feature fusion, no clustering |
| **Complexity** | ✅ Local and global metrics | ❌ Not available |
| **Visualization** | 2D/3D embeddings, complexity plots | Feature trends, similarity matrices |
| **Use Case** | Music structure, timbre clustering | Sequence analysis, ML feature extraction |

## Installation Details

### Dependencies

**Main Project (`timbre-analyze`)**:
- numpy>=1.23
- scipy>=1.10
- librosa>=0.10.1
- soundfile>=0.12
- scikit-learn>=1.3
- hmmlearn>=0.3.0
- umap-learn>=0.5.5
- pandas>=2.0
- matplotlib>=3.7
- seaborn>=0.12
- tqdm>=4.66
- numba>=0.58

**Sequence Analysis (`timbre-sequence-analyze`)**:
- numpy>=1.23
- scipy>=1.10
- librosa>=0.10.1
- soundfile>=0.12
- scikit-learn>=1.3
- pandas>=2.0
- matplotlib>=3.7
- seaborn>=0.12
- tqdm>=4.66

## Directory Structure

```
mel_副本3/
├── timbre_analyze/           # Main analysis package
│   ├── cli.py                # Command-line interface
│   ├── audio.py              # Audio preprocessing
│   ├── features.py           # Feature extraction
│   ├── segmentation.py       # Segmentation algorithms
│   ├── clustering.py         # Clustering algorithms
│   ├── complexity.py         # Complexity analysis
│   ├── metrics.py            # Distance metrics
│   ├── embedding.py          # Dimensionality reduction
│   ├── viz.py                # Visualization
│   └── ...
├── timbre-sequence-analyze/      # Sequence analysis package
│   ├── timbre_sequence_analyze/
│   │   ├── cli.py
│   │   ├── analysis.py
│   │   ├── viz.py
│   │   └── ...
│   └── README.md
├── README.md                 # This file
├── PROJECT_STATUS.md         # Project status summary
├── USAGE_GUIDE.md            # Detailed usage guide
├── requirements.txt          # Main project dependencies
└── pyproject.toml            # Main project configuration
```

## Advanced Examples

### Timbre Structure Analysis with Custom Settings

```bash
timbre-analyze \
  --files audio.wav \
  --out_dir ./output \
  --k_policy per_track \
  --k_select feature_weighted \
  --k_min 4 --k_max 12 \
  --embed_dim 3 \
  --clustering_sensitivity normal
```

### Timbre Sequence Analysis with Custom Parameters

```bash
timbre-sequence-analyze \
  --files audio.wav \
  --out_dir ./output \
  --segment_duration 0.5 \
  --target_points 100 \
  --fusion_method weighted \
  --use_dtw \
  --curve_smooth 5
```

## Troubleshooting

### Common Issues

1. **Command not found**: Ensure packages are installed with `pip install -e .`
2. **GUI not working**: Use macOS fallback or specify files via command line
3. **Import errors**: Check that all dependencies are installed
4. **Memory issues**: Reduce `--target_points` for large files

### Getting Help

- Check `PROJECT_STATUS.md` for known issues and solutions
- Review `USAGE_GUIDE.md` for detailed usage instructions
- Ensure audio files are in supported formats (WAV, FLAC, MP3, etc.)

## Citation

If you use this toolset in your research, please cite the associated paper.

## License

MIT License

## Author

Outora
