"""
Timbre Sequence Analysis CLI
Specialized for 0.5s fixed segmentation and multi-feature fusion analysis
"""
import argparse
from pathlib import Path
import numpy as np
from .utils import ensure_out_dirs, macos_choose_files, macos_choose_folder, macos_choose_output_dir
from .audio import preprocess_file, AudioConfig
from .analysis import FineAnalysisConfig, analyze_multiple_tracks_fine
from .io import resolve_input_files, write_fine_analysis_summary, write_fine_features_csv, write_fine_segments_info
from .viz import create_all_visualizations


def build_parser() -> argparse.ArgumentParser:
    """Build timbre sequence analysis specific argument parser"""
    p = argparse.ArgumentParser(
        prog="timbre-sequence-analyze",
        description="Timbre sequence analysis: 0.5s fixed segmentation and multi-feature fusion analysis",
    )
    
    # Input/Output
    p.add_argument("--in_dir", type=Path, required=False, help="Input audio folder")
    p.add_argument("--out_dir", type=Path, required=False, help="Output directory")
    p.add_argument("--files", type=Path, nargs='*', default=None, help="Specify audio files directly")
    p.add_argument("--include_glob", type=str, nargs='*', default=None, help="Filter files in --in_dir with glob patterns")
    p.add_argument("--gui", action="store_true", help="Open GUI dialog to select files/folders and output directory")

    # Audio preprocessing
    p.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    p.add_argument("--pre_emphasis", type=float, default=0.97, help="Pre-emphasis coefficient")

    # Timbre sequence analysis parameters
    p.add_argument("--segment_duration", type=float, default=0.5, help="Fixed segment duration (seconds)")
    p.add_argument("--curve_smooth", type=int, default=5, help="Curve smoothing window size")
    p.add_argument("--target_points", type=int, default=100, help="Target number of feature points")
    p.add_argument("--fusion_method", type=str, default="weighted", choices=["weighted", "pca"], help="Feature fusion method")
    p.add_argument("--time_normalize", type=str, default="interpolate", choices=["interpolate", "pad", "truncate"], help="Time normalization method")
    
    # Visualization options
    p.add_argument("--no_viz", action="store_true", help="Skip visualization generation")
    p.add_argument("--use_dtw", action="store_true", help="Use DTW alignment before cosine similarity for detailed similarities")

    return p


def main():
    """Timbre sequence analysis main function"""
    parser = build_parser()
    args = parser.parse_args()
    
    # GUI selection (optional)
    if args.gui or (args.in_dir is None and args.files is None) or args.out_dir is None:
        print("[timbre-sequence-analyze] Opening GUI for file selection...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Select files
            filetypes = [
                ("Audio files", ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a", "*.aiff", "*.aif")),
                ("All files", "*.*"),
            ]
            print("Please select audio files in the dialog...")
            picked_files = filedialog.askopenfilenames(title="Select audio files to analyze (multiple selection allowed)", filetypes=filetypes)
            files_gui = [Path(f) for f in picked_files] if picked_files else None
            in_dir_gui = None
            
            if not files_gui:
                print("No files selected, please select input folder...")
                d = filedialog.askdirectory(title="Select input folder (will recursively scan for audio)")
                in_dir_gui = Path(d) if d else None
            
            print("Please select output directory...")
            out_dir_gui = None
            d_out = filedialog.askdirectory(title="Select output directory")
            out_dir_gui = Path(d_out) if d_out else None
            
            # Clean up tkinter
            root.destroy()
            
            if files_gui:
                args.files = files_gui
                print(f"Selected {len(files_gui)} files")
            elif in_dir_gui:
                args.in_dir = in_dir_gui
                print(f"Selected input directory: {in_dir_gui}")
            if out_dir_gui:
                args.out_dir = out_dir_gui
                print(f"Selected output directory: {out_dir_gui}")
                
        except Exception as e:
            print(f"[timbre-sequence-analyze] Tkinter not available ({e}), trying macOS dialog...")
            try:
                print("Using macOS file dialog...")
                files_gui = macos_choose_files()
                in_dir_gui = None
                if not files_gui:
                    print("No files selected, trying folder selection...")
                    in_dir_gui = macos_choose_folder()
                print("Selecting output directory...")
                out_dir_gui = macos_choose_output_dir()
                
                if files_gui:
                    args.files = files_gui
                    print(f"Selected {len(files_gui)} files via macOS dialog")
                elif in_dir_gui:
                    args.in_dir = in_dir_gui
                    print(f"Selected input directory: {in_dir_gui}")
                if out_dir_gui:
                    args.out_dir = out_dir_gui
                    print(f"Selected output directory: {out_dir_gui}")
            except Exception as e2:
                print(f"[timbre-sequence-analyze] GUI selection failed: {e2}")
                print("Please use command line arguments instead:")
                print("  --files file1.wav file2.wav --out_dir ./output")
                print("  --in_dir ./input_folder --out_dir ./output")

    if args.out_dir is None:
        print("[timbre-sequence-analyze] Need to specify --out_dir or select output directory in GUI.")
        return

    ensure_out_dirs(args.out_dir)

    # Resolve input files
    files = resolve_input_files(args.in_dir, args.files, include_glob=args.include_glob)
    if not files:
        print("[timbre-sequence-analyze] No audio files found. Use --gui to open file selection; or check paths/patterns.")
        return

    print(f"[timbre-sequence-analyze] Found {len(files)} audio files")

    # Configure audio preprocessing
    acfg = AudioConfig(target_sr=args.sr, pre_emphasis=args.pre_emphasis)
    
    # Configure timbre sequence analysis
    fine_cfg = FineAnalysisConfig(
        segment_duration=args.segment_duration,
        curve_smooth_window=args.curve_smooth,
        time_normalize_method=args.time_normalize,
        target_points=args.target_points,
        fusion_method=args.fusion_method
    )

    print(f"[timbre-sequence-analyze] Starting timbre sequence analysis...")
    print(f"  Segment duration: {args.segment_duration}s")
    print(f"  Smoothing window: {args.curve_smooth}")
    print(f"  Target points: {args.target_points}")
    print(f"  Fusion method: {args.fusion_method}")

    # Prepare data
    track_data = []
    for p in files:
        name = p.stem
        print(f"  Preprocessing: {name}")
        y, sr = preprocess_file(p, acfg)
        track_data.append((name, y, sr))

    # Execute timbre sequence analysis
    fine_results = analyze_multiple_tracks_fine(track_data, fine_cfg)

    # Save results
    print(f"[timbre-sequence-analyze] Saving analysis results...")
    
    # Save fused features
    np.save(args.out_dir / "fused_features.npy", fine_results['all_fused_features'])
    
    # Save detailed results for each track
    for name, result in fine_results['track_results'].items():
        track_dir = args.out_dir / name
        track_dir.mkdir(exist_ok=True)
        
        # Save fused features
        np.save(track_dir / "fused_features.npy", result['fused_features'])
        
        # Save segment information
        np.save(track_dir / "segment_times.npy", result['segment_times'])
    
    # Write summary information
    write_fine_analysis_summary(args.out_dir, fine_results)
    write_fine_features_csv(args.out_dir, fine_results)
    write_fine_segments_info(args.out_dir, fine_results)
    
    # Create visualizations (unless disabled)
    if not args.no_viz:
        # Always create all visualizations (includes hierarchical clustering and CSV)
        create_all_visualizations(fine_results, args.out_dir, use_dtw=args.use_dtw)
    
    print(f"[timbre-sequence-analyze] Analysis complete!")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Fused feature dimension: {fine_results['all_fused_features'].shape}")
    print(f"  Generated files:")
    print(f"    - fine_analysis_summary.json: Analysis summary")
    print(f"    - fine_features.csv: Fused feature data")
    print(f"    - fine_segments_info.csv: Segment information")
    print(f"    - fused_features.npy: Global fused features")
    print(f"    - {len(fine_results['track_names'])} track subdirectories")
    if not args.no_viz:
        print(f"    - visualizations/ directory with PNG plots")


if __name__ == "__main__":
    main()

