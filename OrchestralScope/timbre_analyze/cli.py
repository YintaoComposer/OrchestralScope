import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="timbre-analyze",
        description="Audio timbre segmentation, clustering, and visualization",
    )
    p.add_argument("--in_dir", type=Path, required=False)
    p.add_argument("--out_dir", type=Path, required=False)
    p.add_argument("--files", type=Path, nargs='*', default=None, help="Specify audio files directly")
    p.add_argument("--include_glob", type=str, nargs='*', default=None, help="Filter files in --in_dir with glob patterns, e.g. '*.wav' '*.flac'")
    p.add_argument("--gui", action="store_true", help="Open GUI dialog to select files/folders and output directory")

    # preprocessing
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--pre_emphasis", type=float, default=0.97)
    p.add_argument("--frame", type=str, default="46ms")
    p.add_argument("--hop", type=str, default="10ms")

    # features
    p.add_argument(
        "--features",
        type=str,
        default="mfcc,delta,delta2,chroma,scflux",
    )

    # segmentation
    p.add_argument("--seg_method", type=str, default="novelty")
    p.add_argument("--seg_kernel_gauss", type=float, default=3.0)
    p.add_argument("--beat_sync", type=str, default="on")
    p.add_argument("--min_seg_beats", type=int, default=2)
    p.add_argument("--adaptive_seg", action="store_true", help="Enable adaptive segmentation (adjust segmentation density based on music style)")
    p.add_argument("--min_seg_beats_traditional", type=int, default=4, help="Minimum segment length for traditional works (in beats)")
    p.add_argument("--min_seg_beats_contemporary", type=int, default=1, help="Minimum segment length for contemporary works (in beats)")

    # clustering & K selection
    p.add_argument("--cluster_method", type=str, default="gmm")
    p.add_argument("--k_policy", type=str, default="auto", help="auto=unified K selection for entire library; per_track=automatic K selection per track; fixed=fixed K")
    p.add_argument("--k_min", type=int, default=4)
    p.add_argument("--k_max", type=int, default=12)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--k_select", type=str, default="bic", help="K selection method: bic|aic|silhouette|db|combined|feature_weighted|conservative")
    p.add_argument("--bic_weight", type=float, default=0.4, help="BIC weight in combined scoring (only for combined method)")
    p.add_argument("--cosine_silhouette_weight", type=float, default=0.3, help="Cosine Silhouette weight in combined scoring (only for combined method)")
    p.add_argument("--gap_weight", type=float, default=0.3, help="Gap statistic weight in combined scoring (only for combined method)")
    # Feature weight parameters (based on timbre-fine-analyze weights)
    p.add_argument("--mfcc_weight", type=float, default=0.50, help="MFCC feature weight (50% - main feature)")
    p.add_argument("--spectral_centroid_weight", type=float, default=0.20, help="Spectral centroid weight (20% - timbre brightness)")
    p.add_argument("--spectral_bandwidth_weight", type=float, default=0.10, help="Spectral bandwidth weight (10% - timbre complexity)")
    p.add_argument("--zcr_weight", type=float, default=0.10, help="Zero crossing rate weight (10% - speech/percussion identification)")
    p.add_argument("--spectral_rolloff_weight", type=float, default=0.06, help="Spectral rolloff weight (6% - moderate contribution)")
    p.add_argument("--rms_weight", type=float, default=0.04, help="RMS energy weight (4% - energy representation)")
    # Clustering sensitivity control
    p.add_argument("--clustering_sensitivity", type=str, default="normal", help="Clustering sensitivity: conservative|normal|aggressive")
    p.add_argument("--min_cluster_size_ratio", type=float, default=0.05, help="Minimum cluster size ratio (relative to total segments)")

    # smoothing
    p.add_argument("--smooth", type=str, default="hmm")
    p.add_argument("--hmm_selfbias", type=float, default=0.6)

    # labeling & palette
    p.add_argument("--label_scheme", type=str, default="global")
    p.add_argument(
        "--palette",
        type=str,
        default="A:#E53935,B:#FB8C00,C:#FDD835,D:#43A047,E:#1E88E5,F:#8E24AA,G:#6D4C41,H:#00ACC1",
    )

    # distances
    p.add_argument("--seq_distance", type=str, default="levenshtein")
    p.add_argument("--dist_distribution", type=str, default="jsd")
    p.add_argument("--dist_markov", type=str, default="frobenius")
    p.add_argument("--alpha", type=float, default=0.6)

    # embedding
    p.add_argument("--embed", type=str, default="umap")
    p.add_argument("--embed_dim", type=int, default=2)


    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Defer imports to reduce CLI cold start time
    from .utils import ensure_out_dirs, macos_choose_files, macos_choose_folder, macos_choose_output_dir, remap_labels_by_first_occurrence
    import numpy as np
    import re
    from .palette import parse_palette
    from .io import resolve_input_files, write_segments_csv, write_sequence_txt, write_distribution_csv, write_markov_csv, write_distance_matrix, write_embedding_csv
    from .audio import preprocess_file, AudioConfig
    from .features import compute_features, FeatureConfig
    from .segmentation import segment_audio, frames_to_segments, SegmentationConfig
    from .clustering import fit_gmm_k_candidates, select_k, KSelectConfig, assign_global_labels, fit_gmm_auto_for_track
    from .smoothing import hmm_smooth_labels
    from .metrics import distribution_vector, markov_matrix, js_divergence, levenshtein, frobenius_distance, combined_distance
    from .embedding import embed_features
    from .viz import plot_embedding_scatter
    from .complexity import ComplexityConfig, track_beats, track_complexity
    from .viz import plot_complexity_timeline

    # GUI selection (optional)
    if args.gui or (args.in_dir is None and args.files is None or args.out_dir is None):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            # Try to select files first; if cancelled, select input folder
            filetypes = [
                ("Audio files", ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a", "*.aiff", "*.aif")),
                ("All files", "*.*"),
            ]
            picked_files = filedialog.askopenfilenames(title="Select audio files to analyze (multiple selection allowed)", filetypes=filetypes)
            files_gui = [Path(f) for f in picked_files] if picked_files else None
            in_dir_gui = None
            if not files_gui:
                d = filedialog.askdirectory(title="Select input folder (will recursively scan for audio)")
                in_dir_gui = Path(d) if d else None
            out_dir_gui = None
            d_out = filedialog.askdirectory(title="Select output directory")
            out_dir_gui = Path(d_out) if d_out else None

            if files_gui:
                args.files = files_gui
            if in_dir_gui is not None:
                args.in_dir = in_dir_gui
            if out_dir_gui is not None:
                args.out_dir = out_dir_gui
        except Exception as e:
            print(f"[timbre-analyze] Tkinter not available ({e}), using macOS selection dialog.")
            files_gui = macos_choose_files()
            in_dir_gui = None
            if not files_gui:
                in_dir_gui = macos_choose_folder()
            out_dir_gui = macos_choose_output_dir()
            if files_gui:
                args.files = files_gui
            if in_dir_gui is not None:
                args.in_dir = in_dir_gui
            if out_dir_gui is not None:
                args.out_dir = out_dir_gui

    if args.out_dir is None:
        print("[timbre-analyze] Need to specify --out_dir or select output directory in GUI.")
        return

    ensure_out_dirs(args.out_dir)

    files = resolve_input_files(args.in_dir, args.files, include_glob=args.include_glob)
    if not files:
        print("[timbre-analyze] No audio files found. Use --gui to open file selection; or check paths/patterns.")
        return

    # Preprocessing and features
    acfg = AudioConfig(target_sr=args.sr, pre_emphasis=args.pre_emphasis)
    fcfg = FeatureConfig(
        frame_ms=float(args.frame.replace("ms", "")),
        hop_ms=float(args.hop.replace("ms", "")),
    )
    segcfg = SegmentationConfig(
        kernel_gauss=args.seg_kernel_gauss,
        beat_sync=(args.beat_sync.lower() == "on"),
        min_seg_beats=args.min_seg_beats,
        adaptive_segmentation=args.adaptive_seg,
        min_seg_beats_traditional=args.min_seg_beats_traditional,
        min_seg_beats_contemporary=args.min_seg_beats_contemporary,
    )

    track_names = []
    track_feats_mean = []  # For library-wide clustering (segment mean pool)
    per_track = []  # Temporary storage for segments and frame feature mapping

    for p in files:
        name = p.stem
        y, sr = preprocess_file(p, acfg)
        F, _ = compute_features(y, sr, fcfg)
        bounds, times = segment_audio(y, sr, fcfg.frame_ms, fcfg.hop_ms, segcfg)
        segs = frames_to_segments(bounds, F.shape[1])
        # Segment mean as segment feature
        seg_feats = []
        for a, b in segs:
            seg_feats.append(F[:, a:b].mean(axis=1))
        seg_feats = np.stack(seg_feats, axis=0)

        per_track.append((name, segs, times, seg_feats))
        track_names.append(name)
        track_feats_mean.append(seg_feats)

    # Complexity configuration
    cplx_cfg = ComplexityConfig()

    all_seg_feats = np.concatenate(track_feats_mean, axis=0)

    # Clustering and K selection (GMM only)
    if args.k_policy == "fixed" and args.k is not None:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=int(args.k), covariance_type="full", random_state=42).fit(all_seg_feats)
        K = int(args.k)
    elif args.k_policy == "auto":
        models = fit_gmm_k_candidates(all_seg_feats, args.k_min, args.k_max)
        
        # Build feature weight dictionary
        feature_weights = {
            'mfcc': args.mfcc_weight,
            'spectral_centroid': args.spectral_centroid_weight,
            'spectral_bandwidth': args.spectral_bandwidth_weight,
            'zcr': args.zcr_weight,
            'spectral_rolloff': args.spectral_rolloff_weight,
            'rms': args.rms_weight
        }
        
        K = select_k(all_seg_feats, models, KSelectConfig(
            method=args.k_select, 
            k_min=args.k_min, 
            k_max=args.k_max,
            bic_weight=args.bic_weight,
            cosine_silhouette_weight=args.cosine_silhouette_weight,
            gap_weight=args.gap_weight,
            feature_weights=feature_weights,
            clustering_sensitivity=args.clustering_sensitivity,
            min_cluster_size_ratio=args.min_cluster_size_ratio
        ))
        gmm = models[K]
    else:  # per_track mode: independent K selection and GMM fitting for each track
        gmm = None
        K = None

    # Label mapping A/B/...
    # palette does not depend on K, parse in advance
    palette = parse_palette(args.palette)

    # Generate labels, smoothing, statistics and output for each track
    track_to_p = {}
    track_to_T = {}
    sequences_int = {}
    complexity_rows = []
    for name, segs, times, seg_feats in per_track:
        if args.k_policy == "per_track":
            # Build feature weight dictionary
            feature_weights = {
                'mfcc': args.mfcc_weight,
                'spectral_centroid': args.spectral_centroid_weight,
                'spectral_bandwidth': args.spectral_bandwidth_weight,
                'zcr': args.zcr_weight,
                'spectral_rolloff': args.spectral_rolloff_weight,
                'rms': args.rms_weight
            }
            
            # Create configuration for per_track mode
            cfg = KSelectConfig(
                method=args.k_select,
                k_min=args.k_min,
                k_max=args.k_max,
                bic_weight=args.bic_weight,
                cosine_silhouette_weight=args.cosine_silhouette_weight,
                gap_weight=args.gap_weight,
                feature_weights=feature_weights,
                clustering_sensitivity=args.clustering_sensitivity,
                min_cluster_size_ratio=args.min_cluster_size_ratio
            )
            models_t = fit_gmm_k_candidates(seg_feats, args.k_min, args.k_max)
            K_t = select_k(seg_feats, models_t, cfg)
            gmm_t = models_t[K_t]
            labels = assign_global_labels(seg_feats, gmm_t)
            labels_sm = hmm_smooth_labels(labels, num_classes=K_t, self_bias=args.hmm_selfbias)
            letters_local = [chr(ord('A') + i) for i in range(K_t)]
        else:
            labels = assign_global_labels(seg_feats, gmm)
            labels_sm = hmm_smooth_labels(labels, num_classes=K, self_bias=args.hmm_selfbias)
            letters_local = [chr(ord('A') + i) for i in range(K)]
        # Remap based on "first occurrence order", making the first timbre 0, second new timbre 1, and so on
        labels_sm, old2new = remap_labels_by_first_occurrence(labels_sm)
        # Synchronously update letter and color mapping (A,B,...) based on new indices
        letters_use = letters_local
        # Output segments CSV and sequence TXT
        segments_rows = []
        letter_seq = []
        for (a, b), lab in zip(segs, labels_sm):
            L = letters_use[lab]
            c = palette.get(L, "#000000")
            segments_rows.append((float(times[np.where(times == times)[0][0]] if len(times) else 0.0), float(times[min(len(times) - 1, np.where(times == times)[0][0])]) if len(times) else 0.0, int(lab), c))
            letter_seq.append(L)
        # Fix time using frame index mapping to seconds
        # Recalculate more rigorously:
        segments_rows = []
        for (a, b), lab in zip(segs, labels_sm):
            start_sec = (a * fcfg.hop_ms) / 1000.0
            end_sec = (b * fcfg.hop_ms) / 1000.0
            L = letters_use[lab]
            c = palette.get(L, "#000000")
            segments_rows.append((start_sec, end_sec, int(lab), c))
        write_segments_csv(args.out_dir, name, segments_rows)
        write_sequence_txt(args.out_dir, name, letter_seq)

        # Calculate distribution and transition
        numK = len(letters_use)
        pvec = distribution_vector(labels_sm, numK)
        Tm = markov_matrix(labels_sm, numK)
        track_to_p[name] = pvec
        track_to_T[name] = Tm
        sequences_int[name] = labels_sm.tolist()

        # Complexity: beat tracking + timeline + peaks + full track complexity
        hop_length = int(round(fcfg.hop_ms * 1e-3 * sr))
        _, beat_times = track_beats(y, sr, hop_length)
        # Segment boundary times (seconds)
        boundary_times = []
        for (a, b) in segs:
            boundary_times.append((a * fcfg.hop_ms) / 1000.0)
        boundary_times = np.array(boundary_times, dtype=float)
        cplx = track_complexity(boundary_times, beat_times, cplx_cfg, labels_sm)
        from .io import write_complexity_series
        write_complexity_series(args.out_dir, name, cplx['beat_times'], cplx['C_raw'], cplx['C_smooth'])
        plot_complexity_timeline(args.out_dir, name, cplx['beat_times'], cplx['C_raw'], cplx['C_smooth'], cplx['peak'])
        # Summary row
        row = {
            'track': name,
            'C_track_per_beat': cplx['C_track'],
            'C_track_per_sec': cplx['C_time'],
        }
        if cplx['peak'] is not None:
            row.update({
                'peak_beat': cplx['peak']['peak_beat'],
                'peak_r': cplx['peak']['peak_r'],
                'peak_time': cplx['peak']['peak_time'],
                'peak_C': cplx['peak']['peak_C'],
                'prominence': cplx['peak']['prominence'],
                'width_beats': cplx['peak']['width_beats'],
            })
        complexity_rows.append(row)

    # Write distribution and Markov
    # Library-level distribution/Markov: in per_track mode, alphabet is not unified.
    # Pad each track's alphabet length to the library's maximum K (pad with 0) to avoid out of bounds.
    if args.k_policy == "per_track":
        maxK = max((len(pvec) for pvec in track_to_p.values()), default=0)
        letters_header = [chr(ord('A') + i) for i in range(maxK)]
        # Pad distribution
        track_to_p_padded = {}
        for name, p in track_to_p.items():
            if len(p) < maxK:
                p_pad = np.pad(p, (0, maxK - len(p)))
            else:
                p_pad = p
            track_to_p_padded[name] = p_pad
        # Pad Markov
        track_to_T_padded = {}
        for name, T in track_to_T.items():
            kcur = T.shape[0]
            if kcur < maxK:
                T_pad = np.pad(T, ((0, maxK - kcur), (0, maxK - kcur)))
            else:
                T_pad = T
            track_to_T_padded[name] = T_pad
        write_distribution_csv(args.out_dir, track_to_p_padded, letters_header)
        write_markov_csv(args.out_dir, track_to_T_padded, letters_header)
    else:
        letters_header = [chr(ord('A') + i) for i in range(K)]
        write_distribution_csv(args.out_dir, track_to_p, letters_header)
        write_markov_csv(args.out_dir, track_to_T, letters_header)

    # Distance matrix
    N = len(track_names)
    D_jsd = np.zeros((N, N))
    D_lev = np.zeros((N, N))
    D_mark = np.zeros((N, N))
    D_comb = np.zeros((N, N))
    # In per_track mode, pad p/T with zeros to maxK first to ensure same dimension for comparison
    if args.k_policy == "per_track":
        maxK = len(letters_header)
        track_to_p_for_dist = {}
        track_to_T_for_dist = {}
        for n in track_names:
            p = track_to_p[n]
            T = track_to_T[n]
            if len(p) < maxK:
                p = np.pad(p, (0, maxK - len(p)))
                T = np.pad(T, ((0, maxK - T.shape[0]), (0, maxK - T.shape[1])))
            track_to_p_for_dist[n] = p
            track_to_T_for_dist[n] = T
    else:
        track_to_p_for_dist = track_to_p
        track_to_T_for_dist = track_to_T

    for i in range(N):
        for j in range(N):
            pi = track_to_p_for_dist[track_names[i]]
            pj = track_to_p_for_dist[track_names[j]]
            D_jsd[i, j] = js_divergence(pi, pj)
            li = sequences_int[track_names[i]]
            lj = sequences_int[track_names[j]]
            D_lev[i, j] = levenshtein(li, lj)
            Ti = track_to_T_for_dist[track_names[i]]
            Tj = track_to_T_for_dist[track_names[j]]
            D_mark[i, j] = frobenius_distance(Ti, Tj)
            # Normalize sequence distance to [0,1] (by maximum length)
            max_len = max(len(li), len(lj)) or 1
            D_order_norm = D_lev[i, j] / max_len
            D_comb[i, j] = combined_distance(D_order_norm, D_jsd[i, j], alpha=args.alpha)

    write_distance_matrix(args.out_dir, "jsd", track_names, D_jsd)
    write_distance_matrix(args.out_dir, "levenshtein", track_names, D_lev)
    write_distance_matrix(args.out_dir, "markov", track_names, D_mark)
    write_distance_matrix(args.out_dir, "combined", track_names, D_comb)
    # Hierarchical clustering based on combined distance matrix (average linkage)
    try:
        from .viz import plot_hierarchical_from_distance
        plot_hierarchical_from_distance(
            args.out_dir,
            track_names,
            D_comb,
            method="average",
            title="Hierarchical clustering on combined distance (average linkage)",
        )
    except Exception as e:
        print(f"[timbre-analyze] Hierarchical clustering plot failed: {e}")

    # Embedding (using distribution vector + transition matrix flattened and concatenated)
    X_repr = []
    for n in track_names:
        p = track_to_p[n]
        T = track_to_T[n]
        # For per_track scenario, align to maximum K (pad)
        if args.k_policy == "per_track":
            Kp = len(p)
            maxK = len(letters_header)
            if Kp < maxK:
                pad_p = np.pad(p, (0, maxK - Kp))
                pad_T = np.pad(T, ((0, maxK - Kp), (0, maxK - Kp)))
                X_repr.append(np.concatenate([pad_p, pad_T.reshape(-1)]))
            else:
                X_repr.append(np.concatenate([p, T.reshape(-1)]))
        else:
            X_repr.append(np.concatenate([p, T.reshape(-1)]))
    X_repr = np.stack(X_repr, axis=0)
    X_emb = embed_features(X_repr, method=args.embed, dim=args.embed_dim)
    name_csv = f"{args.embed}_{args.embed_dim}d"
    write_embedding_csv(args.out_dir, name_csv, track_names, X_emb)
    # Simultaneously output visualization scatter plot (colored by dominant timbre)
    # Color is taken from the letter color of the category with the largest proportion in each track's distribution
    colors = []
    for n in track_names:
        dom_idx = int(np.argmax(track_to_p[n]))
        # Use each track's alphabet color (A,B,... unchanged)
        L = chr(ord('A') + dom_idx)
        colors.append(palette.get(L, "#000000"))
    # More user-friendly plot: point labels use track names, legend uses letter colors, close to Nature style
    # Annotation text: take the first "word" of the track name (separated by space or underscore), ignore subsequent parts
    def _first_token(name: str) -> str:
        parts = re.split(r"[ _]+", name.strip())
        return parts[0] if parts and parts[0] else name
    labels_for_plot = [_first_token(n) for n in track_names]
    # Title takes the first track name (or its first 10 characters, following 2D/3D logic)
    fig_title = labels_for_plot[0] if labels_for_plot else f"Embedding: {args.embed.upper()} {args.embed_dim}D"
    plot_embedding_scatter(
        args.out_dir,
        name_csv,
        X_emb,
        colors=colors,
        point_labels=labels_for_plot,
        class_letters=letters_header,
        dominant_class_indices=[int(np.argmax(track_to_p[n])) for n in track_names],
        letter_to_color=palette,
        title=fig_title,
        annotate_3d=True,
        legend_loc="upper right",
    )
    print(f"[timbre-analyze] Complete. K={K}, generated analysis results for {len(track_names)} tracks.")
    # Write complexity summary
    from .io import write_complexity_summary
    write_complexity_summary(args.out_dir, complexity_rows)


if __name__ == "__main__":
    main()


