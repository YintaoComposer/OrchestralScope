from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def list_audio_files(in_dir: Path, include_glob: Optional[List[str]] = None) -> List[Path]:
    in_dir = Path(in_dir)
    files: List[Path] = []
    if include_glob:
        for pat in include_glob:
            for p in sorted(in_dir.rglob(pat)):
                if p.is_file():
                    files.append(p)
    else:
        for p in sorted(in_dir.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS and p.is_file():
                files.append(p)
    # Deduplicate and sort
    uniq = sorted({str(p.resolve()): p for p in files}.values(), key=lambda x: str(x))
    return uniq


def resolve_input_files(in_dir: Optional[Path], files: Optional[List[Path]], include_glob: Optional[List[str]] = None) -> List[Path]:
    if files:
        # Normalize paths, only keep existing files
        picked = []
        for f in files:
            p = Path(f)
            if p.is_file():
                picked.append(p.resolve())
        if not picked:
            return []
        return sorted(picked, key=lambda x: str(x))
    if in_dir is not None:
        return list_audio_files(in_dir, include_glob=include_glob)
    return []


def write_segments_csv(out_dir: Path, track_name: str, segments: List[Tuple[float, float, int, str]]) -> None:
    # segments: (start_sec, end_sec, label_id, color)
    df = pd.DataFrame(segments, columns=["start", "end", "label", "color"])
    (Path(out_dir) / "segments" / f"{track_name}.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "segments" / f"{track_name}.csv", index=False)


def write_sequence_txt(out_dir: Path, track_name: str, label_letters: List[str]) -> None:
    s = "-".join(label_letters)
    (Path(out_dir) / "sequences").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "sequences", f"{track_name}.txt").write_text(s)


def write_distribution_csv(out_dir: Path, track_to_p: Dict[str, np.ndarray], letters: List[str]) -> None:
    rows = []
    for name, p in track_to_p.items():
        row = {"track": name}
        for i, L in enumerate(letters):
            row[L] = float(p[i])
        rows.append(row)
    df = pd.DataFrame(rows)
    (Path(out_dir) / "stats").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "stats" / "distribution.csv", index=False)


def write_markov_csv(out_dir: Path, track_to_T: Dict[str, np.ndarray], letters: List[str]) -> None:
    records = []
    for name, T in track_to_T.items():
        for i, Li in enumerate(letters):
            for j, Lj in enumerate(letters):
                records.append({"track": name, "from": Li, "to": Lj, "p": float(T[i, j])})
    df = pd.DataFrame(records)
    (Path(out_dir) / "stats").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "stats" / "markov.csv", index=False)


def write_distance_matrix(out_dir: Path, name: str, labels: List[str], D: np.ndarray) -> None:
    df = pd.DataFrame(D, index=labels, columns=labels)
    (Path(out_dir) / "distance").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "distance" / f"{name}.csv")


def write_embedding_csv(out_dir: Path, name: str, labels: List[str], X: np.ndarray) -> None:
    df = pd.DataFrame(X, index=labels)
    (Path(out_dir) / "embedding").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "embedding" / f"{name}.csv")


def write_complexity_series(out_dir: Path, track_name: str, beat_times: np.ndarray, C_raw: np.ndarray, C_smooth: np.ndarray) -> None:
    df = pd.DataFrame({
        'beat_time': beat_times[:len(C_raw)],
        'C_raw': C_raw,
        'C_smooth': C_smooth if len(C_smooth) == len(C_raw) else C_raw,
    })
    (Path(out_dir) / "complexity").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "complexity" / f"{track_name}_complexity.csv", index=False)


def write_complexity_summary(out_dir: Path, rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)
    (Path(out_dir) / "complexity").mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "complexity" / "summary.csv", index=False)


