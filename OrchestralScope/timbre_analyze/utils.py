from pathlib import Path
from typing import List, Optional
import subprocess
import numpy as np


RESULT_SUBDIRS = [
    "segments",
    "sequences",
    "stats",
    "distance",
    "embedding",
    "figs",
]


def ensure_out_dirs(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    for sub in RESULT_SUBDIRS:
        Path(out_dir, sub).mkdir(parents=True, exist_ok=True)


# macOS AppleScript selector (can be used without Tk)
def macos_choose_files(title: str = "Select audio files", allow_multiple: bool = True) -> List[Path]:
    # Use multi-line AppleScript, pass through multiple -e, avoid f-string escaping issues
    osa = [
        "osascript",
        "-e", f'display dialog "{title}" with icon note buttons {{"Continue"}}',
        "-e", 'set f to choose file with multiple selections allowed of type {"wav","flac","mp3","ogg","m4a","aiff","aif"}',
        "-e", 'set out to ""',
        "-e", 'repeat with i in f',
        "-e", 'set out to out & (POSIX path of i) & linefeed',
        "-e", 'end repeat',
        "-e", 'out',
    ]
    try:
        out = subprocess.check_output(osa, text=True)
        lines = [line for line in out.splitlines() if line.startswith("/")]
        return [Path(p.strip()) for p in lines]
    except Exception:
        # Directly select files (without showing prompt dialog)
        try:
            osa2 = [
                "osascript",
                "-e", 'set f to choose file with multiple selections allowed of type {"wav","flac","mp3","ogg","m4a","aiff","aif"}',
                "-e", 'set out to ""',
                "-e", 'repeat with i in f',
                "-e", 'set out to out & (POSIX path of i) & linefeed',
                "-e", 'end repeat',
                "-e", 'out',
            ]
            out = subprocess.check_output(osa2, text=True)
            lines = [line for line in out.splitlines() if line.startswith("/")]
            return [Path(p.strip()) for p in lines]
        except Exception:
            return []


def macos_choose_folder(title: str = "Select input folder") -> Optional[Path]:
    try:
        out = subprocess.check_output(["osascript", "-e", f'set d to choose folder with prompt "{title}" \n POSIX path of d'], text=True)
        p = Path(out.strip())
        return p if p.exists() else None
    except Exception:
        return None


def macos_choose_output_dir(title: str = "Select output directory") -> Optional[Path]:
    # choose folder; create if not exists
    return macos_choose_folder(title)


def remap_labels_by_first_occurrence(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Remap labels to 0,1,2,... by first occurrence order.

    For example, original sequence [2,2,0,3,3,2,1] -> first see 2->0, then 0->1, 3->2, 1->3, resulting in [0,0,1,2,2,0,3]

    Returns: remapped sequence and old->new mapping dictionary.
    """
    mapping: dict[int, int] = {}
    next_id = 0
    out = np.empty_like(labels)
    for i, lab in enumerate(labels):
        if int(lab) not in mapping:
            mapping[int(lab)] = next_id
            next_id += 1
        out[i] = mapping[int(lab)]
    return out, mapping


