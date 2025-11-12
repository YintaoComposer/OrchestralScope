"""
Utility functions for timbre sequence analysis
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def ensure_out_dirs(out_dir: Path) -> None:
    """
    Ensure output directories exist
    
    Args:
        out_dir: Output directory path
    """
    out_dir.mkdir(parents=True, exist_ok=True)


def macos_choose_files() -> Optional[List[Path]]:
    """
    Use macOS dialog to choose files
    
    Returns:
        List of selected file paths or None
    """
    try:
        script = [
            'osascript', '-e',
            'tell application "System Events" to return POSIX path of (choose file with prompt "Select audio files" with multiple selections allowed of type {"public.audio"})'
        ]
        result = subprocess.run(script, capture_output=True, text=True, check=True, timeout=60)
        
        if result.stdout.strip():
            # Parse the output - macOS returns newline-separated paths
            paths = [Path(p.strip()) for p in result.stdout.strip().split('\n') if p.strip()]
            return paths
        return None
    except subprocess.TimeoutExpired:
        print("File selection dialog timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"File selection dialog failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in file selection: {e}")
        return None


def macos_choose_folder() -> Optional[Path]:
    """
    Use macOS dialog to choose folder
    
    Returns:
        Selected folder path or None
    """
    try:
        script = [
            'osascript', '-e',
            'tell application "System Events" to return POSIX path of (choose folder with prompt "Select input folder")'
        ]
        result = subprocess.run(script, capture_output=True, text=True, check=True, timeout=60)
        
        if result.stdout.strip():
            return Path(result.stdout.strip())
        return None
    except subprocess.TimeoutExpired:
        print("Folder selection dialog timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Folder selection dialog failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in folder selection: {e}")
        return None


def macos_choose_output_dir() -> Optional[Path]:
    """
    Use macOS dialog to choose output directory
    
    Returns:
        Selected output directory path or None
    """
    try:
        script = [
            'osascript', '-e',
            'tell application "System Events" to return POSIX path of (choose folder with prompt "Select output directory")'
        ]
        result = subprocess.run(script, capture_output=True, text=True, check=True, timeout=60)
        
        if result.stdout.strip():
            return Path(result.stdout.strip())
        return None
    except subprocess.TimeoutExpired:
        print("Output directory selection dialog timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Output directory selection dialog failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in output directory selection: {e}")
        return None

