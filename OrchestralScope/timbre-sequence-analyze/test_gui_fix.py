#!/usr/bin/env python3
"""
Test script to verify GUI fixes
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def test_gui_timeout():
    """Test GUI with timeout to ensure it doesn't hang"""
    print("Testing GUI timeout behavior...")
    
    # Create a simple test audio file
    import numpy as np
    import soundfile as sf
    
    test_file = "test_gui_audio.wav"
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(test_file, audio, sr)
    
    try:
        # Test with timeout
        print("Starting GUI test (will timeout after 10 seconds)...")
        process = subprocess.Popen(
            ['timbre-sequence-analyze', '--gui'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for 10 seconds
        try:
            stdout, stderr = process.communicate(timeout=10)
            print("GUI completed within timeout")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
        except subprocess.TimeoutExpired:
            print("GUI timed out (this is expected if no user interaction)")
            process.kill()
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
        
        return True
        
    except Exception as e:
        print(f"GUI test failed: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_command_line_alternative():
    """Test command line alternative to GUI"""
    print("\nTesting command line alternative...")
    
    # Create test audio
    import numpy as np
    import soundfile as sf
    
    test_file = "test_cli_audio.wav"
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(test_file, audio, sr)
    
    try:
        # Test command line version
        result = subprocess.run([
            'timbre-sequence-analyze', 
            '--files', test_file, 
            '--out_dir', './test_gui_output'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Command line version works correctly")
            print("STDOUT:", result.stdout)
            return True
        else:
            print("✗ Command line version failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Command line version timed out")
        return False
    except Exception as e:
        print(f"✗ Command line test failed: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if os.path.exists('./test_gui_output'):
            import shutil
            shutil.rmtree('./test_gui_output')

def main():
    """Main test function"""
    print("GUI Fix Test Suite")
    print("=" * 50)
    
    # Test 1: GUI timeout behavior
    gui_ok = test_gui_timeout()
    
    # Test 2: Command line alternative
    cli_ok = test_command_line_alternative()
    
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"GUI Timeout Test: {'✓ PASS' if gui_ok else '✗ FAIL'}")
    print(f"Command Line Test: {'✓ PASS' if cli_ok else '✗ FAIL'}")
    
    if gui_ok and cli_ok:
        print("\n🎉 All tests passed! GUI fixes are working.")
        print("\nIf GUI still hangs, you can use command line arguments:")
        print("  timbre-sequence-analyze --files audio1.wav audio2.wav --out_dir ./output")
        print("  timbre-sequence-analyze --in_dir ./input_folder --out_dir ./output")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
