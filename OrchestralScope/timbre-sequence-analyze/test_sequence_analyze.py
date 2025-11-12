#!/usr/bin/env python3
"""
Test script for timbre sequence analysis
"""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf


def create_test_audio(filename="test_audio.wav", duration=10.0):
    """Create test audio file"""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create multi-segment audio with different characteristics
    audio = np.zeros_like(t)
    
    # Segment 1: Low frequency
    audio[:int(sr * 2)] = 0.3 * np.sin(2 * np.pi * 220 * t[:int(sr * 2)])
    
    # Segment 2: High frequency
    audio[int(sr * 2):int(sr * 4)] = 0.3 * np.sin(2 * np.pi * 880 * t[:int(sr * 2)])
    
    # Segment 3: Noise
    audio[int(sr * 4):int(sr * 6)] = 0.2 * np.random.randn(int(sr * 2))
    
    # Segment 4: Complex signal
    audio[int(sr * 6):int(sr * 8)] = 0.2 * (np.sin(2 * np.pi * 440 * t[:int(sr * 2)]) + 
                                           0.5 * np.sin(2 * np.pi * 660 * t[:int(sr * 2)]))
    
    # Segment 5: Decaying signal
    audio[int(sr * 8):] = 0.3 * np.sin(2 * np.pi * 330 * t[:int(sr * 2)]) * np.exp(-t[:int(sr * 2)])
    
    sf.write(filename, audio, sr)
    return filename


def test_basic_functionality():
    """Test basic timbre sequence analysis functionality"""
    print("🧪 Testing Timbre Sequence Analysis")
    print("=" * 40)
    
    # Create test audio
    test_file = create_test_audio("test_fine.wav", duration=8.0)
    print(f"✓ Created test audio: {test_file}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            
            # Test basic analysis
            print("\n1. Testing basic analysis...")
            cmd = f"timbre-sequence-analyze --files {test_file} --out_dir {output_dir}"
            result = os.system(cmd)
            
            if result == 0:
                print("   ✓ Basic analysis successful")
                
                # Check output files
                expected_files = [
                    "fine_analysis_summary.json",
                    "fine_features.csv",
                    "fine_segments_info.csv",
                    "fused_features.npy"
                ]
                
                for filename in expected_files:
                    file_path = output_dir / filename
                    if file_path.exists():
                        print(f"   ✓ {filename} exists")
                        
                        # Check file content
                        if filename.endswith('.json'):
                            import json
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                print(f"     Config: {data.get('config', {})}")
                        elif filename.endswith('.npy'):
                            data = np.load(file_path)
                            print(f"     Shape: {data.shape}")
                    else:
                        print(f"   ✗ {filename} missing")
                        return False
            else:
                print("   ✗ Basic analysis failed")
                return False
            
            # Test with different parameters
            print("\n2. Testing different parameters...")
            cmd2 = f"timbre-sequence-analyze --files {test_file} --out_dir {output_dir} --segment_duration 0.3 --target_points 50"
            result2 = os.system(cmd2)
            
            if result2 == 0:
                print("   ✓ Parameter customization successful")
            else:
                print("   ✗ Parameter customization failed")
                return False
            
            print("\n🎉 All tests passed!")
            return True
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False
    
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✓ Cleaned up test file: {test_file}")


def test_gui_functionality():
    """Test GUI functionality (simulation)"""
    print("\n🖥️  Testing GUI functionality...")
    
    # Test GUI parameter (but don't actually open GUI)
    test_file = create_test_audio("test_gui.wav", duration=5.0)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            
            # Test GUI mode (should work even without actual GUI)
            cmd = f"timbre-sequence-analyze --files {test_file} --out_dir {output_dir} --gui"
            result = os.system(cmd)
            
            if result == 0:
                print("   ✓ GUI functionality works")
                return True
            else:
                print("   ✗ GUI functionality failed")
                return False
                
    except Exception as e:
        print(f"   ✗ GUI test failed: {e}")
        return False
    
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    """Main test function"""
    print("Timbre Sequence Analysis Test Suite")
    print("=" * 50)
    
    # Check if package is installed
    try:
        import timbre_sequence_analyze
        print(f"✓ Package installed: {timbre_sequence_analyze.__file__}")
    except ImportError as e:
        print(f"✗ Package not installed: {e}")
        print("Please install the package first: pip install -e .")
        return False
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("GUI Functionality", test_gui_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Timbre sequence analysis is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

