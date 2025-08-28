import os
import sys
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import subprocess
import tqdm
import psutil

def convert_mp3_to_wav_robust(file_info):
    """
    Convert MP3 to WAV with multiple fallback options and detailed error reporting.
    """
    input_path, output_path = file_info
    ffmpeg_path = r"C:\Users\JHANI\Downloads\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\bin\ffmpeg.exe"
    
    # Try multiple conversion strategies
    conversion_attempts = [
        # Attempt 1: Standard conversion with auto detection
        [ffmpeg_path, "-y", "-i", input_path, "-f", "wav", output_path],
        
        # Attempt 2: Force MP3 decoder and basic WAV output
        [ffmpeg_path, "-y", "-f", "mp3", "-i", input_path, "-acodec", "pcm_s16le", "-ar", "44100", output_path],
        
        # Attempt 3: Most permissive settings
        [ffmpeg_path, "-y", "-i", input_path, "-acodec", "pcm_s16le", output_path],
        
        # Attempt 4: Let ffmpeg auto-detect everything
        [ffmpeg_path, "-y", "-i", input_path, output_path]
    ]
    
    for attempt_num, cmd in enumerate(conversion_attempts, 1):
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                # Verify the output file is valid and not empty
                if os.path.getsize(output_path) > 0:
                    return (True, input_path, output_path, None)
                else:
                    # File was created but is empty, try next method
                    continue
            
        except subprocess.TimeoutExpired:
            return (False, input_path, output_path, f"Timeout after 5 minutes on attempt {attempt_num}")
        except Exception as e:
            continue  # Try next method
    
    # If all attempts failed, return the error from the last attempt
    try:
        result = subprocess.run(conversion_attempts[0], capture_output=True, text=True, timeout=60)
        error_msg = result.stderr.strip() if result.stderr else f"Unknown error (return code: {result.returncode})"
        return (False, input_path, output_path, error_msg)
    except Exception as e:
        return (False, input_path, output_path, str(e))

def test_single_file(input_dir):
    """Test conversion of a single MP3 file to diagnose issues."""
    mp3_files = list(Path(input_dir).rglob("*.mp3"))
    if mp3_files:
        test_file = mp3_files[0]
        test_output = Path(input_dir).parent / "test_output.wav"
        
        print(f"\n Testing single file conversion:")
        print(f"Input: {test_file}")
        print(f"Output: {test_output}")
        
        result = convert_mp3_to_wav_robust((str(test_file), str(test_output)))
        success, inp, out, error = result
        
        if success:
            print("Test conversion successful!")
            os.remove(test_output)  # Clean up test file
            return True
        else:
            print(f" Test conversion failed: {error}")
            return False
    return False

def find_mp3_files(input_directory):
    """Find all MP3 files and validate them."""
    mp3_files = []
    input_path = Path(input_directory)
    
    if not input_path.exists():
        raise ValueError(f"Input directory '{input_directory}' does not exist")
    
    all_mp3s = list(input_path.rglob("*.mp3"))
    
    # Filter out files that are too small (likely corrupted)
    for mp3_file in all_mp3s:
        try:
            if mp3_file.stat().st_size > 1000:  # At least 1KB
                mp3_files.append(mp3_file)
            else:
                print(f" Skipping tiny file: {mp3_file.name} ({mp3_file.stat().st_size} bytes)")
        except OSError:
            print(f" Cannot access file: {mp3_file}")
    
    return mp3_files

def prepare_file_pairs(input_directory, output_directory, preserve_structure=True):
    """Prepare input-output file pairs for conversion."""
    mp3_files = find_mp3_files(input_directory)
    
    if not mp3_files:
        raise ValueError(f"No valid MP3 files found in '{input_directory}'")
    
    file_pairs = []
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    
    for mp3_file in mp3_files:
        if preserve_structure:
            relative_path = mp3_file.relative_to(input_path)
            wav_file = output_path / relative_path.with_suffix('.wav')
        else:
            wav_file = output_path / mp3_file.with_suffix('.wav').name
        
        wav_file.parent.mkdir(parents=True, exist_ok=True)
        file_pairs.append((str(mp3_file), str(wav_file)))
    
    return file_pairs

def convert_files_parallel(input_directory, output_directory, max_workers=None, preserve_structure=True):
    """Convert MP3 files to WAV using parallel processing."""
    print(" Enhanced MP3 to WAV Converter (Multiple ffmpeg strategies)")
    print("=" * 70)
    
    if max_workers is None:
        max_workers = max(1, int(cpu_count() * 0.5))  # More conservative
    
    print(f"CPU Cores: {cpu_count()}")
    print(f"Worker Processes: {max_workers}")
    print(f"Available RAM: {psutil.virtual_memory().total // (1024**3)} GB")
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    
    # Test a single file first
    if not test_single_file(input_directory):
        print("\n Single file test failed. Check your ffmpeg installation and file format.")
        return
    
    print("-" * 70)
    
    try:
        file_pairs = prepare_file_pairs(input_directory, output_directory, preserve_structure)
        total_files = len(file_pairs)
        print(f"Found {total_files:,} valid MP3 files to convert")
    except Exception as e:
        print(f" Error preparing files: {e}")
        return
    
    if total_files == 0:
        print("No files to convert!")
        return
    
    start_time = time.time()
    successful_conversions = 0
    failed_conversions = 0
    failed_files = []
    
    print("\n Starting parallel conversion...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(convert_mp3_to_wav_robust, file_pair): file_pair for file_pair in file_pairs}
        
        with tqdm.tqdm(total=total_files, desc="Converting", unit="files") as pbar:
            for future in as_completed(future_to_file):
                success, input_file, output_file, error = future.result()
                
                if success:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
                    failed_files.append((Path(input_file).name, error))
                    if failed_conversions <= 10:  # Only show first 10 errors
                        print(f"\n Failed: {Path(input_file).name} - {error}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': successful_conversions,
                    'Failed': failed_conversions
                })
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print(" DETAILED CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Successful conversions: {successful_conversions:,}")
    print(f" Failed conversions: {failed_conversions:,}")
    print(f" Success rate: {(successful_conversions/total_files)*100:.1f}%")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    
    if successful_conversions > 0:
        print(f" Output saved to: {output_directory}")
    
    # Show sample of failed files for debugging
    if failed_files:
        print(f"\n Sample of failed files (first 5):")
        for i, (filename, error) in enumerate(failed_files[:5]):
            print(f"  {i+1}. {filename}: {error}")
        
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

def main():
    # Your existing paths
    INPUT_DIRECTORY = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\stego"
    OUTPUT_DIRECTORY = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\wav stego"
    MAX_WORKERS = 4  # Reduced for stability
    PRESERVE_STRUCTURE = True
    
    ffmpeg_path = r"C:\Users\JHANI\Downloads\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\bin\ffmpeg.exe"
    
    if not os.path.exists(ffmpeg_path):
        print(f" ffmpeg not found at: {ffmpeg_path}")
        return
    
    if not os.path.exists(INPUT_DIRECTORY):
        print(f" Input directory not found: {INPUT_DIRECTORY}")
        return
    
    print(f" ffmpeg found at: {ffmpeg_path}")
    print(f" Input directory found: {INPUT_DIRECTORY}")
    
    try:
        convert_files_parallel(INPUT_DIRECTORY, OUTPUT_DIRECTORY, MAX_WORKERS, PRESERVE_STRUCTURE)
    except KeyboardInterrupt:
        print("\n  Conversion stopped by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
    try:
        import tqdm
        import psutil
    except ImportError as e:
        print(f" Missing required package: {e}")
        print("Install with: pip install tqdm psutil")
        sys.exit(1)
    
    main()
