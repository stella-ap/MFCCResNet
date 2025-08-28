import os
import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys

# Configuration
FFMPEG_PATH = r"C:\Users\JHANI\Downloads\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\bin\ffmpeg.exe"
INPUT_DIR = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\mp3 converted back"
OUTPUT_DIR = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\mp3_96kbps"

def convert_mp3(args):
    """Convert a single MP3 file to 96kbps"""
    input_file, output_file, file_index, total_files = args
    
    try:
        # Check if FFmpeg exists
        if not os.path.exists(FFMPEG_PATH):
            return f"ERROR: FFmpeg not found at {FFMPEG_PATH}"
        
        # FFmpeg command for high-speed conversion
        cmd = [
            FFMPEG_PATH,
            '-i', str(input_file),
            '-b:a', '96k',
            '-ac', '2',
            '-ar', '44100',
            '-acodec', 'libmp3lame',
            '-q:a', '4',
            '-y',  # Overwrite output files
            str(output_file)
        ]
        
        # Run conversion with minimal output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        if result.returncode != 0:
            return f"ERROR {file_index}/{total_files}: {input_file.name} - {result.stderr[:100]}"
        
        # Progress update
        if file_index % 50 == 0:
            print(f"Progress: {file_index}/{total_files} ({(file_index/total_files)*100:.1f}%)")
        
        return f"SUCCESS {file_index}/{total_files}: {input_file.name}"
        
    except Exception as e:
        return f"EXCEPTION {file_index}/{total_files}: {input_file.name} - {str(e)}"

def get_all_mp3_files(input_dir):
    """Get all MP3 files from input directory"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return []
    
    print(f"Scanning directory: {input_path}")
    mp3_files = list(input_path.rglob("*.mp3"))
    print(f"Found {len(mp3_files)} MP3 files")
    
    return mp3_files

def prepare_conversion_tasks(mp3_files, output_dir):
    """Prepare conversion tasks with all necessary information"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    total_files = len(mp3_files)
    
    for i, mp3_file in enumerate(mp3_files, 1):
        # Preserve directory structure if needed
        relative_path = mp3_file.relative_to(Path(INPUT_DIR))
        output_file = output_path / relative_path
        
        # Create output subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        tasks.append((mp3_file, output_file, i, total_files))
    
    return tasks

def check_prerequisites():
    """Check if all required components are available"""
    print("Checking prerequisites...")
    
    # Check FFmpeg
    if not os.path.exists(FFMPEG_PATH):
        print(f"ERROR: FFmpeg not found at: {FFMPEG_PATH}")
        print("Please check the FFMPEG_PATH in the script")
        return False
    
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return False
    
    # Try to create output directory
    try:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create output directory: {e}")
        return False
    
    print("Prerequisites check passed!")
    return True

def main():
    """Main conversion function"""
    print("=" * 60)
    print("HIGH-PERFORMANCE MP3 CONVERTER - 96 KBPS")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"FFmpeg: {FFMPEG_PATH}")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        input("Press Enter to exit...")
        return
    
    # Get system info
    cpu_count = multiprocessing.cpu_count()
    print(f"System CPU cores: {cpu_count}")
    
    # Allow user to choose number of workers
    max_workers = min(cpu_count, 8)  # Limit to 8 to avoid overwhelming system
    print(f"Using {max_workers} parallel workers")
    print()
    
    # Get all MP3 files
    print("Scanning for MP3 files...")
    mp3_files = get_all_mp3_files(INPUT_DIR)
    
    if not mp3_files:
        print("No MP3 files found!")
        input("Press Enter to exit...")
        return
    
    # Prepare conversion tasks
    print("Preparing conversion tasks...")
    tasks = prepare_conversion_tasks(mp3_files, OUTPUT_DIR)
    total_files = len(tasks)
    
    print(f"Ready to convert {total_files} files")
    
    # Confirm before starting
    response = input("Start conversion? (y/N): ").strip().lower()
    if response != 'y':
        print("Conversion cancelled.")
        return
    
    # Start timing
    start_time = time.time()
    print(f"\nStarting conversion with {max_workers} parallel processes...")
    print("=" * 60)
    
    # Track results
    successful = 0
    failed = 0
    
    # Process files in parallel
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(convert_mp3, task) for task in tasks]
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.startswith("SUCCESS"):
                        successful += 1
                    else:
                        failed += 1
                        print(result)  # Print errors
                        
                except Exception as e:
                    failed += 1
                    print(f"Task failed with exception: {e}")
    
    except KeyboardInterrupt:
        print("\nConversion interrupted by user!")
        return
    
    # Final statistics
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"Total files processed: {successful + failed}")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/(successful + failed)*100):.1f}%")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    if total_files > 0:
        print(f"Average time per file: {(duration/total_files):.3f} seconds")
        print(f"Files per second: {(total_files/duration):.2f}")
    
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Keep window open
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    # Ensure proper multiprocessing on Windows
    multiprocessing.freeze_support()
    main()