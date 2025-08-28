#!/usr/bin/env python3
"""
Ultra-Clean MP3 Batch Converter
Optimized for Ryzen 5 7600X (6C/12T) + 16GB DDR5
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Thread, Lock
import psutil
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Disable all logging
logging.disable(logging.CRITICAL)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    ffmpeg_path: str = r"C:\Users\JHANI\Downloads\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\ffmpeg-2025-08-07-git-fa458c7243-essentials_build\bin\ffmpeg.exe"
    input_dir: str = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\mp3 converted back"
    output_dir_96k: str = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\mp3_96kbps"
    output_dir_256k: str = r"C:\RESEARCH 2\Paper 1\archive (1)\Dataset\mp3_256kbps"
    max_workers: int = 4  # Further reduced for stability
    chunk_size: int = 20  # Smaller batches
    timeout_per_file: int = 180

config = Config()

# =============================================================================
# UTILITIES
# =============================================================================

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str, width: int = 60):
    """Print clean header"""
    clear_screen()
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│{title.center(width - 2)}│")
    print("└" + "─" * (width - 2) + "┘")
    print()

def print_box(content: List[str], width: int = 60):
    """Print content in a clean box"""
    print("┌" + "─" * (width - 2) + "┐")
    for line in content:
        line_len = len(line)
        if line_len > width - 4:
            line = line[:width-7] + "..."
        padding = max(0, width - 4 - len(line))
        print(f"│ {line}{' ' * padding} │")
    print("└" + "─" * (width - 2) + "┘")

def print_progress_bar(current: int, total: int, prefix: str = "", width: int = 40):
    """Print clean progress bar"""
    if total == 0:
        return
        
    percent = min(1.0, current / total)
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    
    print(f"\r{prefix} [{bar}] {current:,}/{total:,} ({percent*100:.1f}%)", end="", flush=True)

class PerformanceMonitor:
    def __init__(self):
        self.start_time = 0
        self.processed = 0
        self.failed = 0
        self.total = 0
        self.lock = Lock()
        self.active = False
        
    def start(self, total_files: int):
        with self.lock:
            self.start_time = time.perf_counter()
            self.total = total_files
            self.processed = 0
            self.failed = 0
            self.active = True
        
    def update(self, success: bool = True):
        with self.lock:
            if self.active:
                if success:
                    self.processed += 1
                else:
                    self.failed += 1
    
    def stop(self):
        with self.lock:
            self.active = False
    
    def get_stats(self) -> dict:
        with self.lock:
            elapsed = max(1, time.perf_counter() - self.start_time)
            completed = self.processed + self.failed
            
            return {
                'processed': self.processed,
                'failed': self.failed,
                'total': self.total,
                'completed': completed,
                'elapsed': elapsed,
                'rate': completed / elapsed,
                'eta': max(0, (self.total - completed) / (completed / elapsed)) if completed > 0 else 0,
                'progress_pct': (completed / self.total * 100) if self.total > 0 else 0,
                'active': self.active
            }

monitor = PerformanceMonitor()

# =============================================================================
# CONVERSION ENGINE
# =============================================================================

def build_ffmpeg_command(input_file: Path, output_file: Path, bitrate: str) -> List[str]:
    """Build optimized FFmpeg command"""
    cmd = [
        config.ffmpeg_path,
        '-threads', '1',
        '-i', str(input_file),
        '-c:a', 'libmp3lame',
        '-b:a', bitrate,
        '-ac', '2',
        '-ar', '44100',
        '-map_metadata', '0',
        '-f', 'mp3',
        '-y',
        str(output_file)
    ]
    return cmd

def convert_single_file(args: Tuple[Path, Path, str, int]) -> dict:
    """Convert single file with minimal overhead"""
    input_file, output_file, bitrate, file_index = args
    
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if exists and is newer
        if output_file.exists():
            try:
                if output_file.stat().st_mtime > input_file.stat().st_mtime:
                    monitor.update(True)
                    return {'success': True, 'skipped': True, 'file': input_file.name}
            except (OSError, IOError):
                pass  # Continue with conversion if stat fails
        
        # Build command
        cmd = build_ffmpeg_command(input_file, output_file, bitrate)
        
        # Execute with proper error handling
        try:
            if os.name == 'nt':
                # Windows-specific flags
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    text=True
                )
            else:
                # Unix-like systems
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )
        except FileNotFoundError:
            monitor.update(False)
            return {'success': False, 'error': 'ffmpeg_not_found', 'file': input_file.name}
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=config.timeout_per_file)
            success = process.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0
            monitor.update(success)
            
            if success:
                return {'success': True, 'file': input_file.name}
            else:
                return {'success': False, 'error': 'conversion_failed', 'file': input_file.name, 'stderr': stderr[:200]}
                
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=5)
            except:
                pass
            monitor.update(False)
            return {'success': False, 'error': 'timeout', 'file': input_file.name}
            
    except Exception as e:
        monitor.update(False)
        return {'success': False, 'error': f'exception: {str(e)[:100]}', 'file': input_file.name}

def scan_mp3_files(input_dir: Path) -> List[Path]:
    """Scan for MP3 files"""
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    print("Scanning for MP3 files...")
    mp3_files = []
    
    try:
        # Use iterative approach for better error handling
        for item in input_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() == '.mp3':
                mp3_files.append(item)
    except Exception as e:
        print(f"Warning during scan: {e}")
    
    if not mp3_files:
        raise FileNotFoundError("No MP3 files found in the specified directory")
    
    return sorted(mp3_files)

def prepare_tasks(mp3_files: List[Path], output_dir: Path, bitrate: str) -> List[Tuple]:
    """Prepare conversion tasks"""
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    base_input = Path(config.input_dir)
    
    for i, mp3_file in enumerate(mp3_files, 1):
        try:
            # Try to get relative path
            try:
                relative_path = mp3_file.relative_to(base_input)
            except ValueError:
                # If relative path fails, use just the filename
                relative_path = mp3_file.name
            
            output_file = output_dir / relative_path
            tasks.append((mp3_file, output_file, bitrate, i))
        except Exception as e:
            print(f"Warning: Could not prepare task for {mp3_file}: {e}")
            continue
    
    return tasks

def estimate_storage(file_count: int, bitrate: str) -> float:
    """Estimate storage requirements in GB"""
    try:
        bitrate_value = int(bitrate.rstrip('k'))
        # Assume 4-minute average duration
        size_gb = file_count * 4 * 60 * (bitrate_value / 8) / (1024 ** 3)
        return size_gb
    except:
        return 0.0

def progress_reporter():
    """Clean progress reporting"""
    last_completed = 0
    start_time = time.time()
    
    while True:
        stats = monitor.get_stats()
        
        # Always show initial progress
        if stats['completed'] >= 0:
            if stats['completed'] > 0 and stats['elapsed'] > 1:
                eta_min = stats['eta'] / 60 if stats['eta'] < 3600 else stats['eta'] / 3600
                eta_unit = "min" if stats['eta'] < 3600 else "hr"
                
                print_progress_bar(
                    stats['completed'], 
                    stats['total'], 
                    f"Converting ({stats['rate']:.1f}/sec, ETA: {eta_min:.1f}{eta_unit}):"
                )
            else:
                print_progress_bar(
                    stats['completed'], 
                    stats['total'], 
                    "Converting (starting...):"
                )
            last_completed = stats['completed']
        
        # Exit conditions
        if not stats['active'] or stats['completed'] >= stats['total']:
            break
        
        # Timeout safety (exit after 3 hours)
        if time.time() - start_time > 10800:
            break
            
        time.sleep(1)  # Update every second for better responsiveness

def convert_batch(tasks: List[Tuple], bitrate: str) -> dict:
    """Main conversion with clean output"""
    if not tasks:
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    print(f"\nStarting conversion to {bitrate}...")
    print(f"Workers: {config.max_workers} | Files: {len(tasks):,}")
    print("Initializing workers...")
    
    monitor.start(len(tasks))
    
    # Start progress reporter
    progress_thread = Thread(target=progress_reporter, daemon=True)
    progress_thread.start()
    
    results = {'success': 0, 'failed': 0, 'skipped': 0}
    failed_files = []
    completed_count = 0
    
    try:
        # Use spawn method for better Windows compatibility
        ctx = mp.get_context('spawn')
        
        print("Workers started, processing files...")
        
        with ProcessPoolExecutor(max_workers=config.max_workers, mp_context=ctx) as executor:
            # Submit tasks in smaller batches to avoid hanging
            batch_size = min(config.chunk_size, len(tasks))
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1} ({len(batch_tasks)} files)...")
                
                # Submit batch
                futures = [executor.submit(convert_single_file, task) for task in batch_tasks]
                
                # Collect results from this batch
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=config.timeout_per_file + 30)
                        completed_count += 1
                        
                        if result['success']:
                            if result.get('skipped'):
                                results['skipped'] += 1
                            else:
                                results['success'] += 1
                        else:
                            results['failed'] += 1
                            failed_files.append(result)
                            
                        # Show progress every 10 files
                        if completed_count % 10 == 0:
                            print(f"Processed {completed_count}/{len(tasks)} files...")
                            
                    except Exception as e:
                        completed_count += 1
                        results['failed'] += 1
                        print(f"Task failed with exception: {str(e)[:100]}")
                        monitor.update(False)
                
                # Small delay between batches
                if i + batch_size < len(tasks):
                    time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(f"\n\nConversion stopped by user!")
        monitor.stop()
        return results
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        monitor.stop()
        return results
    
    monitor.stop()
    
    # Wait for progress thread to finish
    progress_thread.join(timeout=5)
    print()  # New line after progress bar
    
    # Show some failed files if any
    if failed_files and len(failed_files) <= 5:
        print(f"\nSome failed files:")
        for fail in failed_files[:5]:
            print(f"  • {fail['file']}: {fail.get('error', 'unknown error')}")
    elif len(failed_files) > 5:
        print(f"\n{len(failed_files)} files failed. First 5:")
        for fail in failed_files[:5]:
            print(f"  • {fail['file']}: {fail.get('error', 'unknown error')}")
    
    return results

# =============================================================================
# USER INTERFACE
# =============================================================================

def show_system_info():
    """Display clean system information"""
    print_header("SYSTEM INFORMATION")
    
    cpu_count = mp.cpu_count()
    try:
        memory = psutil.virtual_memory()
        memory_info = f"{memory.total/(1024**3):.1f} GB ({memory.percent:.1f}% used)"
    except:
        memory_info = "Unable to get memory info"
    
    info = [
        f"CPU Threads: {cpu_count}",
        f"Memory: {memory_info}",
        f"Workers: {config.max_workers}",
        f"Chunk Size: {config.chunk_size}",
        f"FFmpeg Path: {config.ffmpeg_path}",
        f"Input Dir: {config.input_dir}",
        f"Output 96k: {config.output_dir_96k}",
        f"Output 256k: {config.output_dir_256k}"
    ]
    
    print_box(info, 80)
    input("\nPress Enter to continue...")

def show_storage_estimate(file_count: int, bitrate: str):
    """Show storage estimation"""
    size_gb = estimate_storage(file_count, bitrate)
    
    estimate_info = [
        f"Files to convert: {file_count:,}",
        f"Target bitrate: {bitrate}",
        f"Estimated output size: {size_gb:.1f} GB",
        f"Estimated time: {file_count/(config.max_workers*2):.0f} minutes"
    ]
    
    print_box(estimate_info, 50)

def show_final_results(results: dict, elapsed_time: float):
    """Show clean final results"""
    print_header("CONVERSION COMPLETE")
    
    total = results['success'] + results['failed'] + results['skipped']
    success_rate = (results['success'] / total * 100) if total > 0 else 0
    
    final_info = [
        f"Total Processed: {total:,}",
        f"Successfully Converted: {results['success']:,}",
        f"Skipped (existing): {results['skipped']:,}",
        f"Failed: {results['failed']:,}",
        f"Success Rate: {success_rate:.1f}%",
        f"Total Time: {elapsed_time/60:.1f} minutes",
        f"Average Speed: {total/elapsed_time:.1f} files/second" if elapsed_time > 0 else "Average Speed: N/A"
    ]
    
    print_box(final_info)
    input("\nPress Enter to continue...")

def main_menu():
    """Clean main menu"""
    while True:
        print_header("MP3 BATCH CONVERTER")
        
        menu_options = [
            "1. Convert to 96 kbps  (High Compression)",
            "2. Convert to 256 kbps (High Quality)",
            "3. System Information",
            "4. Exit"
        ]
        
        print_box(menu_options)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice in ['1', '2']:
            try:
                bitrate = '96k' if choice == '1' else '256k'
                output_dir = Path(config.output_dir_96k if choice == '1' else config.output_dir_256k)
                
                # Scan files
                print_header("SCANNING FILES")
                try:
                    mp3_files = scan_mp3_files(Path(config.input_dir))
                    print(f"Found {len(mp3_files):,} MP3 files")
                except Exception as e:
                    print(f"Error scanning files: {e}")
                    input("Press Enter to continue...")
                    continue
                
                # Show estimate
                print_header("STORAGE ESTIMATE")
                show_storage_estimate(len(mp3_files), bitrate)
                
                # Confirm
                confirm = input(f"\nProceed with conversion to {bitrate}? (y/N): ").lower()
                if confirm != 'y':
                    continue
                
                # Convert
                print_header("CONVERTING")
                start_time = time.perf_counter()
                
                try:
                    tasks = prepare_tasks(mp3_files, output_dir, bitrate)
                    if not tasks:
                        print("No tasks to process!")
                        input("Press Enter to continue...")
                        continue
                        
                    results = convert_batch(tasks, bitrate)
                    elapsed = time.perf_counter() - start_time
                    
                    # Show results
                    show_final_results(results, elapsed)
                    
                except Exception as e:
                    print(f"Error during conversion: {e}")
                    input("Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user!")
                input("Press Enter to continue...")
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                input("Press Enter to continue...")
        
        elif choice == '3':
            show_system_info()
        
        elif choice == '4':
            print_header("GOODBYE")
            print("Thank you for using MP3 Batch Converter!")
            break
        
        else:
            print("\nInvalid choice! Please select 1-4.")
            time.sleep(1)

def check_requirements() -> bool:
    """Check requirements with clean output"""
    print_header("CHECKING REQUIREMENTS")
    
    checks = []
    all_good = True
    
    # FFmpeg check
    ffmpeg_path = Path(config.ffmpeg_path)
    if ffmpeg_path.exists():
        checks.append("✓ FFmpeg found")
    else:
        checks.append("✗ FFmpeg not found")
        checks.append(f"  Path: {config.ffmpeg_path}")
        all_good = False
    
    # Input directory
    input_path = Path(config.input_dir)
    if input_path.exists():
        checks.append("✓ Input directory found")
        try:
            # Quick test for MP3 files
            mp3_count = len(list(input_path.rglob("*.mp3")))
            checks.append(f"  MP3 files found: {mp3_count}")
        except Exception:
            checks.append("  Could not count MP3 files")
    else:
        checks.append("✗ Input directory not found")
        checks.append(f"  Path: {config.input_dir}")
        all_good = False
    
    # Output directories
    for name, path_str in [("96k output", config.output_dir_96k), ("256k output", config.output_dir_256k)]:
        try:
            Path(path_str).mkdir(parents=True, exist_ok=True)
            checks.append(f"✓ {name} directory ready")
        except Exception as e:
            checks.append(f"✗ Cannot create {name} directory")
            checks.append(f"  Error: {str(e)[:50]}")
            all_good = False
    
    if all_good:
        checks.append("")
        checks.append("✓ All requirements satisfied")
    
    print_box(checks, 80)
    
    if not all_good:
        print("\nPlease fix the above issues before continuing.")
    
    time.sleep(2)
    return all_good

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Clean main entry point"""
    
    # Set high priority (Windows only)
    if os.name == 'nt':
        try:
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except Exception:
            pass  # Ignore if setting priority fails
    
    # Check requirements
    if not check_requirements():
        input("Press Enter to exit...")
        return
    
    # Main menu
    try:
        main_menu()
    except KeyboardInterrupt:
        clear_screen()
        print("Program interrupted by user!")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()