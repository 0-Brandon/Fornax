#!/usr/bin/env python3
"""
Generate measured data for Fornax benchmark visualizations.
This orchestrates the 'fornax' binary to generate real data for the 3-phase demo:
1. Uncontrolled (High Load)
2. 50% Duty Cycle
3. Hysteresis Control

It stitches the results into results/frequency_power.csv
"""

import csv
import subprocess
import shutil
import sys
import os
from pathlib import Path

def run_fornax(duration, flags, output_csv):
    """Run fornax binary and capture output to CSV"""
    cmd = [
        "./build/fornax",
        "--duration", str(duration),
        "--monitor-output", str(output_csv),
        "--trials", "1",
        "--warmup", "0"
    ] + flags
    
    print(f"Running: {' '.join(cmd)}")
    
    # Check if build exists
    if not Path("./build/fornax").exists():
        print("Error: ./build/fornax not found. Please build the project first.")
        sys.exit(1)
        
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running fornax: {e}")
        sys.exit(1)

def merge_csvs(files, output_file, time_offsets):
    """Merge multiple CSVs into one with continuous timestamps"""
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['timestamp_s', 'frequency_mhz', 'power_w', 'throttled'])
        
        current_time_offset = 0.0
        
        for i, (fname, duration) in enumerate(zip(files, time_offsets)):
            print(f"Merging {fname} (offset {current_time_offset}s)...")
            
            with open(fname, 'r') as infile:
                reader = csv.DictReader(infile)
                
                # Verify header
                if reader.fieldnames != ['timestamp_s', 'frequency_mhz', 'power_w', 'throttled']:
                     print(f"Warning: Unexpected header in {fname}: {reader.fieldnames}")

                rows = list(reader)
                if not rows:
                    continue
                    
                # Adjust timestamps
                # The binary outputs time relative to start of run (0.0)
                # We simply add current_time_offset
                
                for row in rows:
                    t = float(row['timestamp_s']) + current_time_offset
                    writer.writerow([
                        f"{t:.4f}", 
                        row['frequency_mhz'], 
                        row['power_w'], 
                        row['throttled']
                    ])
            
            current_time_offset += duration

def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Define phases
    # Phase 1: Uncontrolled (Thermal limit test)
    # Using defaults which uses Schmitt trigger, but we'll set high thresholds to effectively disable it
    # or just run standard mode. Note: 'Uncontrolled' in plot usually means 100% duty.
    # We can approximate 'Uncontrolled' by setting duty cycle 1.0 (though that's manual mode)
    # Or setting high threshold very high.
    # Phase 1: Uncontrolled (High Load)
    # We want 0% throttling (100% work), so we set duty-cycle to 0.0.
    # Note: In FORNAX context, "duty cycle" refers to the THROTTLE duty cycle.
    # So 0.0 means "throttle for 0% of period", i.e., run full speed.
    # (Duty Cycle = Throttle Duration / Total Period)
    phase1_csv = results_dir / "phase1.csv"
    phase1_duration = 3
    run_fornax(phase1_duration, ["--duty-cycle", "0.0"], phase1_csv)
    
    # Phase 2: 50% Duty Cycle
    phase2_csv = results_dir / "phase2.csv"
    phase2_duration = 3
    run_fornax(phase2_duration, ["--duty-cycle", "0.5"], phase2_csv)
    
    # Phase 3: Hysteresis (Default control)
    phase3_csv = results_dir / "phase3.csv"
    phase3_duration = 4
    # Standard hysteresis (defaults: high=50W, low=35W)
    run_fornax(phase3_duration, [], phase3_csv)

    # Phase 4: Latency Test (New)
    latency_csv = results_dir / "latency_data.csv"
    print("Running Latency Test...")
    run_fornax(0, ["--latency-test", "--latency-output", str(latency_csv), "--trials", "100"], latency_csv)
    
    # Merge
    final_csv = results_dir / "frequency_power.csv"
    merge_csvs(
        [phase1_csv, phase2_csv, phase3_csv],
        final_csv,
        [phase1_duration, phase2_duration, phase3_duration]
    )
    
    print(f"Generated {final_csv}")
    
    # Cleanup temps
    for f in [phase1_csv, phase2_csv, phase3_csv]:
        if f.exists():
            f.unlink()

if __name__ == "__main__":
    # Ensure backup of synthetic data if it exists and hasn't been backed up
    synthetic_script = Path("scripts/generate_data_synthetic.py")
    if not synthetic_script.exists():
        if Path("scripts/generate_data.py").exists():
             shutil.copy("scripts/generate_data.py", synthetic_script)
             print(f"Backed up original script to {synthetic_script}")
    
    main()
