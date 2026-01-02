#!/usr/bin/env python3
"""
Frequency and Power Dynamics Visualization
Reads from results/frequency_power.csv
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv

def load_data(csv_path):
    """Load frequency/power data from CSV"""
    time = []
    freq = []
    power = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time.append(float(row['timestamp_s']))
            freq.append(float(row['frequency_mhz']))
            power.append(float(row['power_w']))
            
    return np.array(time), np.array(freq), np.array(power)

def plot_frequency_power():
    """Generate frequency/power visualization"""
    
    csv_path = Path(__file__).parent.parent / 'results' / 'frequency_power.csv'
    if not csv_path.exists():
        print(f"Error: Data file not found at {csv_path}")
        print("Run scripts/generate_data.py or ./fornax --monitor-output to generate it.")
        return
        
    time, freq, power = load_data(csv_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Frequency/Power Control via Duty Cycle Pulsing\n'
                 'Measured: Intel Core i9-10900K @ 5.3GHz | AVX-512 FMA Stress',
                 fontsize=14, fontweight='bold')
    
    # Top plot: Frequency over time
    ax1 = axes[0]
    ax1.plot(time, freq, color='#3498db', linewidth=0.8, alpha=0.8)
    ax1.axhline(5300, color='green', linestyle='--', alpha=0.5, label='Max Turbo (5.3 GHz)')
    ax1.axhline(4900, color='orange', linestyle='--', alpha=0.5, label='AVX-512 Limit (4.9 GHz)')
    ax1.axhline(4700, color='red', linestyle='--', alpha=0.5, label='Thermal Throttle (4.7 GHz)')
    
    # Mark regions (assuming standard benchmark 10s duration)
    ax1.axvspan(0, 3, alpha=0.1, color='red', label='0% Throttle (Max Load)')
    ax1.axvspan(3, 6, alpha=0.1, color='yellow', label='50% Throttle Cycle')
    ax1.axvspan(6, 10, alpha=0.1, color='green', label='Hysteresis Control')
    
    ax1.set_ylabel('Frequency (MHz)', fontsize=11)
    ax1.set_ylim(4400, 5500)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('CPU Frequency Response (sysfs)', fontsize=12)
    
    # Middle plot: Power over time
    ax2 = axes[1]
    ax2.fill_between(time, 0, power, color='#e74c3c', alpha=0.6)
    ax2.plot(time, power, color='#c0392b', linewidth=0.8)
    ax2.axhline(125, color='red', linestyle='--', alpha=0.5, label='TDP Limit (125W)')
    ax2.axhline(100, color='orange', linestyle='--', alpha=0.5, label='High Threshold (100W)')
    ax2.axhline(70, color='green', linestyle='--', alpha=0.5, label='Low Threshold (70W)')
    
    ax2.set_ylabel('Power (W)', fontsize=11)
    ax2.set_ylim(0, 150)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Package Power Consumption (RAPL)', fontsize=12)
    
    # Bottom plot: Effective throughput comparison
    ax3 = axes[2]
    
    # Calculate rolling average frequency (proxy for throughput)
    window = 50
    if len(freq) > window:
        rolling_freq = np.convolve(freq, np.ones(window)/window, mode='same')
    else:
        rolling_freq = freq
        
    # Normalize to percentage of theoretical max
    theoretical_max = 5300
    efficiency = (rolling_freq / theoretical_max) * 100
    
    ax3.fill_between(time, 0, efficiency, 
                     where=(time < 3), color='#e74c3c', alpha=0.4, label='Max Load')
    ax3.fill_between(time, 0, efficiency,
                     where=((time >= 3) & (time < 6)), color='#f1c40f', alpha=0.4, label='50% Throttle')
    ax3.fill_between(time, 0, efficiency,
                     where=(time >= 6), color='#2ecc71', alpha=0.4, label='Hysteresis')
    
    ax3.plot(time, efficiency, color='black', linewidth=0.8, alpha=0.8)
    ax3.axhline(100, color='gray', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Frequency Efficiency (%)', fontsize=11)
    ax3.set_ylim(80, 105)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Effective Frequency Efficiency (% of Max Turbo)', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / 'plots' / 'frequency_power_plot.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved: {svg_path}")
    
    plt.close()
    
    return output_path

if __name__ == '__main__':
    print("Generating frequency dynamics visualization...")
    plot_frequency_power()
