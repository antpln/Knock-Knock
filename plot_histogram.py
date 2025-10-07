#!/usr/bin/env python3
"""
Latency Histogram Plotter for Manual Threshold Analysis

This script visualizes latency distributions and helps identify the optimal
threshold for separating row buffer hits from conflicts.

Usage:
    python3 plot_histogram.py [--interactive] [--save output.png]
    
Options:
    --interactive    Enable interactive mode with zoom and click-to-select threshold
    --save FILE      Save the plot to a file instead of displaying
"""

import sys
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    print(f"Optional plotting skipped: {exc}")
    sys.exit(0)

def load_data():
    """Load all available data files."""
    data = {}
    
    # Load raw latencies
    try:
        data['latencies'] = np.loadtxt("latencies.dat", dtype=int)
        print(f"Loaded {len(data['latencies'])} latency samples")
    except IOError:
        print("Error: latencies.dat not found. Run the analysis first.")
        sys.exit(1)
    
    # Load raw histogram
    try:
        hist_data = np.loadtxt("raw_histogram.dat", delimiter=',', skiprows=1)
        data['raw_hist_x'] = hist_data[:, 0]
        data['raw_hist_y'] = hist_data[:, 1]
        print(f"Loaded raw histogram with {len(data['raw_hist_x'])} bins")
    except IOError:
        print("Warning: raw_histogram.dat not found")
        data['raw_hist_x'] = None
        data['raw_hist_y'] = None
    
    # Load smoothed histogram
    try:
        smoothed_data = np.loadtxt("smoothed_histogram.dat", delimiter=',', skiprows=1)
        data['smoothed_x'] = smoothed_data[:, 0]
        data['smoothed_y'] = smoothed_data[:, 1]
        print(f"Loaded smoothed histogram")
    except IOError:
        print("Warning: smoothed_histogram.dat not found")
        data['smoothed_x'] = None
        data['smoothed_y'] = None
    
    # Load analysis points
    data['analysis_points'] = {}
    try:
        with open("analysis_points.dat", 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue
                name, value = parts
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                if name in {"threshold", "main_peak", "high_peak"}:
                    data['analysis_points'][name] = int(round(numeric))
                else:
                    data['analysis_points'][name] = numeric
        print(f"Loaded analysis points: {list(data['analysis_points'].keys())}")
    except IOError:
        print("Warning: analysis_points.dat not found")
    
    return data

def print_statistics(latencies, threshold=None):
    """Print useful statistics about the latency distribution."""
    print("\n" + "="*60)
    print("LATENCY STATISTICS")
    print("="*60)
    print(f"Total samples: {len(latencies)}")
    print(f"Min latency: {np.min(latencies)}")
    print(f"Max latency: {np.max(latencies)}")
    print(f"Mean: {np.mean(latencies):.1f}")
    print(f"Median: {np.median(latencies):.1f}")
    print(f"Std dev: {np.std(latencies):.1f}")
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        print(f"  P{p}: {np.percentile(latencies, p):.0f}")
    
    if threshold is not None:
        below = np.sum(latencies < threshold)
        above = np.sum(latencies >= threshold)
        print(f"\nWith threshold = {threshold}:")
        print(f"  Below threshold (hits): {below} ({100*below/len(latencies):.2f}%)")
        print(f"  Above threshold (conflicts): {above} ({100*above/len(latencies):.2f}%)")
    print("="*60 + "\n")

def plot_analysis(data, interactive=False, save_file=None):
    """
    Plot the latency histogram with analysis markers.
    """
    latencies = data['latencies']
    analysis_points = data['analysis_points']
    
    # Print statistics
    threshold = analysis_points.get('threshold')
    print_statistics(latencies, threshold)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Calculate bins
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    bins = np.arange(min_lat, max_lat + 2, 1)
    
    # Main plot - histogram
    ax1.hist(latencies, bins=bins, alpha=0.5, color='skyblue', 
             label='Raw Distribution', density=False, edgecolor='black', linewidth=0.5)
    
    # Plot smoothed curve if available
    if data['smoothed_x'] is not None and data['smoothed_y'] is not None:
        # Scale smoothed curve to match histogram scale
        hist_max = np.max(np.histogram(latencies, bins=bins)[0])
        smooth_max = np.max(data['smoothed_y'])
        if smooth_max > 0:
            scale_factor = hist_max / smooth_max
            ax1.plot(data['smoothed_x'], data['smoothed_y'] * scale_factor, 
                    color='darkblue', linewidth=2.5, label='Smoothed Curve', alpha=0.8)
    
    # Mark analysis points
    if 'main_peak' in analysis_points:
        peak = analysis_points['main_peak']
        ax1.axvline(x=peak, color='green', linestyle=':', linewidth=2.5,
                   label=f'Main Peak ({peak})', alpha=0.8)
    
    if 'high_peak' in analysis_points:
        high_peak = analysis_points['high_peak']
        ax1.axvline(x=high_peak, color='purple', linestyle=':', linewidth=2.5,
                   label=f'Conflict Peak ({high_peak})', alpha=0.8)
    
    if 'threshold' in analysis_points:
        thresh = analysis_points['threshold']
        ax1.axvline(x=thresh, color='red', linestyle='--', linewidth=3,
                   label=f'Auto Threshold ({thresh})', alpha=0.9, zorder=10)
        ax1.fill_betweenx([0, ax1.get_ylim()[1]], 0, thresh, 
                          alpha=0.1, color='green', label='Hits (below threshold)')
        ax1.fill_betweenx([0, ax1.get_ylim()[1]], thresh, max_lat, 
                          alpha=0.1, color='red', label='Conflicts (above threshold)')
    
    # Add confidence score
    if 'confidence' in analysis_points:
        confidence = analysis_points['confidence']
        status = "GOOD" if confidence > 0.5 else "POOR"
        color = 'darkgreen' if confidence > 0.5 else 'darkred'
        ax1.text(0.02, 0.98, f"Confidence: {confidence:.3f} ({status})",
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                color=color, weight='bold')
    
    ax1.set_xlabel('Latency (cycles)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Memory Access Latency Distribution - Threshold Analysis', 
                 fontsize=14, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Focus on relevant range
    p1 = np.percentile(latencies, 0.5)
    p99 = np.percentile(latencies, 99.9)
    ax1.set_xlim(p1 - 10, p99 + 30)
    
    # Lower plot - cumulative distribution
    sorted_lat = np.sort(latencies)
    cumulative = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat) * 100
    ax2.plot(sorted_lat, cumulative, color='navy', linewidth=2, label='Cumulative %')
    
    if 'threshold' in analysis_points:
        thresh = analysis_points['threshold']
        ax2.axvline(x=thresh, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({thresh})', alpha=0.8)
        # Mark percentage at threshold
        pct_below = np.sum(sorted_lat < thresh) / len(sorted_lat) * 100
        ax2.plot(thresh, pct_below, 'ro', markersize=10)
        ax2.text(thresh, pct_below + 3, f'{pct_below:.1f}%', 
                ha='center', fontsize=10, weight='bold')
    
    ax2.set_xlabel('Latency (cycles)', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(loc='lower right')
    ax2.set_xlim(p1 - 10, p99 + 30)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_file}")
    else:
        print("\n" + "="*60)
        print("MANUAL THRESHOLD GUIDANCE")
        print("="*60)
        print("Look for the 'left foot' where the high-latency bump begins.")
        print("The threshold should cleanly separate two modes:")
        print("  - Low-latency mode (row buffer hits)")
        print("  - High-latency mode (row buffer conflicts)")
        print("\nIf auto-detection failed, choose a value between the peaks")
        print("where the histogram valley is deepest.")
        print("\nTo use your manual threshold:")
        print("  sudo ./main --full-analysis --threshold <value>")
        print("="*60)
        
        if interactive:
            print("\nClick on the plot to select a threshold value...")
            
            def onclick(event):
                if event.inaxes == ax1 and event.xdata is not None:
                    manual_thresh = int(event.xdata)
                    print(f"\nManual threshold selected: {manual_thresh}")
                    print_statistics(latencies, manual_thresh)
                    print(f"\nTo use this threshold:")
                    print(f"  sudo ./main --full-analysis --threshold {manual_thresh}")
            
            fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plot latency histogram for threshold analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive threshold selection')
    parser.add_argument('--save', metavar='FILE',
                       help='Save plot to file instead of displaying')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data()
    
    # Create plot
    plot_analysis(data, interactive=args.interactive, save_file=args.save)

if __name__ == "__main__":
    main()
