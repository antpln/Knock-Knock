import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_analysis():
    """
    Plots the latency histogram, smoothed curve, and analysis points.
    """
    try:
        # 1. Read raw latencies for histogram
        latencies = np.loadtxt("latencies.dat", dtype=int)
    except IOError:
        print("Error: latencies.dat not found. Cannot plot histogram.")
        return

    try:
        # 2. Read smoothed histogram data
        smoothed_data = np.loadtxt("smoothed_histogram.dat", delimiter=',')
        smoothed_x = smoothed_data[:, 0]
        smoothed_y = smoothed_data[:, 1]
    except IOError:
        print("Warning: smoothed_histogram.dat not found. Skipping smoothed curve plot.")
        smoothed_x, smoothed_y = None, None

    try:
        # 3. Read analysis points (peaks, threshold)
        analysis_points = {}
        with open("analysis_points.dat", 'r') as f:
            for line in f:
                name, value = line.strip().split(',')
                analysis_points[name] = int(value)
    except IOError:
        print("Warning: analysis_points.dat not found. Skipping analysis markers.")
        analysis_points = {}


    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    min_latency = np.min(latencies) if len(latencies) > 0 else 0
    max_latency = np.max(latencies) if len(latencies) > 0 else 0
    # Create bins with a width of 2 latency values
    bins = np.arange(min_latency, max_latency + 2, 2)
    # Plot the main histogram
    plt.hist(latencies, bins=bins, alpha=0.6, color='gray', label='Latency Distribution', density=True)

    # Plot the smoothed curve if available
    if smoothed_x is not None and smoothed_y is not None:
        # Normalize the smoothed histogram to match the density plot
        if np.sum(smoothed_y) > 0:
            bin_width = smoothed_x[1] - smoothed_x[0] if len(smoothed_x) > 1 else 1
            smoothed_y_normalized = smoothed_y / (np.sum(smoothed_y) * bin_width)
            plt.plot(smoothed_x, smoothed_y_normalized, color='cornflowerblue', linewidth=2, label='Smoothed Curve')

    # Plot the vertical lines for analysis points
    if 'threshold' in analysis_points:
        plt.axvline(x=analysis_points['threshold'], color='red', linestyle='--', linewidth=2, label=f"Auto Threshold ({analysis_points['threshold']})")
    
    if 'main_peak' in analysis_points:
        plt.axvline(x=analysis_points['main_peak'], color='green', linestyle=':', linewidth=2, label=f"Main Peak ({analysis_points['main_peak']})")

    if 'high_peak' in analysis_points:
        plt.axvline(x=analysis_points['high_peak'], color='purple', linestyle=':', linewidth=2, label=f"High-Latency Peak ({analysis_points['high_peak']})")


    plt.title('Latency Analysis for Threshold Detection')
    plt.xlabel('Latency (cycles)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set x-axis limits to focus on the relevant range
    if len(latencies) > 0:
        p1 = np.percentile(latencies, 1)
        p99 = np.percentile(latencies, 99.9)
        plt.xlim(p1 - 20, p99 + 50)

    print("Displaying plot. Close the plot window to continue the C++ application...")
    plt.show()

if __name__ == "__main__":
    plot_analysis()
