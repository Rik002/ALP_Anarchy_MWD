import numpy as np
import matplotlib.pyplot as plt
import configparser

def load_config(config_file):
    def clean_value(value):
        value = value.strip()
        if ' ' in value:
            value = value.split()[0]
        if value.endswith(':'):
            value = value[:-1]
        return value

    config = configparser.ConfigParser()
    config.read(config_file)
    return {
        'b_N1': float(clean_value(config['Simulation']['b_N1'])),
        'plot_dir': clean_value(config['Directories']['plot_dir']),
        'data_dir': clean_value(config['Directories']['data_dir'])
    }

def read_bands_file(filename):
    """Read bands data from TXT file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    row = [float(x) for x in parts]
                    data.append(row)
                except ValueError:
                    continue
    return np.array(data)

def plot_results(config_file):
    config = load_config(config_file)
    bands_file = f"{config['data_dir']}bands.txt"

    print(f"Reading bands from {bands_file}...")
    data = read_bands_file(bands_file)

    if len(data) == 0:
        print("❌ No data found in bands file!")
        return

    N_range, lower_90, upper_90, lower_third, upper_third = data.T
    print(f"✓ Read {len(N_range)} data points")

    # Scale data by 10^24 to match paper's units [10^-24 GeV^-1]
    scale_factor = 1e24
    lower_90_scaled = lower_90 * scale_factor
    upper_90_scaled = upper_90 * scale_factor
    lower_third_scaled = lower_third * scale_factor
    upper_third_scaled = upper_third * scale_factor
    b_N1_scaled = config['b_N1'] * scale_factor

    plt.figure(figsize=(12, 10))

    # Plot bands with bold borders
    plt.fill_between(N_range, lower_90_scaled, upper_90_scaled,
                     color='red', alpha=0.3, label='Central 90%',
                     edgecolor='red', linewidth=2)
    plt.fill_between(N_range, lower_third_scaled, upper_third_scaled,
                     color='blue', alpha=0.5, label='Central Third',
                     edgecolor='blue', linewidth=2)

    # Single ALP bound line
    #plt.axhline(b_N1_scaled, color='black', linestyle='-',label='Single ALP Bound', linewidth=0.5)

    # Axis settings
    plt.xlabel('Number of ALPs (N)', fontsize=12)
    plt.ylabel(r'$|g^\gamma g^e|$ [$10^{-24}$ GeV$^{-1}$]', fontsize=12)  # Updated label
    plt.yscale('log')

    # X-axis from 1 to 23 as requested
    plt.xlim(1, 21)
    plt.xticks(range(0, 21, 3))  # Even numbers from 2 to 22

    # Y-axis limits - adjust based on scaled data
    y_min = np.nanmin([np.nanmin(lower_90_scaled), np.nanmin(lower_third_scaled)]) * 0.5
    y_max = np.nanmax([np.nanmax(upper_90_scaled), np.nanmax(upper_third_scaled)]) * 2.0
    plt.ylim(y_min, y_max)
    #plt.ylim(1, 1e3)

    plt.title(r'Bounds on $|g^\gamma g^e|$ for Multi-ALP Scenario', fontsize=14)
    plt.legend(fontsize=11)
    #plt.grid(True, which="both", ls="--", alpha=0.3)

    # Tighter layout
    plt.tight_layout()

    plot_file = f"{config['plot_dir']}figure_4.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_file}")

    # Also show some statistics about the scaled data
    print(f"Data range (scaled): [{y_min:.1f}, {y_max:.1f}] × 10^-24 GeV^-1")
    print(f"Single ALP bound (scaled): {b_N1_scaled:.1f} × 10^-24 GeV^-1")

    plt.close()

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    plot_results(config_file)
