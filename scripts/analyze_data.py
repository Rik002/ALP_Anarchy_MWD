import numpy as np
import configparser
import os
from lib.statistical_tests import perform_ks_test

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
        'N_min': int(clean_value(config['Simulation']['N_min'])),
        'N_max': int(clean_value(config['Simulation']['N_max'])),
        'data_dir': clean_value(config['Directories']['data_dir']),
        'ks_alpha': float(clean_value(config['StatisticalTests']['ks_alpha']))
    }

def read_bounds_file(filename):
    """Read bounds from TXT file, skipping header comments"""
    bounds = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    bound_value = float(parts[1])  # Second column is bound value
                    bounds.append(bound_value)
                except ValueError:
                    continue
    return np.array(bounds)

def analyze_data(config_file):
    config = load_config(config_file)
    N_range = range(config['N_min'], config['N_max'] + 1)
    lower_90, upper_90, lower_third, upper_third = [], [], [], []

    for N in N_range:
        filename = f"{config['data_dir']}bounds_N{N}.txt"
        print(f"Processing {filename}...")

        try:
            data = read_bounds_file(filename)
            print(f"  Read {len(data)} bounds, range [{data.min():.2e}, {data.max():.2e}]")

            finite_data = data[np.isfinite(data) & (data > 0)]
            sorted_data = np.sort(finite_data)

            if len(finite_data) > 0:
                p_value, rejected = perform_ks_test(finite_data, config['ks_alpha'])
                with open(f"{config['data_dir']}stats_N{N}.txt", 'a') as f:
                    f.write(f"# KS Test Results\n")
                    f.write(f"KS_p_value: {p_value:.6f}\n")
                    f.write(f"KS_rejected: {rejected}\n")

                lower_90.append(np.percentile(sorted_data, 5))
                upper_90.append(np.percentile(sorted_data, 95))
                lower_third.append(np.percentile(sorted_data, 33.333))
                upper_third.append(np.percentile(sorted_data, 66.667))
                print(f"  ✓ Valid data: percentiles calculated")
            else:
                print(f"  ❌ No valid data found")
                lower_90.append(np.nan)
                upper_90.append(np.nan)
                lower_third.append(np.nan)
                upper_third.append(np.nan)

        except Exception as e:
            print(f"  ❌ Error reading {filename}: {e}")
            lower_90.append(np.nan)
            upper_90.append(np.nan)
            lower_third.append(np.nan)
            upper_third.append(np.nan)

    # Save bands data as TXT with header
    bands_file = f"{config['data_dir']}bands.txt"
    with open(bands_file, 'w') as f:
        f.write("# ALP Anarchy Bands Data\n")
        f.write("# Columns: N, Lower_90, Upper_90, Lower_Third, Upper_Third\n")
        f.write("#\n")
        for i, N in enumerate(N_range):
            f.write(f"{N:2d}  {lower_90[i]:.6e}  {upper_90[i]:.6e}  {lower_third[i]:.6e}  {upper_third[i]:.6e}\n")

    print(f"✓ Saved bands data to {bands_file}")
    return N_range, lower_90, upper_90, lower_third, upper_third

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    analyze_data(config_file)
