import configparser
import os
import sys

def clean_value(value):
    value = value.strip()
    if ' ' in value:
        value = value.split()[0]
    if value.endswith(':'):
        value = value[:-1]
    return value

def check_simulation(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    data_dir = clean_value(config['Directories']['data_dir'])
    N_min = int(clean_value(config['Simulation']['N_min']))
    N_max = int(clean_value(config['Simulation']['N_max']))
    N_real = int(clean_value(config['Simulation']['N_real']))

    print("Checking if C++ simulation completed successfully...")
    print(f"Expected: {N_max - N_min + 1} CSV files with {N_real} bounds each")

    missing_files = []
    for N in range(N_min, N_max + 1):
        filename = f"{data_dir}bounds_N{N}.txt"
        if not os.path.exists(filename):
            missing_files.append(f"bounds_N{N}.csv")

    if missing_files:
        print(f"ERROR: Missing {len(missing_files)} files:")
        for f in missing_files[:5]:  # Show first 5 missing
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        print("\nPlease run the C++ simulation first!")
        return False
    else:
        print("✓ All expected files found!")
        return True

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    success = check_simulation(config_file)
    sys.exit(0 if success else 1)
