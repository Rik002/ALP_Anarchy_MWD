import numpy as np
import configparser
import sys
import os

def read_bounds_file(filename):
    """Read bounds from TXT file"""
    bounds = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    bounds.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(bounds)



def clean_value(value):
    value = value.strip()
    if ' ' in value:
        value = value.split()[0]
    if value.endswith(':'):
        value = value[:-1]
    return value



def debug_data(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    data_dir = clean_value(config['Directories']['data_dir'])

    print("=" * 60)
    print("DEBUG: Checking TXT data files...")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Data directory '{data_dir}' does not exist!")
        return False

    print(f"Data directory: {data_dir}")

    files = os.listdir(data_dir)
    txt_files = [f for f in files if f.endswith('.txt') and f.startswith('bounds_')]
    print(f"Found {len(txt_files)} bound TXT files: {sorted(txt_files)}")


    config_N_min = int(clean_value(config['Simulation']['N_min']))
    config_N_max = int(clean_value(config['Simulation']['N_max']))

    has_invalid_data = False

    for N in range(config_N_min, min(config_N_max + 1, config_N_min + 8)):  # Check first 8
        filename = f"{data_dir}bounds_N{N}.txt"
        try:
            if os.path.exists(filename):
                data = read_bounds_file(filename)
                print(f"N={N}: {len(data)} bounds")

                if len(data) > 0:
                    print(f"  Range: [{data.min():.2e}, {data.max():.2e}]")
                    print(f"  Sample: {data[:3]}")

                    finite_data = data[np.isfinite(data) & (data > 0)]
                    if len(finite_data) != len(data):
                        print(f"  ❌ WARNING: {len(data) - len(finite_data)} invalid values!")
                        has_invalid_data = True

                    if np.all(data == 0):
                        print(f"  ❌ ERROR: All values are zero!")
                        has_invalid_data = True
                    elif len(finite_data) > 0:
                        print(f"  ✓ Valid range: [{finite_data.min():.2e}, {finite_data.max():.2e}]")
                else:
                    print(f"  ❌ ERROR: File is empty or unreadable!")
                    has_invalid_data = True
            else:
                print(f"N={N}: ❌ File {filename} does not exist!")
                has_invalid_data = True
        except Exception as e:
            print(f"N={N}: ❌ Error reading file - {e}")
            has_invalid_data = True

    print("=" * 60)
    if has_invalid_data:
        print("❌ CRITICAL: Invalid data detected!")
        return False
    else:
        print("✓ All data looks valid!")
        return True

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    success = debug_data(config_file)
    sys.exit(0 if success else 1)
