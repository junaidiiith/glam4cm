import argparse
from tqdm.auto import tqdm
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confs_file', type=str, help='File containing configurations to run')
    args = parser.parse_args()
    return args

def run_tasks(configs):
    for config in tqdm(configs, desc="Running tasks"):
        print(f"Running config: {config}")
        result = subprocess.run(f'python glam_test.py {config}', shell=True)
        if result.returncode != 0:
            print(f"Error running config {config}: {result.stderr}")
        else:
            print(f"Config {config} completed successfully.")
    print("All tasks completed.")
    
def main():
    args = get_args()
    if not args.confs_file:
        print("No configuration file specified. Exiting.")
        return
    try:
        with open(args.confs_file, 'r') as f:
            configs = f.read().splitlines()
        if not configs:
            print("Configuration file is empty. Exiting.")
            return
        print(f"Found {len(configs)} configurations to run.")
        run_tasks(configs)
    except FileNotFoundError:
        print(f"Configuration file {args.confs_file} not found. Exiting.")
    except Exception as e:
        print(f"An error occurred: {e}. Exiting.")

if __name__ == "__main__":
    main()