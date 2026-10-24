"""
Unified script to execute all benchmark experiments sequentially.
"""

import os
import subprocess
import sys


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(project_root, "configs")
    driver_script = os.path.join(project_root, "scripts", "run_experiment.py")

    config_files = sorted([
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith(".yaml") or f.endswith(".yml")
    ])

    if not config_files:
        print(f"No configuration files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(config_files)} experiment configuration(s) to execute.")

    for cfg_path in config_files:
        print(f"\n[Running] {os.path.basename(cfg_path)}")
        cmd = [sys.executable, driver_script, "--config", cfg_path]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[Error] Configuration failed with exit code {ret.returncode}: {cfg_path}")
            sys.exit(ret.returncode)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
