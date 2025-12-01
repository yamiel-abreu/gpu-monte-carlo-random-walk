# Script to run the benchmaks jobs and generate the data for evaluation. 

# Code compilations on release mode needed to start.

# The following excecutables will be used:
# build_cpu\Release\cpu_random_walk.exe
# build_gpu\Release\gpu_random_walk.exe 

import subprocess
import csv
from pathlib import Path

# Paths to the built executables (relative to this script)
ROOT = Path(__file__).resolve().parents[1]
CPU_EXE = ROOT / "build_cpu" / "Release" / "cpu_random_walk.exe"
GPU_EXE = ROOT / "build_gpu" / "Release" / "gpu_random_walk.exe"

NUM_STEPS = 1000
SEED = 42

# You can tweak this list
NUM_WALKERS_LIST = [
    10_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
]

def run_exe(exe_path, num_walkers, num_steps, seed, backend):
    """Run one executable and parse timing from stdout."""
    print(f"Running {backend} with {num_walkers} walkers...")

    result = subprocess.run(
        [str(exe_path), str(num_walkers), str(num_steps), str(seed)],
        capture_output=True, text=True, check=True
    )

    stdout = result.stdout
    time_s = None

    if backend == "CPU":
        # Look for line: "CPU wall time = X s"
        for line in stdout.splitlines():
            if "CPU wall time" in line:
                time_s = float(line.split("=")[1].strip().split()[0])
                break
    else:
        # Look for line: "Total time  = X ms"
        for line in stdout.splitlines():
            if "Total time" in line:
                ms = float(line.split("=")[1].strip().split()[0])
                time_s = ms / 1000.0
                break

    if time_s is None:
        print("Could not parse time from output:")
        print(stdout)
        raise RuntimeError("Timing parse failed")

    return time_s

def main():
    out_csv = Path(__file__).with_name("results.csv")
    print(f"Writing results to {out_csv}")

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "num_walkers", "num_steps", "time_s"])

        for n in NUM_WALKERS_LIST:
            cpu_t = run_exe(CPU_EXE, n, NUM_STEPS, SEED, "CPU")
            writer.writerow(["CPU", n, NUM_STEPS, cpu_t])

            gpu_t = run_exe(GPU_EXE, n, NUM_STEPS, SEED, "GPU")
            writer.writerow(["GPU", n, NUM_STEPS, gpu_t])

    print("Done.")

if __name__ == "__main__":
    main()
