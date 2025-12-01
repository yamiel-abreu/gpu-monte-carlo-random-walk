# analysis/plot_speedup.py
# script to create some performance evaluation plots.

import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

def load_results(path):
    data = defaultdict(dict)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row["backend"]
            n = int(row["num_walkers"])
            t = float(row["time_s"])
            data[n][backend] = t
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_speedup.py results.csv")
        sys.exit(1)

    data = load_results(sys.argv[1])

    sizes = sorted(data.keys())
    cpu_times = [data[n]["CPU"] for n in sizes]
    gpu_times = [data[n]["GPU"] for n in sizes]
    speedups  = [c/g for c, g in zip(cpu_times, gpu_times)]

    # Plot runtime
    plt.figure()
    plt.loglog(sizes, cpu_times, marker='o', label="CPU")
    plt.loglog(sizes, gpu_times, marker='s', label="GPU")
    plt.xlabel("Number of walkers")
    plt.ylabel("Wall time [s]")
    plt.title("Random Walk Monte Carlo: CPU vs GPU runtime")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("runtime_comparison.png", dpi=200)

    # Plot speedup
    plt.figure()
    plt.semilogx(sizes, speedups, marker='^')
    plt.xlabel("Number of walkers")
    plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title("Random Walk Monte Carlo: GPU speedup")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=200)

if __name__ == "__main__":
    main()
