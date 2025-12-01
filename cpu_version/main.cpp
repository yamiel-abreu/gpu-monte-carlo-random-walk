// cpu_version/main.cpp
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <cmath>

struct SimulationConfig {
    int num_walkers;
    int num_steps;
    unsigned int seed;
};

double run_random_walk_cpu(const SimulationConfig& cfg) {
    std::mt19937 rng(cfg.seed);
    std::uniform_int_distribution<int> dist(0, 5);

    double sum_r2 = 0.0;

    for (int i = 0; i < cfg.num_walkers; ++i) {
        int x = 0, y = 0, z = 0;

        for (int step = 0; step < cfg.num_steps; ++step) {
            int dir = dist(rng);
            switch (dir) {
                case 0: x += 1; break;
                case 1: x -= 1; break;
                case 2: y += 1; break;
                case 3: y -= 1; break;
                case 4: z += 1; break;
                case 5: z -= 1; break;
            }
        }

        double r2 = static_cast<double>(x*x + y*y + z*z);
        sum_r2 += r2;
    }

    return sum_r2 / static_cast<double>(cfg.num_walkers);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_walkers> <num_steps> [seed]\n";
        return 1;
    }

    SimulationConfig cfg;
    cfg.num_walkers = std::stoi(argv[1]);
    cfg.num_steps   = std::stoi(argv[2]);
    cfg.seed        = (argc > 3) ? static_cast<unsigned int>(std::stoul(argv[3])) : 42u;

    std::cout << "CPU Random Walk Simulation\n";
    std::cout << "  num_walkers = " << cfg.num_walkers << "\n";
    std::cout << "  num_steps   = " << cfg.num_steps << "\n";
    std::cout << "  seed        = " << cfg.seed << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    double msd = run_random_walk_cpu(cfg);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;

    std::cout << "Result:\n";
    std::cout << "  Mean squared displacement = " << msd << "\n";
    std::cout << "Timing:\n";
    std::cout << "  CPU wall time = " << elapsed.count() << " s\n";

    return 0;
}
