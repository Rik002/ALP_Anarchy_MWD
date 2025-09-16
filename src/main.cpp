#include "Simulation.h"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        std::string config_file = (argc > 1) ? argv[1] : "config.ini";
        Simulation sim(config_file);
        sim.run();
        std::cout << "Simulation completed successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
