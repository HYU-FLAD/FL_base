import flwr as fl
import argparse

# Import base components
from clients.base_client import base_client_fn
from strategies.base_strategy import base_strategy

# Import all available attacks and defenses from the libraries
from attacks import *
from defenses import *

# Map command-line names to the actual client/strategy functions
CLIENT_FACTORIES = {
    "benign": base_client_fn,
    "optimized_poisoning": optimized_poisoning_client_fn
}

STRATEGIES = {
    "fedavg": base_strategy,
    "rfa": rfa_strategy
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Security Simulation")
    parser.add_argument("--attack", type=str, default="benign", choices=CLIENT_FACTORIES.keys())
    parser.add_argument("--defense", type=str, default="fedavg", choices=STRATEGIES.keys())
    args = parser.parse_args()

    print(f"Running simulation with Attack: '{args.attack}' and Defense: '{args.defense}'")

    # Select the client factory and strategy based on arguments
    client_factory = CLIENT_FACTORIES[args.attack]
    strategy = STRATEGIES[args.defense]

    # Start the simulation with the selected components
    fl.simulation.start_simulation(
        client_fn=client_factory,
        strategy=strategy,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=5),
        client_resources={"num_cpus": 1, "num_gpus": 0.2},
    )