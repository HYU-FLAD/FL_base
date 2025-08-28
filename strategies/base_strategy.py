import flwr as fl
import torch
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
from models import CNNet
from utils import load_data

# --------------------------------------------------------------------------
# 0. Environment Setup & Data/Model Loading
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, testset = load_data()
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
server_model = CNNet().to(DEVICE)

# --------------------------------------------------------------------------
# 1. Centralized Evaluation Function Definition
# --------------------------------------------------------------------------
def get_evaluate_fn():
    """Return a centralized evaluation function to be used by the server."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        # Set the current global model parameters on the server-side model
        params_dict = zip(server_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        server_model.load_state_dict(state_dict, strict=True)

        # Evaluate the model
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        server_model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = server_model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / len(testloader)
        accuracy = correct / total
        print(f"Round {server_round}: Server-side evaluation loss {avg_loss} / accuracy {accuracy}")
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate

# --------------------------------------------------------------------------
# 2. Custom Strategy Class Definition
# --------------------------------------------------------------------------
class BaseStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the training results from clients."""
        
        # === PAPER IMPLEMENTATION: DEFENSE LOGIC GOES HERE ===
        # The `results` list contains parameters and sample sizes from each client.
        # You can analyze this data to filter out malicious updates or
        # apply a Robust Aggregation algorithm (e.g., RFA, Krum).
        # e.g.: filtered_results = my_defense_function(results)
        #       aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, filtered_results, failures)
        # =====================================================
        
        # Call the default FedAvg aggregation logic
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        return aggregated_parameters, aggregated_metrics

# --------------------------------------------------------------------------
# 3. Strategy Object Instantiation
# --------------------------------------------------------------------------
# `min_fit_clients`: Minimum number of clients to participate in each training round.
# `min_available_clients`: Minimum number of clients required to start a training round.
# `evaluate_fn`: The function to evaluate the global model after each round.
base_strategy = BaseStrategy(
    min_fit_clients=5,
    min_available_clients=10,
    evaluate_fn=get_evaluate_fn(),
)