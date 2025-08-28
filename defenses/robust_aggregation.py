import numpy as np
from typing import List, Tuple, Optional
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from strategies import BaseStrategy

def geometric_median(points: List[np.ndarray], max_iter=100, tol=1e-5) -> np.ndarray:
    """
    Computes the geometric median of a list of points (vectors) using Weiszfeld's algorithm.
    """
    # Initial guess: the mean
    median = np.mean(points, axis=0)
    
    for i in range(max_iter):
        prev_median = median.copy()
        
        distances = np.linalg.norm(points - median, axis=1)
        
        # Avoid division by zero if a point is the same as the current median
        distances[distances == 0] = 1e-10
        
        weights = 1 / distances
        weights = weights / np.sum(weights)
        
        median = np.sum(points * weights[:, np.newaxis], axis=0)
        
        if np.linalg.norm(median - prev_median) < tol:
            print(f"Geometric median converged after {i+1} iterations.")
            break
            
    return median

class RfaStrategy(BaseStrategy):
    """
    Inherits from the base strategy and overrides the aggregation function
    to use the geometric median (Robust Federated Aggregation - RFA).
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Optional[Parameters]:
        """
        Aggregates model updates using the geometric median.
        """
        if not results:
            return None

        print("----> Applying Robust Federated Aggregation (RFA) defense!")

        # 1. Convert received model parameters into a list of vectors
        # Each client's parameters are flattened into a single long vector
        param_updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Keep the structure of the first model to reshape the result later
        model_shape_template = [arr.shape for arr in param_updates[0]]
        
        flat_updates = [np.concatenate([arr.flatten() for arr in p]) for p in param_updates]

        # 2. Compute the geometric median of the parameter vectors
        geo_median_flat = geometric_median(np.array(flat_updates))

        # 3. Reshape the flat geometric median back to the original model structure
        aggregated_params_list = []
        current_pos = 0
        for shape in model_shape_template:
            num_elements = np.prod(shape)
            arr = geo_median_flat[current_pos : current_pos + num_elements].reshape(shape)
            aggregated_params_list.append(arr)
            current_pos += num_elements
            
        # Convert back to Flower's Parameters format
        aggregated_parameters = ndarrays_to_parameters(aggregated_params_list)

        return aggregated_parameters, {}

# Create an instance of the defense strategy
# This requires the evaluate_fn from the base strategy setup
from strategies.base_strategy import get_evaluate_fn
rfa_strategy = RfaStrategy(
    min_fit_clients=5,
    min_available_clients=10,
    evaluate_fn=get_evaluate_fn(),
)