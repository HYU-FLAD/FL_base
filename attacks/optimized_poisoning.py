import numpy as np
from clients import FlowerClient, train, trainloaders

class OptimizedPoisoningClient(FlowerClient):
    """
    Inherits from the base client and implements a model poisoning attack
    based on reversing the direction of the model update.
    """
    def fit(self, parameters, config):
        """
        This method overrides the base `fit` method to inject the attack.
        The attack works as follows:
        1. Perform standard local training to find the benign update direction.
        2. Intentionally craft a malicious update that points in the opposite direction.
        3. Send this malicious update back to the server.
        """
        print(f"[Client {self.cid}] Running Optimized Poisoning Attack!")

        # 1. Get initial global model parameters
        initial_params = [np.copy(p) for p in parameters]

        # 2. Perform standard local training
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        benign_updated_params = self.get_parameters(config={})

        # 3. Calculate the benign update direction
        # benign_update = (locally_trained_model - global_model)
        benign_update = [
            benign_p - initial_p for benign_p, initial_p in zip(benign_updated_params, initial_params)
        ]

        # 4. Craft the malicious update by reversing the direction and scaling it
        # The 'aggression' factor determines the attack's strength.
        aggression = 5.0
        malicious_update = [-aggression * u for u in benign_update]

        # 5. Create the final poisoned parameters to send to the server
        # poisoned_params = global_model + malicious_update
        poisoned_parameters = [
            initial_p + malicious_u for initial_p, malicious_u in zip(initial_params, malicious_update)
        ]

        return poisoned_parameters, len(self.trainloader.dataset), {}

def optimized_poisoning_client_fn(cid: str) -> OptimizedPoisoningClient:
    """Creates a malicious client for the optimized poisoning attack."""
    trainloader = trainloaders[int(cid)]
    return OptimizedPoisoningClient(cid, trainloader)