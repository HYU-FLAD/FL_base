import flwr as fl
import torch
from collections import OrderedDict
from models import CNNet
from utils import load_data, prepare_federated_data

# --------------------------------------------------------------------------
# 0. Environment Setup & Data Loading
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 10

# Load and partition the dataset
trainset, _ = load_data()
trainloaders = prepare_federated_data(trainset, num_clients=NUM_CLIENTS)

# --------------------------------------------------------------------------
# 1. Model Training & Evaluation Functions
# --------------------------------------------------------------------------
def train(net, trainloader, epochs):
    """Train the model using the provided dataloader."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Evaluate the model's performance on the provided dataloader."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss over all batches in the dataloader
    avg_loss = loss / len(testloader)
    accuracy = correct / total
    return avg_loss, accuracy

# --------------------------------------------------------------------------
# 2. Flower Client Class Definition
# --------------------------------------------------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader):
        self.cid = cid
        self.trainloader = trainloader
        self.net = CNNet().to(DEVICE)

    def get_parameters(self, config):
        """Return the current local model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Update the local model with parameters received from the server."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Update the local model with received parameters, train it on local data, and return the updated parameters."""
        print(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        
        # Train the model on local data
        train(self.net, self.trainloader, epochs=1)
        
        # === PAPER IMPLEMENTATION: ATTACK LOGIC GOES HERE ===
        # You can manipulate the parameters here before returning them to the server.
        # e.g.: updated_params = self.get_parameters(config={})
        #       poisoned_params = my_attack_function(updated_params)
        #       return poisoned_params, len(self.trainloader.dataset), {}
        # ====================================================

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model using parameters received from the server on a local testset."""
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        
        # If clients had their own test sets, they would be evaluated here.
        # Since this example uses centralized evaluation on the server, we return simple values.
        # loss, accuracy = test(self.net, self.valloader) 
        # return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
        num_examples = len(self.trainloader.dataset) # Simply return the number of training examples
        return 0.0, num_examples, {"accuracy": 0.0}

# --------------------------------------------------------------------------
# 3. Client Factory Function (for simulation)
# --------------------------------------------------------------------------
def base_client_fn(cid: str) -> FlowerClient:
    """A function that Flower simulation calls to create each client."""
    # Get the dataloader corresponding to the client ID (cid).
    trainloader = trainloaders[int(cid)]
    return FlowerClient(cid, trainloader)