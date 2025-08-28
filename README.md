# A Simulation Framework for Adversarial Robustness in Federated Learning

## Overview

This project presents a modular simulation framework designed for the empirical study of adversarial attacks and defenses within the Federated Learning (FL) paradigm. Leveraging the Flower simulation engine and the PyTorch deep learning library, this framework facilitates the controlled evaluation of security vulnerabilities and mitigation strategies in a distributed learning environment.

The primary objective is to provide a structured and extensible platform for implementing, testing, and analyzing the dynamics of adversarial interactions in FL. The modular design enables researchers to systematically investigate the efficacy of various security mechanisms against a range of potential threats.

-----

## Features

  * **Modular and Extensible Architecture**: The framework distinctly separates components such as client behaviors, server-side aggregation strategies, model architectures, and data handling. This separation of concerns allows for straightforward extension and integration of new algorithms.
  * **Flexible Experiment Configuration**: A command-line interface enables the flexible configuration of experiments, allowing users to select and combine different attack and defense modules dynamically.
  * **Implementation of Academic Research**: The framework includes implementations of algorithms and concepts from peer-reviewed publications in the field of Federated Learning security, providing a basis for reproducible research.

-----

## Project Structure

The project is organized using a package-based structure to ensure clarity and maintainability.

```
fl_project/
├── main.py               # Main script for executing simulation experiments
│
├── attacks/              # Directory for adversarial client implementations
│   ├── __init__.py
│   └── optimized_poisoning.py
│
├── defenses/             # Directory for robust aggregation strategy implementations
│   ├── __init__.py
│   └── robust_aggregation.py
│
├── clients/              # Contains the base client implementation
│   ├── __init__.py
│   └── base_client.py
│
├── strategies/           # Contains the base server strategy implementation
│   ├── __init__.py
│   └── base_strategy.py
│
├── models/               # Contains model architecture definitions
│   └── ...
│
├── data/                 # Contains data loading and partitioning logic
│   └── ...
│
└── requirements.txt      # Lists project dependencies
```

-----

## Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Dexoculus/FL_base.git
    cd fl_project
    ```

2.  **Create and Activate a Python Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
-----

## Execution of Experiments

Simulations are executed via the `main.py` script. The script accepts command-line arguments to define the experimental conditions.

#### **Available Arguments:**

  * `--attack`: Specifies the client-side behavior model.
      * `benign` (default): Standard, honest client behavior.
      * `optimized_poisoning`: An implementation of a model poisoning attack.
  * `--defense`: Specifies the server-side aggregation strategy.
      * `fedavg` (default): Standard Federated Averaging algorithm.
      * `rfa`: Robust Federated Aggregation using the geometric median.

#### **Example Commands:**

  * **Baseline simulation with benign clients and the FedAvg strategy:**

    ```bash
    python main.py --attack benign --defense fedavg
    ```

  * **Simulation of a model poisoning attack under the standard FedAvg strategy:**

    ```bash
    python main.py --attack optimized_poisoning --defense fedavg
    ```

  * **Evaluation of the RFA defense against the model poisoning attack:**

    ```bash
    python main.py --attack optimized_poisoning --defense rfa
    ```

-----

## Implemented Research

This framework provides implementations of the core concepts from the following publications:

### 1\. Attack: Optimized Model Poisoning

  * **Reference Paper**: Shejwalkar, V., Houmansadr, A., et al. (2021). "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning." *NDSS Symposium 2021*.

  * **Theoretical Background**: The paper posits that naive poisoning attacks, such as random noise injection, are often easily mitigated by robust aggregation rules. A more effective attack must be carefully optimized. The goal of an optimized attack is to craft a malicious model update that, while appearing statistically similar to benign updates to evade detection, maximally deviates the aggregated global model from the optimal convergence path.

  * **Framework Implementation (`attacks/optimized_poisoning.py`)**: Our implementation models this principle through a potent **direction-based attack**. The malicious client first computes its benign local update by training on its data. This establishes a "benign update direction." Subsequently, it crafts a malicious update by scaling this directional vector by a large negative factor. The submitted update thus points in the diametrically opposite direction of the honest update, effectively poisoning the aggregation step by pulling the global model in a direction detrimental to its performance.

### 2\. Defense: Robust Federated Aggregation (RFA)

  * **Reference Paper**: Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2022). "Robust Aggregation for Federated Learning." *IEEE Transactions on Signal Processing*.

  * **Theoretical Background**: The standard FedAvg algorithm, which relies on averaging model parameters, is highly susceptible to outlier updates submitted by malicious clients. RFA replaces the arithmetic mean with the **geometric median**, a classic concept from robust statistics. The geometric median of a set of points is defined as the point that minimizes the sum of Euclidean distances to all points in the set. This property makes it inherently robust to outliers, as a single malicious point cannot arbitrarily shift the aggregate.

  * **Framework Implementation (`defenses/robust_aggregation.py`)**: This module implements RFA as a custom Flower `Strategy`. It overrides the standard `aggregate_fit` method. Within this method, model updates from all participating clients are collected, vectorized, and fed into an iterative algorithm (Weiszfeld's algorithm) to compute their geometric median. The resulting aggregate, which is resistant to poisoning attacks, is then used to update the global model.
