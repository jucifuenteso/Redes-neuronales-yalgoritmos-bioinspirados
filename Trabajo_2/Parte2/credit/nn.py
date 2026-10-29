"""Perceptrón multicapa en PyTorch + réplica en NumPy para inferencia (app sin torch)."""
import time

import numpy as np
import torch
from torch import nn

ACTIVATION_LAYERS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU,
                     "leaky_relu": lambda: nn.LeakyReLU(0.01)}


def build_mlp(n_input_features, hidden_layer_sizes, activation_name, dropout_probability):
    layers, previous_layer_size = [], n_input_features
    for hidden_layer_size in hidden_layer_sizes:
        layers += [nn.Linear(previous_layer_size, hidden_layer_size), ACTIVATION_LAYERS[activation_name](),
                   nn.Dropout(dropout_probability)]
        previous_layer_size = hidden_layer_size
    layers.append(nn.Linear(previous_layer_size, 1))  # logit (PD y LGD: sigmoide + BCE/MSE)
    return nn.Sequential(*layers)


def make_optimizer(optimizer_name, parameters, learning_rate, weight_decay):
    # Adam/RMSprop/SGD con weight_decay = penalización L2 (se suma al gradiente);
    # AdamW desacopla la decaída de pesos del gradiente (Loshchilov & Hutter, 2019).
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(optimizer_name)


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_mlp(features_train, target_train, features_val, target_val, hyperparameters, seed, loss="bce",
            verbose=False):
    """Entrena con parada temprana sobre la pérdida de validación; restaura los mejores pesos.

    loss: 'bce' (Bernoulli / cuasi-Bernoulli para objetivos en [0,1]) o 'mse' (gaussiana).
    Devuelve (modelo, info) con info = {epochs, seconds, val_loss, n_params}.
    """
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = build_mlp(features_train.shape[1], hyperparameters["hidden"], hyperparameters["activation"],
                      hyperparameters["dropout"])
    optimizer = make_optimizer(hyperparameters["optimizer"], model.parameters(), hyperparameters["lr"],
                               hyperparameters["weight_decay"])
    binary_cross_entropy = nn.BCEWithLogitsLoss() if loss == "bce" else None
    train_features_tensor = torch.from_numpy(features_train)
    train_target_tensor = torch.from_numpy(target_train).float()
    val_features_tensor = torch.from_numpy(features_val)
    val_target_tensor = torch.from_numpy(target_val).float()
    shuffling_generator = torch.Generator().manual_seed(seed)
    batch_size, n_train_rows = int(hyperparameters["batch_size"]), len(train_features_tensor)

    def compute_loss(logits, targets):
        if loss == "bce":
            return binary_cross_entropy(logits, targets)
        return torch.mean((torch.sigmoid(logits) - targets) ** 2)

    def compute_validation_loss():
        model.eval()
        with torch.no_grad():
            return float(compute_loss(model(val_features_tensor).squeeze(1), val_target_tensor))

    best_validation_loss, best_weights = float("inf"), None
    epochs_without_improvement, training_start_time, epochs_run = 0, time.time(), 0
    for epochs_run in range(1, int(hyperparameters["epochs"]) + 1):
        model.train()
        shuffled_row_order = torch.randperm(n_train_rows, generator=shuffling_generator)
        for batch_start in range(0, n_train_rows, batch_size):
            batch_rows = shuffled_row_order[batch_start:batch_start + batch_size]
            batch_loss = compute_loss(model(train_features_tensor[batch_rows]).squeeze(1),
                                      train_target_tensor[batch_rows])
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        validation_loss = compute_validation_loss()
        if verbose:
            print(f"  época {epochs_run:3d}  val_loss={validation_loss:.5f}")
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss, epochs_without_improvement = validation_loss, 0
            best_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(hyperparameters["patience"]):
                break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model, {"epochs": epochs_run, "seconds": time.time() - training_start_time,
                   "val_loss": best_validation_loss, "n_params": count_trainable_parameters(model)}


def predict(model, features, batch_size=65536):
    model.eval()
    probability_batches = []
    with torch.no_grad():
        for batch_start in range(0, len(features), batch_size):
            batch_features = torch.from_numpy(features[batch_start:batch_start + batch_size])
            probability_batches.append(torch.sigmoid(model(batch_features).squeeze(1)).numpy())
    return np.concatenate(probability_batches) if probability_batches else np.empty(0)


# ---------- exportación a NumPy (la app y las pruebas no necesitan torch) ----------
def export_numpy(model, activation_name, path):
    weights_and_biases = [(layer.weight.detach().numpy().T, layer.bias.detach().numpy())
                          for layer in model if isinstance(layer, nn.Linear)]
    arrays = {f"W{layer_index}": weights for layer_index, (weights, _) in enumerate(weights_and_biases)}
    arrays.update({f"b{layer_index}": biases for layer_index, (_, biases) in enumerate(weights_and_biases)})
    np.savez(path, activation=np.array(activation_name), n_layers=np.array(len(weights_and_biases)), **arrays)


class NumpyMLP:
    def __init__(self, path):
        saved_arrays = np.load(path)
        self.activation_name = str(saved_arrays["activation"])
        self.n_layers = int(saved_arrays["n_layers"])
        self.weights = [saved_arrays[f"W{layer_index}"] for layer_index in range(self.n_layers)]
        self.biases = [saved_arrays[f"b{layer_index}"] for layer_index in range(self.n_layers)]

    def _activation(self, pre_activation):
        name = self.activation_name
        if name == "relu":
            return np.maximum(pre_activation, 0)
        if name == "tanh":
            return np.tanh(pre_activation)
        if name == "elu":
            return np.where(pre_activation > 0, pre_activation, np.expm1(np.minimum(pre_activation, 0)))
        if name == "silu":
            return pre_activation / (1 + np.exp(-pre_activation))
        if name == "leaky_relu":
            return np.where(pre_activation > 0, pre_activation, 0.01 * pre_activation)
        raise ValueError(name)

    def predict(self, features):
        hidden_activations = np.asarray(features, dtype=np.float64)
        for layer_index in range(self.n_layers - 1):
            hidden_activations = self._activation(hidden_activations @ self.weights[layer_index]
                                                  + self.biases[layer_index])
        logits = (hidden_activations @ self.weights[-1] + self.biases[-1]).ravel()
        return 1.0 / (1.0 + np.exp(-logits))
