
from typing import Any, Generator, Iterable, Union

import numpy as np
import torch
import warnings
from fluke import FlukeENV  # NOQA
from fluke.algorithms import CentralizedFL  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import Message  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.server import EarlyStopping, Server  # NOQA
from fluke.utils import OptimizerConfigurator  # NOQA
from rich.progress import track
from sklearn.linear_model import RidgeClassifierCV  # NOQA
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket
from openmp_utils import transform_features, select_lambda

def transform(X: np.ndarray,
              client_id: int,
              seed: int,
              ppv_only: bool = True,
              caching: bool = True) -> np.ndarray:
    # client_id is retained for API compatibility, never used as data identity.
    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    return transform_features(X, [seed], ppv_only=ppv_only, caching=caching)


def transform_seeds(X: np.ndarray,
                    client_id: int,
                    seeds: Iterable[int],
                    caching: bool = True,
                    use_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    Xnp = transform_features(X, seeds, caching=caching)
    return torch.from_numpy(Xnp) if use_torch else Xnp


class RidgeTorchModel(torch.nn.Module):
    def __init__(self,
                 W: np.ndarray,
                 b: np.ndarray,
                 seeds: Iterable[int]):
        super().__init__()
        self.model = torch.nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
        self.seeds = seeds

        with torch.no_grad():
            self.model.weight.copy_(torch.tensor(W, dtype=torch.float32))
            if b is not None:
                self.model.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def forward(self, X: torch.Tensor):
        X = transform_seeds(X, -1, self.seeds, use_torch=True, caching=False)
        output = self.model(X)
        return output


def compress_model(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    eigvals, eigvecs = np.linalg.eigh(A)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    U_k = eigvecs[:, :k]
    Lambda_k = np.diag(eigvals[:k])
    return (U_k, Lambda_k)


class ClientFed2RC(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,  # Not used
                 loss_fn: torch.nn.Module,  # Not used
                 local_epochs: int = 3,  # Not used
                 fine_tuning_epochs: int = 0,  # Not used
                 clipping: float = 0,  # Not used
                 n_kernels: int = 1,
                 top_k: int = 1,
                 compression_factor: int = 0,
                 **kwargs: dict[str, Any]):
        assert compression_factor >= 1, "Compression factor must be greater or equal than 1"
        assert n_kernels > 0, "Number of kernels must be greater than 0"
        super().__init__(index, train_set, test_set, None, None, 0, 0, 0, **kwargs)
        self.hyper_params.update(n_kernels=n_kernels, top_k=top_k,
                                 compression_factor=compression_factor)
        self.seeds: Iterable[int] = np.arange((self.index) * n_kernels,
                                              (self.index + 1) * n_kernels)
        self.converged: bool = False
        self.A: np.ndarray = None
        self.b: np.ndarray = None

    def send_model(self):
        if self.converged:
            self.channel.send(Message((self.A, self.b), "Ab", self), self.server)
        else:
            self.channel.send(Message(self.top_k_idx, "seeds", self), self.server)

    def receive_model(self):
        self.converged = self.channel.receive(self, self.server, msg_type="converge").payload
        if not self.converged:
            msg = self.channel.receive(self, self.server, msg_type="seeds")
            self.seeds = msg.payload if self.server.rounds != 0 else self.seeds

    def fit(self, **kwargs: dict[str, Any]) -> float:
        X, y = self.train_set.tensors
        X, y = X.numpy(), y.numpy()
        X_trans = transform_seeds(X, self.index, self.seeds)

        if self.converged:
            y_onehot = np.eye(self.train_set.num_labels)[y]
            self.A = X_trans.T @ X_trans
            self.b = X_trans.T @ y_onehot

            if self.hyper_params.compression_factor > 1:
                k = max(1, self.hyper_params.n_kernels // self.hyper_params.compression_factor)
                if self.hyper_params.compression_factor > self.hyper_params.n_kernels:
                    warnings.warn("Compression factor is greater than the number of kernels. "
                                  "Using the number of kernels instead.")

                U_k, Lambda_k = compress_model(self.A, k)
                self.A = (U_k, Lambda_k)
        else:
            ridge = RidgeClassifierCV(alphas=np.logspace(-3, 0, 100)).fit(X_trans, y)
            W = np.atleast_2d(ridge.coef_)
            # Useful if client evaluation is needed
            self.model = RidgeTorchModel(W,
                                         ridge.intercept_,
                                         self.seeds)
            self.top_k_idx = self.seeds[np.argsort(-np.abs(W[0]))
                                        [:self.hyper_params.top_k]]

        return 0.0

    def tune_lambda(self):
        X, y = self.train_set.tensors
        X, y = X.numpy(), y.numpy()
        X_trans = transform_seeds(X, self.index, self.seeds)

        A_global, b_global = self.channel.receive(self, self.server, "Ab").payload
        if isinstance(A_global, tuple):
            U, L = A_global
            A_global = U @ L @ U.T
            self.A = X_trans.T @ X_trans
        A_minus_client = A_global - self.A
        b_minus_client = b_global - self.b

        tmp_lambda, best_accuracy, _ = select_lambda(
            A_minus_client, b_minus_client, X_trans, y, np.logspace(-3, 0, 100)
        )

        self.channel.send(Message((tmp_lambda * best_accuracy, best_accuracy),
                          "lambda", self), self.server)


class ServerFed2RC(Server):

    def __init__(self,
                 model: RidgeTorchModel,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 lr: float = 1.0,
                 tune_lambda: bool = False,
                 **kwargs: dict[str, Any]):
        super().__init__(model=None, test_set=test_set, clients=clients, **kwargs)
        self.hyper_params.update(tune_lambda=tune_lambda)
        self.seeds = []
        self.converged = False
        self.compression_k = 0

    def receive_client_models(self,
                              eligible: Iterable[Client],
                              **kwargs: dict[str, Any]) -> Generator[Any, None, None]:
        if not self.converged:
            for client in eligible:
                client_seeds = self.channel.receive(self, client, "seeds").payload
                yield client_seeds
        else:
            for client in eligible:
                client_Ab = self.channel.receive(self, client, "Ab").payload
                yield client_Ab

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message(self.converged, "converge", self), eligible)
        if not self.converged:
            self.channel.broadcast(Message(self.seeds, "seeds", self), eligible)

    def _compute_final_model(self, lam: float = 0.01) -> None:
        A_final = self.A + lam * np.eye(self.A.shape[0])
        W_final = np.linalg.solve(A_final, self.b)
        W_final = W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
        self.model = RidgeTorchModel(W_final.T, None, self.seeds)

    def aggregate(self,
                  eligible: Iterable[Client],
                  client_seeds_ab: Generator[Any, None, None]) -> None:
        if not self.converged:
            seeds = np.concatenate(list(client_seeds_ab), axis=0)
            seeds = np.unique(seeds)
            self.converged = (len(self.seeds) == len(seeds)) and np.all(self.seeds == seeds)
            self.seeds = seeds
            self._notify_track_item(item="seeds", value=len(self.seeds))
        else:
            A_global = np.zeros((len(self.seeds), len(self.seeds)))
            b_global = None

            for client_Ab in client_seeds_ab:
                A, b = client_Ab
                if isinstance(A, tuple):
                    U, L = A
                    self.compression_k = U.shape[1]
                    A = U @ L @ U.T
                A_global += A
                if b_global is None:
                    b_global = b
                else:
                    b_global += b
            self.A, self.b = A_global, b_global

            if not self.hyper_params.tune_lambda:
                self._compute_final_model(0.01)

            raise EarlyStopping(self.rounds)

    def finalize(self) -> None:

        if self.converged and self.hyper_params.tune_lambda:

            if self.compression_k > 0:
                U, L = compress_model(self.A, self.compression_k)
                self.channel.broadcast(Message(((U, L), self.b), "Ab", self), self.clients)
            else:
                self.channel.broadcast(Message((self.A, self.b), "Ab", self), self.clients)
            local_best_lambdas = np.zeros(len(self.clients))
            local_best_accs = np.zeros(len(self.clients))
            for client in track(self.clients, description="Validating lambda...", transient=True):
                client.tune_lambda()
                local_lam, local_acc = self.channel.receive(self, client, "lambda").payload
                local_best_accs[client.index] = local_acc
                local_best_lambdas[client.index] = local_lam

            lam = np.sum(local_best_lambdas)/np.sum(local_best_accs)
            self._compute_final_model(lam)

        if FlukeENV().get_eval_cfg().server:
            evals = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
            self._notify_evaluation(self.rounds + 1, "global", evals)

            for k,v in evals.items():
                self._notify_track_item(item=k, value=v)
        self._notify_finalize()


class Fed2RC(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return ClientFed2RC

    def get_server_class(self) -> type[Server]:
        return ServerFed2RC
