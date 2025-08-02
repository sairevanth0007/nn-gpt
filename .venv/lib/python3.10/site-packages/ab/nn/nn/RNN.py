import torch
import torch.nn as nn
from torch import Tensor


def supported_hyperparameters() -> set[str]:
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(
        self,
        in_shape: tuple,            # (S,)
        out_shape: tuple,           # (V,)
        prm: dict,
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        self.seq_len: int = in_shape[0]       # S
        self.vocab_size: int = out_shape[0]   # V
        self.hidden_size: int = 256
        self.embed_dim: int = 128

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.i2h = nn.Linear(self.embed_dim,   self.hidden_size, bias=False)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.vocab_size)

    def train_setup(self, prm: dict) -> None:
        self.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(),lr=prm["lr"], momentum=prm["momentum"],)

    def _step(self, token_idx: Tensor, h_t: Tensor) -> tuple[Tensor, Tensor]:
        x_t = self.embed(token_idx)                      # [B, E]
        h_t = torch.tanh(self.i2h(x_t) + self.h2h(h_t))  # [B, H]
        y_t = self.h2o(h_t)                              # [B, V]
        return y_t, h_t

    def forward(self, x_idx: Tensor, h_0: Tensor | None = None,) -> Tensor:
        B, S = x_idx.shape
        h_t = h_0 if h_0 is not None else torch.zeros(B, self.hidden_size, device=self.device)

        logits = []
        for t in range(S):
            y_t, h_t = self._step(x_idx[:, t], h_t)     # y_t: [B, V]
            logits.append(y_t)

        logits = torch.stack(logits, dim=1)             # [B, S, V]
        return logits.reshape(B * S, self.vocab_size)   # [B*S, V]

    def learn(self, train_loader) -> None:
        self.train()
        for x_idx, targets in train_loader:             # x_idx: [B, S] ; targets: [B, S]
            x_idx   = x_idx.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self(x_idx)                        # [B*S, V]
            loss = self.loss_fn(logits, targets.reshape(-1))  # targets: [B*S]

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
