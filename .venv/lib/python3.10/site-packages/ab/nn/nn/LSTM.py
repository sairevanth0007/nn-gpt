import torch
import torch.nn as nn


def supported_hyperparameters() -> set[str]:
    return {"lr", "momentum"}


class Net(nn.Module):
    def __init__(
        self,
        in_shape: tuple,                       # (S,)
        out_shape: tuple,                      # (V,)
        prm: dict,
        device: torch.device,
    ):
        super().__init__()
        self.device: torch.device = device

        self.seq_len: int   = in_shape[0]      # S
        self.vocab_size: int = out_shape[0]    # V

        self.embed_dim:  int = 128
        self.hidden_size: int = 256
        self.num_layers: int  = 2

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm  = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.h2o = nn.Linear(self.hidden_size, self.vocab_size)

    def train_setup(self, prm: dict):
        self.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm["lr"], momentum=prm["momentum"],)

    def forward(self, x_idx: torch.Tensor, h0: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, S = x_idx.shape
        emb = self.embed(x_idx)                # [B,S,E]

        if h0 is None:
            h0 = self.init_zero_hidden(B)

        out, _ = self.lstm(emb, h0)            # [B,S,H]
        logits = self.h2o(out)                 # [B,S,V]
        return logits.reshape(B * S, self.vocab_size)

    def learn(self, train_loader):
        self.train()
        for x_idx, targets in train_loader:    # x_idx:[B,S] targets:[B,S]
            x_idx = x_idx.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self(x_idx)               # [B*S,V]
            loss = self.loss_fn(logits, targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()

    def init_zero_hidden(self, batch: int):
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=self.device)
        return h0, c0
