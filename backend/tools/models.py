"""
backend/tools/models.py
=========================
PyTorch model definitions for the Adaptive ML Experiment Agent.

NOTE on naming: this module lives in ``backend/tools`` (not
``backend/models``) because ``backend/models`` is already the established
package for Pydantic data contracts (see ``backend/models/__init__.py``).
Putting the PyTorch ``nn.Module`` here avoids a name collision between "the
application's typed data models" and "a neural network model" — two
different meanings of "model" that the codebase deliberately keeps apart.

TabularMLP
    Single-hidden-layer feed-forward network, generalized to any tabular
    dataset via ``input_dim``/``output_dim`` derived from a
    ``DatasetProfile``. Replaces the old MNIST-specific ``MNISTClassifier``
    — see DESIGN_REVIEW_CHANGES.md's "Architecture Revision" entry.

Requirements
------------
3.3  mlp model: configurable hidden_size and dropout, dataset-agnostic dims
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Single-hidden-layer feed-forward network for tabular data.

    Works for both classification and regression:
    - classification: ``output_dim`` = n_classes; ``forward`` returns raw
      logits (not softmax probabilities) — ``nn.CrossEntropyLoss`` applies
      ``log_softmax`` internally, so applying softmax here would feed a
      doubly-softmaxed input into the loss. Predictions use ``argmax`` on
      the logits directly, which softmax would not change.
    - regression: ``output_dim`` = 1; ``forward`` returns a single scalar
      per row (squeezed from shape ``(batch, 1)`` to ``(batch,)``) so it
      matches the shape ``nn.MSELoss`` expects against a 1-D target.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        if self._output_dim == 1:
            return out.squeeze(-1)
        return out
