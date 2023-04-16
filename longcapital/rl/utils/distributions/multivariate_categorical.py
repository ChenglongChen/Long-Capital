import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution


class MultivariateCategorical(Distribution):
    def __init__(self, nvec, probs=None, logits=None, validate_args=None):
        nvec = list(nvec)
        dims = np.cumsum([0] + nvec)
        if probs is not None:
            self._dists = [
                Categorical(probs=probs[:, i:j]) for i, j in zip(dims[:-1], dims[1:])
            ]
        elif logits is not None:
            self._dists = [
                Categorical(logits=logits[:, i:j]) for i, j in zip(dims[:-1], dims[1:])
            ]
        else:
            raise ValueError("probs and logits can not be both None.")
        batch_shape = self._dists[0].batch_shape
        super(MultivariateCategorical, self).__init__(batch_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        sample = torch.stack([d.sample(sample_shape) for d in self._dists], dim=1)
        return sample

    def log_prob(self, value):
        log_prob = torch.stack(
            [d.log_prob(value[:, i]) for i, d in enumerate(self._dists)], dim=1
        )
        return log_prob.sum(1)

    def entropy(self):
        entropy = torch.stack([d.entropy() for d in self._dists], dim=1)
        return entropy.sum(1)

    def argmax(self):
        return torch.stack([d.logits.argmax(1) for d in self._dists], dim=1)
