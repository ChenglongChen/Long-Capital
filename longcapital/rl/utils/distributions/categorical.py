import torch


class Categorical(torch.distributions.Categorical):
    """Sample ranking index according to Softmax distribution"""

    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)
        self._event_shape = (self._param.size()[-1],)

    def sample(self, sample_shape=torch.Size(), replacement=False):
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, self._num_events, replacement)
        return samples_2d

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.logits.gather(-1, value.long())
