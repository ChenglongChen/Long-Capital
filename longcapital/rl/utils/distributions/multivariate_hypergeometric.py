import torch
from longcapital.utils.constant import EPS


class MultivariateHypergeometric(torch.distributions.Categorical):
    """Sample ranking index according to Softmax distribution"""

    def __init__(self, probs=None, logits=None, validate_args=None, topk=None):
        super().__init__(probs, logits, validate_args)
        self._topk = topk if topk else self._num_events
        self._event_shape = (self._topk,)

    def sample(self, sample_shape=torch.Size(), replacement=False):
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, self._topk, replacement)
        return samples_2d

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.prob(value).log()

    def prob(self, value):
        """
        P(x1,x2,...,xn) = (P(x1)/1) * (P(x2)/(1-P(x1))) * ... * (P(xn)/(1-P(x1)-P(x2)-...-P(xn-1)))
                           = prod_{k=1}^{n} P(xk)/(1-sum_{i=1}^{k-1} P(xi))

        log P(x1,x2,...,xn) = sum_{k=1}^{n} {log P(xk) - log (1-sum_{i=1}^{k-1} P(xi))}
        """
        numerator = self.probs.gather(-1, value.long())
        denominator = torch.hstack(
            [torch.zeros_like(numerator[:, :1]), numerator[:, :-1]]
        )
        denominator = 1 - torch.cumsum(denominator, dim=1)
        p = torch.prod(numerator / (denominator + EPS), dim=1)
        return p
