import numpy as np
import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.contrib.evaluate import risk_analysis
from qlib.rl.reward import Reward


def identity(x):
    return x


class ExecutionInformationRatioAndExcessReturnReward(Reward[TradeStrategyState]):
    def __init__(
        self,
        scale=1.0,
        ir_weight: float = 1.0,
        rr_weight: float = 1.0,
        excess_return: bool = True,
        log_rr: bool = False,
    ):
        self.scale = scale
        self.ir_weight = ir_weight
        self.rr_weight = rr_weight
        self.excess_weight = 1 if excess_return else 0
        self.rr_func = np.log1p if log_rr else identity

    def __str__(self):
        return "ExecutionInformationRatioAndExcessReturnReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        (
            portfolio_metrics,
            _,
        ) = state.trade_executor.trade_account.get_portfolio_metrics()
        if state.info["ready"] and len(portfolio_metrics) >= 2:
            analysis = risk_analysis(
                portfolio_metrics["return"]
                - portfolio_metrics["cost"]
                - self.excess_weight * portfolio_metrics["bench"]
            )
            ir = float(analysis.loc["information_ratio"])

            last_metrics = portfolio_metrics.iloc[-1]
            rr = (
                last_metrics["return"]
                - last_metrics["cost"]
                - self.excess_weight * last_metrics["bench"]
            )
            rr = self.rr_func(rr)
            reward = self.ir_weight * ir + self.rr_weight * rr
        return reward * self.scale


class ExecutionInformationRatioReward(ExecutionInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(ExecutionInformationRatioReward, self).__init__(
            scale=scale, ir_weight=1.0, rr_weight=0.0, excess_return=True, log_rr=log_rr
        )

    def __str__(self):
        return "ExecutionInformationRatioReward"


class ExecutionSharpRatioReward(ExecutionInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(ExecutionSharpRatioReward, self).__init__(
            scale=scale,
            ir_weight=1.0,
            rr_weight=0.0,
            excess_return=False,
            log_rr=log_rr,
        )

    def __str__(self):
        return "ExecutionSharpRatioReward"


class ExecutionExcessReturnReward(ExecutionInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(ExecutionExcessReturnReward, self).__init__(
            scale=scale, ir_weight=0.0, rr_weight=1.0, excess_return=True, log_rr=log_rr
        )

    def __str__(self):
        return "ExecutionExcessReturnReward"


class ExecutionReturnReward(ExecutionInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(ExecutionReturnReward, self).__init__(
            scale=scale,
            ir_weight=0.0,
            rr_weight=1.0,
            excess_return=False,
            log_rr=log_rr,
        )

    def __str__(self):
        return "ExecutionReturnReward"


class ExecutionExcessMeanVarianceReward(Reward[TradeStrategyState]):
    """Reward derived under the Mean-variance equivalence assumption.
    This reward is proved to maximize utility of final wealth, and can achieve better Sharp Ratio/Information Ratio
    than expected-profit maximization (where k=0).

    Reference:
    1. Machine Learning for Trading, https://cims.nyu.edu/~ritter/ritter2017machine.pdf
    2. Reinforcement Learning Applications in Real Time Trading
    """

    def __init__(self, scale=1.0, k=1e-4, excess_return=True, log_rr: bool = False):
        self.scale = scale
        self.k = k
        self.excess_weight = 1 if excess_return else 0
        self.rr_func = np.log1p if log_rr else identity

    def __str__(self):
        return "ExecutionExcessMeanVarianceReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        (
            portfolio_metrics,
            _,
        ) = state.trade_executor.trade_account.get_portfolio_metrics()

        if state.info["ready"] and len(portfolio_metrics):
            last_metrics = portfolio_metrics.iloc[-1]
            rr = (
                last_metrics["return"]
                - last_metrics["cost"]
                - self.excess_weight * last_metrics["bench"]
            )
            rr = self.rr_func(rr)
            reward = rr - 0.5 * self.k * (rr**2)
        return reward * self.scale


class ExecutionMeanVarianceReward(ExecutionExcessMeanVarianceReward):
    """Similar as ExecutionExcessMeanVarianceReward but use Return in reward."""

    def __init__(self, scale=1.0, k=1e-4, log_rr: bool = False):
        super(ExecutionMeanVarianceReward, self).__init__(
            scale=scale, k=k, excess_return=False, log_rr=log_rr
        )

    def __str__(self):
        return "ExecutionMeanVarianceReward"
