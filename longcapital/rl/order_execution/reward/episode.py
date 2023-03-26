import numpy as np
import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.contrib.evaluate import risk_analysis
from qlib.rl.reward import Reward


def identity(x):
    return x


class EpisodeInformationRatioAndExcessReturnReward(Reward[TradeStrategyState]):
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
        return "EpisodeInformationRatioAndExcessReturnReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        if state.info["ready"] and state.trade_executor.finished():
            (
                portfolio_metrics,
                _,
            ) = state.trade_executor.trade_account.get_portfolio_metrics()
            analysis = risk_analysis(
                portfolio_metrics["return"]
                - portfolio_metrics["cost"]
                - self.excess_weight * portfolio_metrics["bench"]
            )
            ir = float(analysis.loc["information_ratio"])

            # for rr, excess_return will need minus rr of benchmark during the whole period, which
            # is a constant and will not make any difference
            position = state.trade_executor.trade_account.current_position
            rr = float(position.calculate_value()) / float(position.init_cash)
            rr = self.rr_func(rr)
            reward = self.ir_weight * ir + self.rr_weight * rr
        return reward * self.scale


class EpisodeInformationRatioReward(EpisodeInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(EpisodeInformationRatioReward, self).__init__(
            scale=scale, ir_weight=1.0, rr_weight=0.0, excess_return=True, log_rr=log_rr
        )

    def __str__(self):
        return "EpisodeInformationRatioReward"


class EpisodeSharpRatioReward(EpisodeInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(EpisodeSharpRatioReward, self).__init__(
            scale=scale,
            ir_weight=1.0,
            rr_weight=0.0,
            excess_return=False,
            log_rr=log_rr,
        )

    def __str__(self):
        return "EpisodeSharpRatioReward"


class EpisodeExcessReturnReward(EpisodeInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0, log_rr: bool = False):
        super(EpisodeExcessReturnReward, self).__init__(
            scale=scale, ir_weight=0.0, rr_weight=1.0, excess_return=True, log_rr=log_rr
        )

    def __str__(self):
        return "EpisodeExcessReturnReward"


class EpisodeReturnReward(EpisodeInformationRatioAndExcessReturnReward):
    """Same as EpisodeExcessReturnReward"""

    def __init__(self, scale=1.0, log_rr: bool = False):
        super(EpisodeReturnReward, self).__init__(
            scale=scale,
            ir_weight=0.0,
            rr_weight=1.0,
            excess_return=False,
            log_rr=log_rr,
        )

    def __str__(self):
        return "EpisodeReturnReward"
