import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.contrib.evaluate import risk_analysis
from qlib.rl.reward import Reward


class EpisodeInformationRatioAndExcessReturnReward(Reward[TradeStrategyState]):
    def __init__(
        self,
        scale=1.0,
        ir_weight: float = 1.0,
        rr_weight: float = 1.0,
        excess_return=True,
    ):
        self.scale = scale
        self.ir_weight = ir_weight
        self.rr_weight = rr_weight
        self.excess_weight = 1 if excess_return else 0

    def __str__(self):
        return "EpisodeInformationRatioAndExcessReturnReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        if state.trade_executor.finished():
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
            reward = self.ir_weight * ir + self.rr_weight * rr
        return reward * self.scale


class EpisodeInformationRatioReward(EpisodeInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0):
        super(EpisodeInformationRatioReward, self).__init__(
            scale=scale, ir_weight=1.0, rr_weight=0.0, excess_return=True
        )

    def __str__(self):
        return "EpisodeInformationRatioReward"


class EpisodeSharpRatioReward(EpisodeInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0):
        super(EpisodeSharpRatioReward, self).__init__(
            scale=scale, ir_weight=1.0, rr_weight=0.0, excess_return=False
        )

    def __str__(self):
        return "EpisodeSharpRatioReward"


class EpisodeExcessReturnReward(EpisodeInformationRatioAndExcessReturnReward):
    def __init__(self, scale=1.0):
        super(EpisodeExcessReturnReward, self).__init__(
            scale=scale, ir_weight=0.0, rr_weight=1.0, excess_return=True
        )

    def __str__(self):
        return "EpisodeExcessReturnReward"


class EpisodeReturnReward(EpisodeInformationRatioAndExcessReturnReward):
    """Same as EpisodeExcessReturnReward"""

    def __init__(self, scale=1.0):
        super(EpisodeReturnReward, self).__init__(
            scale=scale, ir_weight=0.0, rr_weight=1.0, excess_return=False
        )

    def __str__(self):
        return "EpisodeReturnReward"
