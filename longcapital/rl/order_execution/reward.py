import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.contrib.evaluate import risk_analysis
from qlib.rl.reward import Reward


class ExcessReturnReward(Reward[TradeStrategyState]):
    def __init__(self, scale=1.0):
        self.scale = scale

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        (
            portfolio_metrics,
            _,
        ) = state.trade_executor.trade_account.get_portfolio_metrics()

        if len(portfolio_metrics):
            last_metrics = portfolio_metrics.iloc[-1]
            reward = float(
                last_metrics["return"] - last_metrics["bench"] - last_metrics["cost"]
            )
        return reward * self.scale

    def __str__(self):
        return "ExcessReturnReward"


class InformationRatioReward(Reward[TradeStrategyState]):
    def __init__(self, scale=1.0):
        self.scale = scale

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        if state.trade_executor.finished():
            (
                portfolio_metrics,
                _,
            ) = state.trade_executor.trade_account.get_portfolio_metrics()

            analysis = risk_analysis(
                portfolio_metrics["return"]
                - portfolio_metrics["bench"]
                - portfolio_metrics["cost"]
            )
            reward = float(analysis.loc["information_ratio"])
        return reward * self.scale

    def __str__(self):
        return "InformationRatioReward"
