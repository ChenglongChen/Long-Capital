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


class EpisodeInformationRatioReward(Reward[TradeStrategyState]):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __str__(self):
        return "EpisodeInformationRatioReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        if state.trade_executor.finished():
            (
                portfolio_metrics,
                _,
            ) = state.trade_executor.trade_account.get_portfolio_metrics()
            if len(portfolio_metrics) >= 2:
                analysis = risk_analysis(
                    portfolio_metrics["return"]
                    - portfolio_metrics["bench"]
                    - portfolio_metrics["cost"]
                )
                reward = float(analysis.loc["information_ratio"])
        return reward * self.scale


class ExecutionInformationRatioReward(Reward[TradeStrategyState]):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __str__(self):
        return "ExecutionInformationRatioReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        (
            portfolio_metrics,
            _,
        ) = state.trade_executor.trade_account.get_portfolio_metrics()
        if len(portfolio_metrics) >= 2:
            analysis = risk_analysis(
                portfolio_metrics["return"]
                - portfolio_metrics["bench"]
                - portfolio_metrics["cost"]
            )
            reward = float(analysis.loc["information_ratio"])
        return reward * self.scale


class ExcessExecutionInformationRatioReward(Reward[TradeStrategyState]):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __str__(self):
        return "ExcessExecutionInformationRatioReward"

    def reward(self, state: TradeStrategyState) -> float:
        reward = 0.0
        (
            portfolio_metrics,
            _,
        ) = state.trade_executor.trade_account.get_portfolio_metrics()
        if len(portfolio_metrics) > 2:
            analysis0 = risk_analysis(
                portfolio_metrics["return"].iloc[:-1]
                - portfolio_metrics["bench"].iloc[:-1]
                - portfolio_metrics["cost"].iloc[:-1]
            )
            analysis1 = risk_analysis(
                portfolio_metrics["return"]
                - portfolio_metrics["bench"]
                - portfolio_metrics["cost"]
            )
            reward = float(analysis1.loc["information_ratio"]) - float(
                analysis0.loc["information_ratio"]
            )
        return reward * self.scale
