import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.rl.reward import Reward


class TradeStrategyReward(Reward[TradeStrategyState]):
    def __init__(self, scale=1.0):
        self.scale = scale

    def reward(self, state: TradeStrategyState) -> float:
        (
            portfolio_metrics,
            _,
        ) = state.trade_executor.trade_account.get_portfolio_metrics()
        reward = 0.0
        if len(portfolio_metrics):
            last_metrics = portfolio_metrics.iloc[-1]
            reward = float(
                last_metrics["return"] - last_metrics["bench"] - last_metrics["cost"]
            )
        return reward * self.scale
