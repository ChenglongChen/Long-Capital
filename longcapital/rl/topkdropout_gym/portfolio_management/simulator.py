from typing import NamedTuple, Any

from qlib.backtest import get_exchange, create_account_instance
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.utils import TradeCalendarManager, CommonInfrastructure
from qlib.rl.reward import Reward
from qlib.rl.simulator import Simulator
from qlib.utils import init_instance_by_config


class PortfolioManagementState(NamedTuple):
    # NOTE:
    # - for avoiding recursive import
    # - typing annotations is not reliable
    from qlib.backtest.executor import BaseExecutor  # pylint: disable=C0415

    executor: BaseExecutor
    trade_calendar: TradeCalendarManager


class PortfolioManagementInitiateState(NamedTuple):
    start_time: str
    end_time: str


class PortfolioManagementSimulator(Simulator[None, PortfolioManagementState, TradeDecisionWO]):
    def __init__(self,
                 start_time: str,
                 end_time: str,
                 account: float = 1000000,
                 benchmark: str = "SH000300",
                 exchange_kwargs: Any = {},
                 initial: Any = None,
                 pos_type: str = "Position",
                 verbose: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(initial)
        self.verbose = verbose
        # Similar logic as qlib.backtest.get_strategy_executor, initialize trade executor

        # NOTE:
        # - for avoiding recursive import
        # - typing annotations is not reliable
        from qlib.backtest.executor import BaseExecutor  # pylint: disable=C0415

        trade_exchange = get_exchange(**exchange_kwargs)

        trade_account = create_account_instance(
            start_time=start_time,
            end_time=end_time,
            benchmark=benchmark,
            account=account,
            pos_type=pos_type,
        )

        self.common_infra = CommonInfrastructure(trade_account=trade_account, trade_exchange=trade_exchange)

        executor = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": self.verbose,
                "indicator_config": {
                    "show_indicator": self.verbose,
                },
                "common_infra": self.common_infra
            },
        }

        self.trade_executor = init_instance_by_config(executor, accept_types=BaseExecutor)
        self.trade_executor.reset(start_time=start_time, end_time=end_time)
        self.trade_executor.get_level_infra().trade_calendar

    def step(self, action: TradeDecisionWO) -> None:
        trade_decisions = action
        for _execute_result in self.trade_executor.collect_data(trade_decisions, level=0):
            if self.verbose:
                print("Execution result:", len(_execute_result))

    def get_state(self) -> PortfolioManagementState:
        return PortfolioManagementState(
            executor=self.trade_executor,
            trade_calendar=self.trade_executor.get_level_infra().trade_calendar
        )

    def done(self) -> bool:
        return self.trade_executor.finished()


class PortfolioReward(Reward[PortfolioManagementState]):
    def reward(self, simulator_state: PortfolioManagementState) -> float:
        # Use last_action to calculate reward. This is why it should be in the state.
        last_record = simulator_state.executor.trade_account.get_portfolio_metrics()[0].iloc[-1]
        relative_return = (last_record["return"] - last_record["bench"])
        if relative_return < 0:
            # adjust for negative return. As (1-x) * (1+x) < 1
            # If we want (1-a) * (1+b) = 1, a = 1-1/(1+b)
            relative_return = 1 - 1 / (1 + relative_return)
        relative_return *= 100
        return relative_return
