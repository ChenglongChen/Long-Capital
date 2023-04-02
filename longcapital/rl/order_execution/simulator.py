from typing import Any

import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.state import (
    TradeStrategyInitialState,
    TradeStrategyState,
)
from longcapital.rl.order_execution.strategy import BaseTradeStrategy
from longcapital.rl.order_execution.utils import random_daterange
from qlib.backtest import create_account_instance, get_exchange
from qlib.backtest.utils import CommonInfrastructure
from qlib.rl.simulator import Simulator
from qlib.utils import init_instance_by_config


class TradeStrategySimulator(
    Simulator[TradeStrategyInitialState, TradeStrategyState, Any]
):
    def __init__(
        self,
        trade_strategy: BaseTradeStrategy,
        initial_state: TradeStrategyInitialState,
        benchmark: str = "SH000300",
        account: float = 100000000,
        pos_type: str = "Position",
        exchange_kwargs: Any = {},
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(initial_state)
        self.verbose = verbose
        self.trade_strategy = trade_strategy

        # NOTE:
        # - for avoiding recursive import
        # - typing annotations is not reliable
        from qlib.backtest.executor import BaseExecutor  # pylint: disable=C0415

        trade_exchange = get_exchange(**exchange_kwargs)

        # config start_time and end_time for trading
        if initial_state.sample_date:
            start_time, end_time = random_daterange(
                initial_state.start_time, initial_state.end_time
            )
        else:
            start_time, end_time = initial_state.start_time, initial_state.end_time

        trade_account = create_account_instance(
            start_time=start_time,
            end_time=end_time,
            benchmark=benchmark,
            account=account,
            pos_type=pos_type,
        )

        self.common_infra = CommonInfrastructure(
            trade_account=trade_account, trade_exchange=trade_exchange
        )

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
                "common_infra": self.common_infra,
            },
        }

        self.trade_executor = init_instance_by_config(
            executor, accept_types=BaseExecutor
        )

        # move necessary steps forward to the first tradable time
        self.trade_executor.reset(start_time=start_time, end_time=end_time)
        if initial_state.skip_nontradable_start_time:
            trade_len = self.trade_executor.trade_calendar.get_trade_len()
            trade_start_time = start_time
            for trade_step in range(trade_len):
                trade_start_time = self.trade_executor.trade_calendar.get_step_time(
                    trade_step, shift=0
                )[0]
                pred_start_time = self.trade_executor.trade_calendar.get_step_time(
                    trade_step, shift=1
                )[0]
                if pred_start_time >= self.trade_executor.trade_calendar.start_time:
                    break
            start_time = trade_start_time
            end_time = self.trade_executor.trade_calendar.end_time
            self.trade_executor.reset(start_time=start_time, end_time=end_time)
        start_time = self.trade_executor.trade_calendar.start_time
        end_time = self.trade_executor.trade_calendar.end_time
        print(f"start_time: {start_time}, end_time: {end_time}")

        # reset for trade_strategy
        self.trade_strategy.reset_common_infra(self.common_infra)
        self.trade_strategy.reset_level_infra(self.trade_executor.get_level_infra())
        self.trade_strategy.reset_initial_state(initial_state)

    def step(self, action: Any) -> None:
        if action.ready:
            trade_decisions = self.trade_strategy.generate_trade_decision(action=action)
            for _execute_result in self.trade_executor.collect_data(
                trade_decisions, level=0
            ):
                if self.verbose:
                    print("Execution result:", len(_execute_result))

    def get_state(self) -> TradeStrategyState:
        return TradeStrategyState(
            trade_executor=self.trade_executor,
            trade_strategy=self.trade_strategy,
            feature=self.trade_strategy.get_feature(),
            initial_state=self.trade_strategy.initial_state,
        )

    def done(self) -> bool:
        return self.trade_executor.finished()
