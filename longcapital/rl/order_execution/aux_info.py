from typing import Any, Dict

from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.rl.aux_info import AuxiliaryInfoCollector


class ImitationSignalCollector(AuxiliaryInfoCollector[TradeStrategyState, Any]):
    def __init__(self, stock_num: int, signal_key: str = "signal"):
        self.stock_num = stock_num
        self.signal_key = signal_key

    def collect(self, state: TradeStrategyState) -> Dict[str, Any]:
        if state.feature is None:
            return {}

        signal = state.feature[("feature", self.signal_key)][: self.stock_num].values
        return {"signal": signal}
