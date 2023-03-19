from typing import Any, Dict

from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.rl.aux_info import AuxiliaryInfoCollector


class ImitationLabelCollector(AuxiliaryInfoCollector[TradeStrategyState, Any]):
    def __init__(self, stock_num: int, label_key: str = "label"):
        self.stock_num = stock_num
        self.label_key = label_key

    def collect(self, state: TradeStrategyState) -> Dict[str, Any]:
        if state.feature is None:
            return {}
        label_key = (
            ("label", "LABEL0")
            if self.label_key == "label"
            else ("feature", self.label_key)
        )
        label = state.feature[label_key][: self.stock_num].values
        return {self.label_key: label}
