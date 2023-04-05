from typing import Any, Dict

from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.rl.aux_info import AuxiliaryInfoCollector


class ImitationLabelCollector(AuxiliaryInfoCollector[TradeStrategyState, Any]):
    def collect(self, state: TradeStrategyState) -> Dict[str, Any]:
        return {
            "label": state.label,
            "topk": state.initial_state.topk,
            "stock_num": state.initial_state.stock_num,
        }
