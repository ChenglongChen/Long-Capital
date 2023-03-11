import logging
from pprint import pprint

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import get_module_logger
from qlib.workflow.record_temp import SignalRecord as QlibSignalRecord

from ..contrib.data.utils.neutralize import get_riskest_features, neutralize

logger = get_module_logger("workflow", logging.INFO)


class SignalRecord(QlibSignalRecord):
    """
    This is the Signal Record class with Neutralize. This class inherits the ``SignalRecord`` class.
    """

    def __init__(
        self,
        model=None,
        dataset=None,
        recorder=None,
        neutralize=False,
        riskiest_features_num=50,
        normalize=True,
    ):
        super().__init__(model=model, dataset=dataset, recorder=recorder)
        self.neutralize = neutralize
        self.riskiest_features_num = riskiest_features_num
        self.normalize = normalize
        self.riskiest_features = []

    def generate(self, **kwargs):
        super(SignalRecord, self).generate(**kwargs)
        if self.neutralize:
            df_train = self.dataset.prepare(
                "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
            )
            self.riskiest_features = get_riskest_features(
                df_train, N=self.riskiest_features_num
            )

            df_test = self.dataset.prepare(
                "test", col_set="feature", data_key=DataHandlerLP.DK_I
            )
            pred = self.recorder.load_object("pred.pkl")
            df_test["score"] = pred.values
            pred = neutralize(
                df=df_test,
                columns=["score"],
                neutralizers=self.riskiest_features,
                proportion=1.0,
                normalize=self.normalize,
                era_col="datetime",
            )
            self.save(**{"pred.pkl": pred})

            logger.info(
                f"Signal record 'pred.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
            )
            # print out results
            pprint(
                f"The following are prediction results of the {type(self.model).__name__} model."
            )
            pprint(pred.head(5))
