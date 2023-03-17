from qlib.contrib.data.handler import Alpha158 as QlibAlpha158
from qlib.contrib.data.handler import DataHandlerLP, check_transform_proc

from ...data.dataset.processor import CSBucketizeLabel

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha158(QlibAlpha158):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        loss_type="mse",
        curr_label_price_expr="$open",
        next_label_price_expr="$open",
        days_ahead=2,
        bucket_size=10,
        include_volume=False,
        **kwargs,
    ):
        self.loss_type = loss_type
        self.curr_label_price_expr = curr_label_price_expr
        self.next_label_price_expr = next_label_price_expr
        self.days_ahead = days_ahead
        self.bucket_size = bucket_size
        self.include_volume = include_volume

        infer_processors = check_transform_proc(
            infer_processors, fit_start_time, fit_end_time
        )
        learn_processors = check_transform_proc(
            learn_processors, fit_start_time, fit_end_time
        )
        if self.loss_type in ["lambdarank"]:
            learn_processors.append(CSBucketizeLabel(bucket_size=bucket_size))

        fields, names = self.get_feature_config()
        other_fields, other_names = kwargs.pop("feature", ([], []))
        fields += other_fields
        names += other_names

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
            },
        }
        super(QlibAlpha158, self).__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        if self.include_volume:
            conf.update({"volume": {"windows": [0, 1, 2, 3, 4, 10, 20, 30, 60]}})
        return self.parse_config_to_fields(conf)

    def get_label_config(self):
        if self.loss_type in ["mse", "lambdarank"]:
            return (
                [
                    f"Ref({self.next_label_price_expr}, -{self.days_ahead})/Ref({self.curr_label_price_expr}, -1) - 1"
                ],
                ["LABEL0"],
            )
        elif self.loss_type in ["mse_log"]:
            return (
                [
                    f"Log(Ref({self.next_label_price_expr}, -{self.days_ahead})/Ref({self.curr_label_price_expr}, -1))"
                ],
                ["LABEL0"],
            )
        elif self.loss_type in ["binary"]:
            return (
                [
                    f"If(Gt(Ref({self.next_label_price_expr}, -{self.days_ahead}), Ref({self.curr_label_price_expr}, -1)), 1, 0)"
                ],
                ["LABEL0"],
            )
        else:
            raise NotImplementedError(f"Not supported loss type: {self.loss_type}")
