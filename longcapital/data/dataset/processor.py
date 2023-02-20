from qlib.data.dataset.processor import get_group_columns, Processor
from ..ops import *


class Fillna(Processor):
    """Process NaN
    Original implementation of Qlib has issue for MultiIndex fillna
    """

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)
            df[cols] = df[cols].fillna(self.fill_value)
        return df


class CSBucketizeLabel(Processor):
    """
    Cross Sectional Bucketize.
    """

    def __init__(self, fields_group="label", bucket_size=10):
        self.fields_group = fields_group
        self.bucket_size = bucket_size

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime", group_keys=False).apply(
            lambda x: pd.Series(pd.qcut(x.values.flatten(), self.bucket_size, labels=False), index=x.index)
        )
        return df


class AlphaProcessor(Processor):
    def __init__(self):
        pass

    def register_alpha(self, df, res):
        res.values[~np.isfinite(res)] = np.nan
        df.insert(len(df.columns) - 1, ("feature", self.__class__.__name__), res)
        return df


class GTJAAlpha1(AlphaProcessor):
    """(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))"""

    def __call__(self, df):
        res = -1 * Corr(
            Rank(Delta(Log(Ref(df, "volume")), 1)),
            Rank((Ref(df, "close") - Ref(df, "open")) / Ref(df, "open")),
            6
        )
        return self.register_alpha(df, res)


class GTJAAlpha2(AlphaProcessor):
    """(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))"""

    def __call__(self, df):
        res = (-1 * Delta((((Ref(df,"close") - Ref(df,"low")) - (Ref(df,"high") - Ref(df,"close"))) / (Ref(df,"high") - Ref(df,"low"))), 1))
        return self.register_alpha(df, res)


class GTJAAlpha3(AlphaProcessor):
    """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)"""

    def __call__(self, df):
        res = Sum(If(Eq(Ref(df,"close"),Delay(Ref(df,"close"),1)), 0, Ref(df,"close")-(If(Gt(Ref(df,"close"),Delay(Ref(df,"close"),1)),Less(Ref(df,"low"),Delay(Ref(df,"close"),1)),Greater(Ref(df,"high"),Delay(Ref(df,"close"),1))))),6)
        return self.register_alpha(df, res)


class GTJAAlpha4(AlphaProcessor):
    """((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <
    ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /
    MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))"""
    def __call__(self, df):
        res = If(
            Lt(((Sum(Ref(df,"close"), 8) / 8) + Std(Ref(df,"close"), 8)), (Sum(Ref(df,"close"), 2) / 2)),
            (-1 * 1),
            If(
                Lt((Sum(Ref(df,"close"), 2) / 2), ((Sum(Ref(df,"close"), 8) / 8) - Std(Ref(df,"close"), 8))),
               1,
               If(
                   Or(Lt(1, (Ref(df,"volume") / Mean(Ref(df,"volume"),20))), ((Ref(df,"volume") /Mean(Ref(df,"volume"),20)) == 1)),
                   1,
                   (-1 * 1)
               )
            )
        )
        return self.register_alpha(df, res)


class GTJAAlpha5(AlphaProcessor):
    """(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))"""
    def __call__(self, df):
        res = (-1 * Max(Corr(Tsrank(Ref(df, "volume"), 5), Tsrank(Ref(df, "high"), 5), 5), 3))
        return self.register_alpha(df, res)


class GTJAAlpha6(AlphaProcessor):
    """(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)"""
    def __call__(self, df):
        res = (Rank(Sign(Delta((((Ref(df, "open") * 0.85 + Ref(df, "high") * 0.15))), 4)))* -1)
        return self.register_alpha(df, res)


class GTJAAlpha7(AlphaProcessor):
    """((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))"""
    def __call__(self, df):
        res = ((Rank(Max((Ref(df,"vwap") - Ref(df,"close")), 3)) + Rank(Min((Ref(df,"vwap") - Ref(df,"close")), 3))) * Rank(Delta(Ref(df,"volume"), 3)))
        return self.register_alpha(df, res)


class GTJAAlpha8(AlphaProcessor):
    """RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)"""
    def __call__(self, df):
        res = Rank(Delta(((((Ref(df,"high") + Ref(df,"low")) / 2) * 0.2) + (Ref(df,"vwap") * 0.8)), 4) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha9(AlphaProcessor):
    """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)"""
    def __call__(self, df):
        res = Sma(((Ref(df,"high")+Ref(df,"low"))/2-(Delay(Ref(df,"high"),1)+Delay(Ref(df,"low"),1))/2)*(Ref(df,"high")-Ref(df,"low"))/Ref(df,"volume"),7,2)
        return self.register_alpha(df, res)


class GTJAAlpha10(AlphaProcessor):
    """(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))"""
    def __call__(self, df):
        ret = Ret(df)
        res = (Rank(Max(If(Lt(ret, 0), Std(ret, 20), Ref(df,"close"))**2,5)))
        return self.register_alpha(df, res)


class GTJAAlpha11(AlphaProcessor):
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)"""
    def __call__(self, df):
        res = Sum(((Ref(df,"close")-Ref(df,"low"))-(Ref(df,"high")-Ref(df,"close")))/(Ref(df,"high")-Ref(df,"low"))*Ref(df,"volume"),6)
        return self.register_alpha(df, res)


class GTJAAlpha12(AlphaProcessor):
    """(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))"""
    def __call__(self, df):
        res = (Rank((Ref(df,"open") - (Sum(Ref(df,"vwap"), 10) / 10)))) * (-1 * (Rank(Abs((Ref(df,"close") - Ref(df,"vwap"))))))
        return self.register_alpha(df, res)


class GTJAAlpha13(AlphaProcessor):
    """(((HIGH * LOW)^0.5) - VWAP)"""
    def __call__(self, df):
        res = (((Ref(df,"high") * Ref(df,"low"))**0.5) - Ref(df,"vwap"))
        return self.register_alpha(df, res)


class GTJAAlpha14(AlphaProcessor):
    """CLOSE-DELAY(CLOSE,5)"""
    def __call__(self, df):
        res = Ref(df,"close")-Delay(Ref(df,"close"),5)
        return self.register_alpha(df, res)


class GTJAAlpha15(AlphaProcessor):
    """OPEN/DELAY(CLOSE,1)-1"""
    def __call__(self, df):
        res = Ref(df,"open")/Delay(Ref(df,"close"),1)-1
        return self.register_alpha(df, res)


class GTJAAlpha16(AlphaProcessor):
    """(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))"""
    def __call__(self, df):
        res = (-1 * Max(Rank(Corr(Rank(Ref(df,"volume")), Rank(Ref(df,"vwap")), 5)), 5))
        return self.register_alpha(df, res)


class GTJAAlpha17(AlphaProcessor):
    """RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)"""
    def __call__(self, df):
        res = Rank((Ref(df,"vwap") - Max(Ref(df,"vwap"), 15)))**Delta(Ref(df,"close"), 5)
        return self.register_alpha(df, res)


class GTJAAlpha18(AlphaProcessor):
    """CLOSE/DELAY(CLOSE,5)"""
    def __call__(self, df):
        res = Ref(df,"close")/Delay(Ref(df,"close"),5)
        return self.register_alpha(df, res)


class GTJAAlpha19(AlphaProcessor):
    """(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))"""
    def __call__(self, df):
        res = If(
            Lt(Ref(df,"close"),Delay(Ref(df,"close"),5)),
            (Ref(df,"close")-Delay(Ref(df,"close"),5))/Delay(Ref(df,"close"),5),
            If(
                Eq(Ref(df,"close"),Delay(Ref(df,"close"),5)),
                0,
                (Ref(df,"close")-Delay(Ref(df,"close"),5))/Ref(df,"close")
            )
        )
        return self.register_alpha(df, res)


class GTJAAlpha20(AlphaProcessor):
    """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100"""
    def __call__(self, df):
        res = (Ref(df,"close")-Delay(Ref(df,"close"),6))/Delay(Ref(df,"close"),6)*100
        return self.register_alpha(df, res)


class GTJAAlpha21(AlphaProcessor):
    """REGBETA(MEAN(CLOSE,6),SEQUENCE(6))"""
    def __call__(self, df):
        res = Regbeta(Mean(Ref(df,"close"),6),Sequence(6),6)
        return self.register_alpha(df, res)


class GTJAAlpha22(AlphaProcessor):
    """SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)"""
    def __call__(self, df):
        res = Sma(((Ref(df,"close")-Mean(Ref(df,"close"),6))/Mean(Ref(df,"close"),6)-Delay((Ref(df,"close")-Mean(Ref(df,"close"),6))/Mean(Ref(df,"close"),6),3)),12,1)
        return self.register_alpha(df, res)


class GTJAAlpha23(AlphaProcessor):
    """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1
    )+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100"""
    def __call__(self, df):
        res = (
                Sma(If(Gt(Ref(df,"close"), Delay(Ref(df,"close"),1)),Std(Ref(df,"close"), 20),0),20,1)/
                (
                      Sma(If(Gt(Ref(df,"close"), Delay(Ref(df,"close"),1)), Std(Ref(df,"close"),20), 0),20,1)+
                      Sma(If(Le(Ref(df,"close"), Delay(Ref(df,"close"),1)), Std(Ref(df,"close"),20), 0),20,1)
                )
              )*100
        return self.register_alpha(df, res)


class GTJAAlpha24(AlphaProcessor):
    """SMA(CLOSE-DELAY(CLOSE,5),5,1)"""
    def __call__(self, df):
        res = Sma(Ref(df,"close")-Delay(Ref(df,"close"),5),5,1)
        return self.register_alpha(df, res)


class GTJAAlpha25(AlphaProcessor):
    """((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))"""
    def __call__(self, df):
        ret = Ret(df)
        res = ((-1 * Rank((Delta(Ref(df, "close"), 7) * (1 - Rank(Decaylinear((Ref(df, "volume") / Mean(Ref(df, "volume"),20)), 9)))))) * (1 + Rank(Sum(ret, 250))))
        return self.register_alpha(df, res)


class GTJAAlpha32(AlphaProcessor):
    """(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))"""

    def __call__(self, df):
        res = (-1 * Sum(Rank(Corr(Rank(Ref(df, "high")), Rank(Ref(df, "volume")), 3)), 3))
        return self.register_alpha(df, res)


class GTJAAlpha33(AlphaProcessor):
    """((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))"""

    def __call__(self, df):
        ret = Ret(df)
        res = ((((-1 * Min(Ref(df, "low"), 5)) + Delay(Min(Ref(df, "low"), 5), 5)) * Rank(
            ((Sum(ret, 240) - Sum(ret, 20)) / 220))) * Tsrank(Ref(df, "volume"), 5))
        return self.register_alpha(df, res)


class GTJAAlpha35(AlphaProcessor):
    """(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +
    (OPEN *0.35)), 17),7))) * -1)"""

    def __call__(self, df):
        res = (Less(Rank(Decaylinear(Delta(Ref(df, "open"), 1), 15)), Rank(
            Decaylinear(Corr((Ref(df, "volume")), ((Ref(df, "open") * 0.65) + (Ref(df, "open") * 0.35)), 17), 7))) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha36(AlphaProcessor):
    """RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)"""
    def __call__(self, df):
        res = Rank(Sum(Corr(Rank(Ref(df,"volume")), Rank(Ref(df,"vwap")), 6), 2))
        return self.register_alpha(df, res)


class GTJAAlpha37(AlphaProcessor):
    """(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))"""
    def __call__(self, df):
        ret = Ret(df)
        res = (-1 * Rank(((Sum(Ref(df,"open"), 5) * Sum(ret, 5)) - Delay((Sum(Ref(df,"open"), 5) * Sum(ret, 5)), 10))))
        return self.register_alpha(df, res)


class GTJAAlpha39(AlphaProcessor):
    """((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),
    SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)"""

    def __call__(self, df):
        res = ((Rank(Decaylinear(Delta((Ref(df, "close")), 2), 8)) - Rank(Decaylinear(
            Corr(((Ref(df, "vwap") * 0.3) + (Ref(df, "open") * 0.7)), Sum(Mean(Ref(df, "volume"), 180), 37), 14),
            12))) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha41(AlphaProcessor):
    """(RANK(MAX(DELTA((VWAP), 3), 5))* -1)"""
    def __call__(self, df):
        res = (Rank(Max(Delta((Ref(df,"vwap")), 3), 5))* -1)
        return self.register_alpha(df, res)


class GTJAAlpha42(AlphaProcessor):
    """((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))"""
    def __call__(self, df):
        res = ((-1 * Rank(Std(Ref(df,"high"), 10))) * Corr(Ref(df,"high"), Ref(df,"volume"), 10))
        return self.register_alpha(df, res)


class GTJAAlpha45(AlphaProcessor):
    """(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))"""
    def __call__(self, df):
        res = (Rank(Delta((((Ref(df, "close") * 0.6) + (Ref(df, "open") *0.4))), 1)) * Rank(Corr(Ref(df, "vwap"), Mean(Ref(df, "volume"),150), 15)))
        return self.register_alpha(df, res)


class GTJAAlpha48(AlphaProcessor):
    """(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) +
    SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))"""
    def __call__(self, df):
        res = (-1*((Rank(((Sign((Ref(df, "close") - Delay(Ref(df, "close"), 1))) + Sign((Delay(Ref(df, "close"), 1) - Delay(Ref(df, "close"), 2)))) + Sign((Delay(Ref(df, "close"), 2) - Delay(Ref(df, "close"), 3)))))) * Sum(Ref(df, "volume"), 5)) / Sum(Ref(df, "volume"), 20))
        return self.register_alpha(df, res)


class GTJAAlpha54(AlphaProcessor):
    """(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))"""
    def __call__(self, df):
        res = (-1 * Rank((Std(Abs(Ref(df,"close") - Ref(df,"open")), 10) + (Ref(df,"close") - Ref(df,"open"))) + Corr(Ref(df,"close"), Ref(df,"open"),10)))
        return self.register_alpha(df, res)


class GTJAAlpha56(AlphaProcessor):
    """(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),
    SUM(MEAN(VOLUME,40), 19), 13))^5)))"""
    def __call__(self, df):
        res = Lt(Rank((Ref(df, "open") - Min(Ref(df, "open"), 12))), Rank((Rank(Corr(Sum(((Ref(df, "high") + Ref(df, "low")) / 2), 19), Sum(Mean(Ref(df, "volume"),40), 19), 13))**5)))
        return self.register_alpha(df, res)


class GTJAAlpha61(AlphaProcessor):
    """(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
    RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)"""

    def __call__(self, df):
        res = (Greater(Rank(Decaylinear(Delta(Ref(df, "vwap"), 1), 12)),
                       Rank(Decaylinear(Rank(Corr((Ref(df, "low")), Mean(Ref(df, "volume"), 80), 8)), 17))) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha62(AlphaProcessor):
    """(-1 * CORR(HIGH, RANK(VOLUME), 5))"""
    def __call__(self, df):
        res = (-1 * Corr(Ref(df,"high"), Rank(Ref(df,"volume")), 5))
        return self.register_alpha(df, res)


class GTJAAlpha64(AlphaProcessor):
    """(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
    RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)"""
    def __call__(self, df):
        res = (Greater(Rank(Decaylinear(Corr(Rank(Ref(df, "vwap")), Rank(Ref(df, "volume")), 4), 4)), Rank(Decaylinear(Max(Corr(Rank(Ref(df, "close")), Rank(Mean(Ref(df, "volume"),60)), 4), 13), 14))) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha73(AlphaProcessor):
    """((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -
    RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)"""

    def __call__(self, df):
        res = ((Tsrank(Decaylinear(Decaylinear(Corr((Ref(df, "close")), Ref(df, "volume"), 10), 16), 4), 5) - Rank(
            Decaylinear(Corr(Ref(df, "vwap"), Mean(Ref(df, "volume"), 30), 4), 3))) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha74(AlphaProcessor):
    """(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) +
    RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))"""
    def __call__(self, df):
        res = (Rank(Corr(Sum(((Ref(df,"low") * 0.35) + (Ref(df,"vwap") * 0.65)), 20), Sum(Mean(Ref(df,"volume"),40), 20), 7)) + Rank(Corr(Rank(Ref(df,"vwap")), Rank(Ref(df,"volume")), 6)))
        return self.register_alpha(df, res)


class GTJAAlpha77(AlphaProcessor):
    """MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),
    RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))"""

    def __call__(self, df):
        res = Less(Rank(Decaylinear(
            ((((Ref(df, "high") + Ref(df, "low")) / 2) + Ref(df, "high")) - (Ref(df, "vwap") + Ref(df, "high"))), 20)),
                   Rank(Decaylinear(Corr(((Ref(df, "high") + Ref(df, "low")) / 2), Mean(Ref(df, "volume"), 40), 3), 6)))
        return self.register_alpha(df, res)


class GTJAAlpha83(AlphaProcessor):
    """(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))"""
    def __call__(self, df):
        res = (-1 * Rank(Cov(Rank(Ref(df,"high")), Rank(Ref(df,"volume")), 5)))
        return self.register_alpha(df, res)


class GTJAAlpha87(AlphaProcessor):
    """((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /
    (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)"""
    def __call__(self, df):
        res = ((Rank(Decaylinear(Delta(Ref(df, "vwap"), 4), 7)) + Tsrank(Decaylinear(((((Ref(df, "low") * 0.9) + (Ref(df, "low") * 0.1)) - Ref(df, "vwap")) /(Ref(df, "open") - ((Ref(df, "high") + Ref(df, "low")) / 2))), 11), 7)) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha90(AlphaProcessor):
    """( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)"""
    def __call__(self, df):
        res = ( Rank(Corr(Rank(Ref(df,"vwap")), Rank(Ref(df,"volume")), 5)) * -1)
        return self.register_alpha(df, res)


class GTJAAlpha91(AlphaProcessor):
    """((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)"""
    def __call__(self, df):
        res = ((Rank((Ref(df,"close") - Max(Ref(df,"close"), 5)))*Rank(Corr((Mean(Ref(df,"volume"),40)), Ref(df,"low"), 5))) * -1)
        return self.register_alpha(df, res)


GTJA_ALPHA_PROCESSORS = [
    GTJAAlpha1(),
    GTJAAlpha2(),
    GTJAAlpha3(),
    GTJAAlpha4(),
    GTJAAlpha5(),
    GTJAAlpha6(),
    GTJAAlpha7(),
    GTJAAlpha8(),
    GTJAAlpha9(),
    GTJAAlpha10(),
    GTJAAlpha11(),
    GTJAAlpha12(),
    GTJAAlpha13(),
    GTJAAlpha14(),
    GTJAAlpha15(),
    GTJAAlpha16(),
    GTJAAlpha17(),
    GTJAAlpha18(),
    GTJAAlpha19(),
    GTJAAlpha20(),
    GTJAAlpha21(),
    GTJAAlpha22(),
    GTJAAlpha23(),
    GTJAAlpha24(),
    GTJAAlpha25(),
    GTJAAlpha32(),
    GTJAAlpha33(),
    GTJAAlpha35(),
    GTJAAlpha36(),
    GTJAAlpha37(),
    GTJAAlpha41(),
    GTJAAlpha42(),
    GTJAAlpha45(),
    GTJAAlpha48(),
    GTJAAlpha54(),
    GTJAAlpha56(),
    GTJAAlpha61(),
    GTJAAlpha62(),
    GTJAAlpha64(),
    GTJAAlpha73(),
    GTJAAlpha74(),
    GTJAAlpha77(),
    GTJAAlpha83(),
    GTJAAlpha87(),
    GTJAAlpha90(),
    GTJAAlpha91(),
]
