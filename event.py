#!/usr/bin/env python

import pandas as pd
from pandas import HDFStore, bdate_range
from pandas.tseries.offsets import BDay

crsp = HDFStore('/home/chad/WrdsData/hdf/crsp/crsp.h5')
famafrench = HDFStore('/home/chad/WrdsData/hdf/famafrench/famafrench.h5')
DAILY_FACTORS = famafrench.select('/famafrench/factors_daily')


class Event(object):

    def __init__(self, id, evt_date, gap=5, est_period=252,
                 frequency='B', evt_start=-2, evt_end=2):
        self._id = id
        self.evt_date = pd.to_datetime(evt_date)
        self.frequency = frequency
        self._has_data = self._has_models = False
        self.evt_window = bdate_range(start=evt_date - BDay(abs(evt_start)),
                                      end=evt_date + BDay(evt_end))
        self.est_period = bdate_range(end=evt_date - BDay(abs(evt_start - gap)),
                                      periods=est_period)

    def run_study(self):
        if not self._has_data:
            self._merge_data()
        if not self._has_models:
            self._run_regressions()
        self.get_returns()

    def get_returns(self):
        self.abret_capm = (self.evt_data['RETX'] -
                           self.model_capm.predict(x=self.evt_data))
        self.abret_ff3f = (self.evt_data['RETX'] -
                           self.model_ff3f.predict(x=self.evt_data))
        self.abret_ff4f = (self.evt_data['RETX'] -
                           self.model_ff4f.predict(x=self.evt_data))

        self.cum_data = (self.evt_data.loc[:, 'RETX':'mkt'] + 1).cumprod() - 1
        self.car_capm, self.car_ff3f, self.car_ff4f = (self.abret_capm.sum(),
                                                       self.abret_ff3f.sum(),
                                                       self.abret_ff4f.sum())
        self.bhar_mkt = self.cum_data['RETX'] - self.cum_data['mkt']
        self.bhar_capm = (self.cum_data['RETX'] -
                          self.model_capm.predict(x=self.cum_data))
        self.bhar_ff3f = (self.cum_data['RETX'] -
                          self.model_ff3f.predict(x=self.cum_data))
        self.bhar_ff4f = (self.cum_data['RETX'] -
                          self.model_ff4f.predict(x=self.cum_data))

    def _merge_data(self):
        self.est_data = crsp.select('/crsp/dsf',
                                    where=[pd.Term('PERMNO', '=', self._id),
                                    pd.Term('DATE', '=', self.est_period.tolist())])
        self.evt_data = crsp.select('/crsp/dsf',
                                    where=[pd.Term('PERMNO', '=', self._id),
                                    pd.Term('DATE', '=', self.evt_window.tolist())])
        self.est_data = self.est_data.reset_index(level=0).join(DAILY_FACTORS)
        self.evt_data = self.evt_data.reset_index(level=0).join(DAILY_FACTORS)
        self._has_data = True

    def _run_regressions(self):
        self.model_capm = pd.ols(y=self.est_data['RETX'],
                                 x=self.est_data[['mkt']])
        self.model_ff3f = pd.ols(y=self.est_data['RETX'],
                                 x=self.est_data[['mkt', 'smb', 'hml']])
        self.model_ff4f = pd.ols(y=self.est_data['RETX'],
                                 x=self.est_data[['mkt', 'smb', 'hml', 'umd']])
        self._has_models = True

    def _cleanup(self):
        for attr in ('cum_data', 'est_data', 'evt_data'):
            delattr(self, attr)
        self._has_data = False

    def __repr__(self):
        return ("Company: {_id}\n"
                "Event date: {evt_date}\n"
                "Event window: {evt_window}\n"
                "Estimation period: {est_period}".format(**self.__dict__))


class MultipleEvents(object):

    def __init__(self, id, evt_dates, gap=5, est_period=252,
                 frequency='B', evt_start=-2, evt_end=2):
        self.events = [Event(id, evt_date, est_period=est_period,
                             frequency=frequency, evt_start=evt_start,
                             evt_end=evt_end) for evt_date in evt_dates]
