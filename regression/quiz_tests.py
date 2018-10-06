from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output

@project_test
def test_regression_slope_and_intercept(fn):

    # just set the seed for the random number generator
    np.random.seed(100)
    # use returns to create a price series
    drift = 100
    r0 = pd.Series(np.random.normal(0, 1, 1000))
    s0 = pd.Series(np.cumsum(r0), name='s0') + drift

    noise1 = np.random.normal(0, 0.4, 1000)
    drift1 = 50
    r1 = r0 + noise1
    s1 = pd.Series(np.cumsum(r1), name='s1') + drift1

    noise2 = np.random.normal(0, 0.4, 1000)
    drift2 = 60
    r2 = r0 + noise2
    s2 = pd.Series(np.cumsum(r2), name='s1') + drift2

    xSeries = s1
    ySeries = s2

    lr = LinearRegression()
    xVar = xSeries.values.reshape(-1,1)
    yVar = ySeries.values.reshape(-1,1)
    lr.fit(xVar,yVar)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    fn_inputs = {
        'xSeries': s1,
        'ySeries': s2
        }

    fn_correct_outputs = OrderedDict([
        ('slope',slope),
        ('intercept',intercept)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)
