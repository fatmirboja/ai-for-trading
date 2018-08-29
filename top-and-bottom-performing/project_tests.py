from collections import OrderedDict
import pandas as pd
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output


@project_test
def test_date_top_industries(fn):
    tickers = generate_random_tickers(10)
    dates = generate_random_dates(2)

    fn_inputs = {
        'prices': pd.DataFrame(
            [
                [21.050810483942833, 17.013843810658827, 10.984503755486879, 11.248093428369392, 12.961712733997235,
                 482.34539247360806, 35.202580592515041, 3516.5416782257166, 66.405314327318209, 13.503960481087077],
                [15.63570258751384, 14.69054309070934, 11.353027688995159, 475.74195118202061, 11.959640427803022,
                 10.918933017418304, 17.9086438675435, 24.801265417692324, 12.488954191854916, 15.63570258751384]],
            dates, tickers),
        'sector': pd.Series(
            ['ENERGY', 'MATERIALS', 'ENERGY', 'ENERGY', 'TELECOM', 'FINANCIALS',
             'TECHNOLOGY', 'HEALTH', 'MATERIALS', 'REAL ESTATE'],
            tickers),
        'date': dates[-1],
        'top_n': 4}
    fn_correct_outputs = OrderedDict([
        (
            'top_industries',
            set(['ENERGY', 'HEALTH', 'TECHNOLOGY']))])

    assert_output(fn, fn_inputs, fn_correct_outputs)
