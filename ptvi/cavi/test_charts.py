import unittest

from ptvi.cavi.test_inference import _gen_ar2
from ptvi.cavi.inference import arp_mcmc_forecast
from ptvi.cavi.charts import *


class ChartTest(unittest.TestCase):
    """Test matplotlib charts."""

    def test_fan_chart(self):
        np.random.seed(123)
        y, _ = _gen_ar2(do_fit=False)
        kwargs = dict(warmup=1_000, chains=4, a_0=2, b_0=0.5, c_0=2, d_0=0.5)
        fc = arp_mcmc_forecast(y, p=2, steps=10, draws=1_000, **kwargs)
        fig = fan_chart(y, fc)

