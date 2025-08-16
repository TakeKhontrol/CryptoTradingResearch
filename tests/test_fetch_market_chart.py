import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import crypto_trading_app as cta


def test_fetch_market_chart_includes_interval_parameter():
    with patch('requests.Session.get') as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        cta.fetch_market_chart('bitcoin', vs_currency='usd', days=1, interval='daily')
        args, kwargs = mock_get.call_args
        assert 'interval=daily' in args[0]
