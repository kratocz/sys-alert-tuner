import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import os

from sys_alert_tuner.zabbix_client import ZabbixClient


class TestZabbixClient:
    
    @patch.dict(os.environ, {
        'ZABBIX_URL': 'http://test.zabbix.com',
        'ZABBIX_USER': 'test_user',
        'ZABBIX_PASSWORD': 'test_password'
    })
    @patch('sys_alert_tuner.zabbix_client.ZabbixAPI')
    def test_init_with_username_password(self, mock_zabbix_api):
        """Test initialization with username/password"""
        mock_api = Mock()
        mock_zabbix_api.return_value = mock_api
        
        client = ZabbixClient()
        
        assert client.url == 'http://test.zabbix.com'
        assert client.user == 'test_user'
        assert client.password == 'test_password'
        assert client.token is None
        
        mock_zabbix_api.assert_called_once_with('http://test.zabbix.com')
        mock_api.login.assert_called_once_with('test_user', 'test_password')
    
    @patch.dict(os.environ, {
        'ZABBIX_URL': 'http://test.zabbix.com',
        'ZABBIX_TOKEN': 'test_token_123'
    })
    @patch('sys_alert_tuner.zabbix_client.ZabbixAPI')
    def test_init_with_token(self, mock_zabbix_api):
        """Test initialization with API token"""
        mock_api = Mock()
        mock_api.apiinfo.version.return_value = '6.0.0'
        mock_zabbix_api.return_value = mock_api
        
        client = ZabbixClient()
        
        assert client.url == 'http://test.zabbix.com'
        assert client.token == 'test_token_123'
        
        # Should not call login when using token
        mock_api.login.assert_not_called()
        mock_api.apiinfo.version.assert_called_once()
    
    @patch.dict(os.environ, {'ZABBIX_URL': 'http://test.zabbix.com'})
    def test_init_missing_credentials(self):
        """Test initialization with missing credentials"""
        with pytest.raises(ValueError, match="Missing Zabbix credentials"):
            ZabbixClient()
    
    @patch.dict(os.environ, {})
    def test_init_missing_url(self):
        """Test initialization with missing URL"""
        with pytest.raises(ValueError, match="Missing ZABBIX_URL"):
            ZabbixClient()
    
    @patch.dict(os.environ, {
        'ZABBIX_URL': 'http://test.zabbix.com',
        'ZABBIX_USER': 'test_user',
        'ZABBIX_PASSWORD': 'test_password'
    })
    @patch('sys_alert_tuner.zabbix_client.ZabbixAPI')
    def test_get_hosts(self, mock_zabbix_api):
        """Test getting hosts"""
        mock_api = Mock()
        mock_api.host.get.return_value = [
            {'hostid': '1', 'name': 'Host 1', 'host': 'host1'},
            {'hostid': '2', 'name': 'Host 2', 'host': 'host2'}
        ]
        mock_zabbix_api.return_value = mock_api
        
        client = ZabbixClient()
        hosts = client.get_hosts()
        
        assert len(hosts) == 2
        assert hosts[0]['name'] == 'Host 1'
        mock_api.host.get.assert_called()
    
    @patch.dict(os.environ, {
        'ZABBIX_URL': 'http://test.zabbix.com',
        'ZABBIX_USER': 'test_user',
        'ZABBIX_PASSWORD': 'test_password'
    })
    @patch('sys_alert_tuner.zabbix_client.ZabbixAPI')
    def test_get_historical_data(self, mock_zabbix_api):
        """Test getting historical data"""
        mock_api = Mock()
        mock_api.history.get.return_value = [
            {'clock': '1640995200', 'value': '75.5', 'ns': '0'},
            {'clock': '1640995260', 'value': '78.2', 'ns': '0'}
        ]
        mock_zabbix_api.return_value = mock_api
        
        client = ZabbixClient()
        time_from = datetime.now() - timedelta(hours=1)
        
        df = client.get_historical_data(['12345'], time_from)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'itemid' in df.columns
        assert 'timestamp' in df.columns
        assert 'value' in df.columns
    
    @patch.dict(os.environ, {
        'ZABBIX_URL': 'http://test.zabbix.com',
        'ZABBIX_USER': 'test_user',
        'ZABBIX_PASSWORD': 'test_password'
    })
    @patch('sys_alert_tuner.zabbix_client.ZabbixAPI')
    def test_disconnect_with_password(self, mock_zabbix_api):
        """Test disconnect with password auth"""
        mock_api = Mock()
        mock_zabbix_api.return_value = mock_api
        
        client = ZabbixClient()
        client.disconnect()
        
        mock_api.user.logout.assert_called_once()
    
    @patch.dict(os.environ, {
        'ZABBIX_URL': 'http://test.zabbix.com',
        'ZABBIX_TOKEN': 'test_token_123'
    })
    @patch('sys_alert_tuner.zabbix_client.ZabbixAPI')
    def test_disconnect_with_token(self, mock_zabbix_api):
        """Test disconnect with token auth"""
        mock_api = Mock()
        mock_api.apiinfo.version.return_value = '6.0.0'
        mock_zabbix_api.return_value = mock_api
        
        client = ZabbixClient()
        client.disconnect()
        
        # Should not call logout when using token
        mock_api.user.logout.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__])