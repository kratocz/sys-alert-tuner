import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from pyzabbix import ZabbixAPI
from dotenv import load_dotenv

load_dotenv()


class ZabbixClient:
    def __init__(self):
        self.url = os.getenv('ZABBIX_URL')
        self.user = os.getenv('ZABBIX_USER')
        self.password = os.getenv('ZABBIX_PASSWORD')
        self.token = os.getenv('ZABBIX_TOKEN')
        
        if not self.url:
            raise ValueError("Missing ZABBIX_URL in .env file")
        
        # Validate authentication method
        if self.token:
            # Token authentication - only needs token
            if self.user or self.password:
                print("Warning: ZABBIX_TOKEN found, ignoring ZABBIX_USER and ZABBIX_PASSWORD")
        else:
            # Username/password authentication
            if not all([self.user, self.password]):
                raise ValueError("Missing Zabbix credentials: either ZABBIX_TOKEN or both ZABBIX_USER and ZABBIX_PASSWORD required")
        
        self.zapi = None
        self.connect()
    
    def connect(self):
        """Connect to Zabbix API using token or username/password"""
        try:
            self.zapi = ZabbixAPI(self.url)
            
            if self.token:
                # Token-based authentication
                self.zapi.session.headers.update({
                    'Authorization': f'Bearer {self.token}',
                    'Content-Type': 'application/json'
                })
                # Test connection by getting API version
                try:
                    api_info = self.zapi.apiinfo.version()
                    print(f"Connected to Zabbix server: {self.url} (version: {api_info}) using token")
                except Exception as e:
                    print(f"Token authentication failed: {e}")
                    raise ValueError("Invalid Zabbix token or token authentication not supported")
            else:
                # Username/password authentication
                self.zapi.login(self.user, self.password)
                print(f"Connected to Zabbix server: {self.url} using username/password")
                
        except Exception as e:
            print(f"Failed to connect to Zabbix: {e}")
            raise
    
    def get_hosts(self, group_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get hosts, optionally filtered by host groups"""
        try:
            if group_names:
                hostgroups = self.zapi.hostgroup.get(
                    filter={'name': group_names}
                )
                groupids = [group['groupid'] for group in hostgroups]
                hosts = self.zapi.host.get(
                    groupids=groupids,
                    output=['hostid', 'name', 'host']
                )
            else:
                hosts = self.zapi.host.get(output=['hostid', 'name', 'host'])
            
            return hosts
        except Exception as e:
            print(f"Error getting hosts: {e}")
            return []
    
    def get_items(self, hostids: List[str], key_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get items for specified hosts"""
        try:
            params = {
                'hostids': hostids,
                'output': ['itemid', 'name', 'key_', 'hostid', 'value_type', 'units']
            }
            
            if key_pattern:
                params['search'] = {'key_': key_pattern}
                
            items = self.zapi.item.get(**params)
            return items
        except Exception as e:
            print(f"Error getting items: {e}")
            return []
    
    def get_triggers(self, hostids: List[str] = None) -> List[Dict[str, Any]]:
        """Get triggers for specified hosts"""
        try:
            params = {
                'output': ['triggerid', 'description', 'expression', 'priority', 'status'],
                'expandExpression': True,
                'selectHosts': ['hostid', 'name'],
                'selectItems': ['itemid', 'name', 'key_']
            }
            
            if hostids:
                params['hostids'] = hostids
                
            triggers = self.zapi.trigger.get(**params)
            return triggers
        except Exception as e:
            print(f"Error getting triggers: {e}")
            return []
    
    def get_historical_data(self, itemids: List[str], time_from: datetime, 
                          time_till: datetime = None) -> pd.DataFrame:
        """Get historical data for items"""
        try:
            if time_till is None:
                time_till = datetime.now()
            
            history_data = []
            
            for itemid in itemids:
                history = self.zapi.history.get(
                    itemids=[itemid],
                    time_from=int(time_from.timestamp()),
                    time_till=int(time_till.timestamp()),
                    output='extend',
                    sortfield='clock',
                    sortorder='ASC'
                )
                
                for point in history:
                    history_data.append({
                        'itemid': itemid,
                        'timestamp': datetime.fromtimestamp(int(point['clock'])),
                        'value': float(point['value']),
                        'ns': int(point.get('ns', 0))
                    })
            
            df = pd.DataFrame(history_data)
            return df
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def get_events(self, hostids: List[str] = None, time_from: datetime = None,
                   time_till: datetime = None) -> List[Dict[str, Any]]:
        """Get events (trigger fires) for analysis"""
        try:
            if time_from is None:
                time_from = datetime.now() - timedelta(days=7)
            if time_till is None:
                time_till = datetime.now()
            
            params = {
                'output': ['eventid', 'source', 'object', 'objectid', 'clock', 'value', 'acknowledged'],
                'source': 0,  # Trigger events
                'object': 0,  # Trigger object
                'time_from': int(time_from.timestamp()),
                'time_till': int(time_till.timestamp()),
                'selectRelatedObjects': ['triggerid', 'description', 'priority'],
                'sortfield': 'clock',
                'sortorder': 'DESC'
            }
            
            if hostids:
                params['hostids'] = hostids
                
            events = self.zapi.event.get(**params)
            return events
        except Exception as e:
            print(f"Error getting events: {e}")
            return []
    
    def disconnect(self):
        """Disconnect from Zabbix API"""
        if self.zapi:
            # Only call logout if using username/password authentication
            if not self.token and self.user and self.password:
                try:
                    self.zapi.user.logout()
                except Exception as e:
                    print(f"Warning: Could not properly logout: {e}")
            print("Disconnected from Zabbix")