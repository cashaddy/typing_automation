
# coding: utf-8

import pandas as pd
from ua_parser import user_agent_parser
import processing

print('Reading participants.csv...')
participants = processing.get_participants()

print('Parsing user agent (this may take a few minutes)...)
participants['browser'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseUserAgent(x)['family'])
participants['os'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseOS(x)['family'])
participants['device_family'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseDevice(x)['family'])
participants['device_brand'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseDevice(x)['brand'])
participants['device_model'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseDevice(x)['model'])

print('Saving to participants.csv...')
participants.to_csv('./participants.csv',sep='\t',index=False)

print('DONE')