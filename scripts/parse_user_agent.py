
# coding: utf-8

import argparse, sys
import pandas as pd
from ua_parser import user_agent_parser
sys.path.append('../')
import processing

parser = argparse.ArgumentParser(description='Parse the user agent of the participants data')
parser.add_argument('--path',required=True,help='Path to the participant data to be parsed')
args = parser.parse_args()

if __name__ == "__main__":
    print('Reading participants.csv...')
    participants = pd.read_csv(args.path, sep='\t')

    print('Parsing user agent (this may take a few minutes)...')
    participants['browser'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseUserAgent(x)['family'])
    participants['os'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseOS(x)['family'])
    participants['device_family'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseDevice(x)['family'])
    participants['device_brand'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseDevice(x)['brand'])
    participants['device_model'] = participants.USER_AGENT.apply(lambda x:  user_agent_parser.ParseDevice(x)['model'])

    print('Saving to participants.csv...')
    participants.to_csv(args.path,sep='\t',index=False)

    print('DONE')