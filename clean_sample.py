import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Clean log data')
parser.add_argument('--path',required=True,help='Path to the log data to be cleaned')
args = parser.parse_args()

if __name__ == "__main__":
    print('Reading file...')
    log = pd.read_csv(args.path,sep='\t')

    print('Changing columns names...')
    log.columns = [
        'ts_id',
        'key',
        'input_len',
        'lev_dist',
        'autocorr',
        'predict',
        'swype',
        'text_field',
        'timestamp'    
    ]

    print('Updating file...')
    log.to_csv(args.path,index=False)
    print('DONE')
      
