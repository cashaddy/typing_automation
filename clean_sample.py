import pandas as pd

print('Reading file...')
log = pd.read_csv('./data/log_sample.csv',sep='\t')

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
log.to_csv('./data/log_sample.csv',index=False)
print('DONE')
      
