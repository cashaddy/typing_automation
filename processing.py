import pandas as pd
import json

def get_log(start,nrows):
    if nrows > 2000000:
        raise Exception('Too large')
    
    else:
        log = pd.read_csv('./data/log_data_ready.csv',
                        nrows=nrows,
                        usecols = [0,1,2,5,6,8] + list(range(11,18)),
                        header=None,
                        skiprows=start,
                        encoding = "ISO-8859-1",
                        sep='\t')
    

    
    return log


def process_log(log):
    # Set header
    log.columns = [
        'log_data_id',
        'ts_id',
        'type',
        'key',
        'text_field',
        'timestamp',
        'input_len',
        'lev_dist',
        'input_len_prev',
        'lev_dist_prev',
        'swype',
        'predict',
        'autocorr']
    
    log.log_data_id = log.log_data_id.astype('int')
    
    # Sort by timestamp
    log.sort_values(['ts_id','timestamp','log_data_id'],inplace=True)
    log.reset_index(drop=True, inplace=True)    

    # Add IKI
    log['iki'] = calculate_iki(log)
    # Filter large iki
    log = filter_iki(log,5000)
    # Normalize iki
    log['iki_norm'] = calculate_iki_norm(log)
    
    # Calculate length difference
    log['len_diff'] = calculate_len_diff(log)
    
    # Decode ite
    log['ite'] = log[['swype','predict','autocorr']].idxmax(axis=1)
    # Fill in zeros
    log.loc[log[['swype','predict','autocorr']].sum(axis=1) == 0,'ite'] = 'none'
    
    return log

def get_lab_results():
    ts = get_test_sections()
    
    lab_participants = [
        252249,
        254300,
        254745,
        263374,
        263720,
        265900,
        265924,
        267956,
        268067,
        268085
    ]

    lab_ts = ts.loc[ts[2].isin(lab_participants)]

    logs = []
    for i in range(24,27):
        log = get_log(i*1000000,1000000)
        logs.append(log.loc[log[1].isin(lab_ts[0].unique())])

    lab_log = pd.concat(logs)

    lab_log = process_log(lab_log)

    lab_log['participant_id'] = lab_log.ts_id.map(
        dict(zip(lab_ts[0],lab_ts[2]))
    )
    
    return lab_log
    
def calculate_iki(log):
    return log.groupby('ts_id').timestamp.diff()


def filter_iki(log, thresh):
    return log.groupby('ts_id').filter(lambda x: (x.iki.dropna() < thresh).all()).copy()


def calculate_len_diff(log):
    return log.groupby('ts_id').input_len.diff()


def calculate_iki_norm(log):
     return log.iki / log.lev_dist

    
def describe_ite(log):
    ite = log[['swype','predict','autocorr']].idxmax(axis=1)
    ite[log[['swype','predict','autocorr']].sum(axis=1) == 0] = 'none'
    return ite


def get_test_sections():
    return pd.read_csv('./data/test_sections_ready.csv',
                    header=None,
                    encoding = "ISO-8859-1",
                    sep='\t')


def get_participants():
    return pd.read_csv('./data/participants_mod.csv',
                       index_col=0,
                       encoding = "ISO-8859-1",
                       sep=',')

def get_header(table):
    with open('./tracked_data.json','r') as f:
        headers = json.load(f)
       
    return headers[table]