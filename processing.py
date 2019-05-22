import pandas as pd
import json

def get_word_frequency():
    # https://www.wordfrequency.info/free.asp
    word_freq = pd.read_csv('./word_frequency.csv',sep='\t')
    # Remove random white spaces
    word_freq.Word = word_freq.Word.str.strip()
    # Set as lower case
    word_freq.Word = word_freq.Word.str.lower()
    # Words sometimes appear twice, get the most frequent one
    word_freq.drop_duplicates('Word',keep='first',inplace=True)
    
    return word_freq

def get_log(start,nrows):
    if nrows > 2000000:
        raise Exception('Too large')
    
    else:
        log = pd.read_csv('./data/log_data_ready.csv',
                        nrows=nrows,
                        usecols = [1,5,6,8,11,12,15,16,17],
                        header=None,
                        skiprows=start,
                        encoding = "ISO-8859-1",
                        sep='\t')
    
    # Set header
    log.columns = [
        'ts_id',
        'key',
        'text_field',
        'timestamp',
        'input_len',
        'lev_dist',
        'swype',
        'predict',
        'autocorr'
    ]
    
    return log


def log_process(log):
    log = log.copy()
    # Sort by timestamp
    log.sort_values(['ts_id','timestamp'],inplace=True)
    log.reset_index(drop=True, inplace=True)
    
    # Remove nulls, cast to correct type
    log = log_preprocess(log)
    
    # Add IKI and len diff. We need these to validate afterwards
    log['len_diff'] = calculate_len_diff(log)
    log['iki'] = calculate_iki(log)
    
    # Fill in keys, remove junk rows
    log = log_validate(log)    
    
    # Decode ite
    log['ite'] = log[['swype','predict','autocorr']].idxmax(axis=1)
    # Fill in zeros
    log.loc[log[['swype','predict','autocorr']].sum(axis=1) == 0,'ite'] = 'none'
    # Remove old columns
    log.drop(['swype','predict','autocorr'],axis=1,inplace=True)
    
    # Remove punctuation
    log = log.loc[~log.key.str.contains('\.|,|\?')].copy()
    log.reset_index(drop=True, inplace=True)
    
    # Drop more unused columns
    log.drop(
        ['timestamp','input_len'],
        axis=1,
        inplace=True
    )
    return log

    

def log_preprocess(log):
    log = log.copy()
    
    # Replace null text field with empty field (it seems to be the only case where it happens)
    log.loc[log.text_field.isna(),'text_field'] = ''
    
    # Replace backspaces
    log.loc[log.key.isna(),'key'] = '_'
    # Replace null text field (means empty field)
    log.loc[log.text_field.isnull(),'text_field'] = ''
    
    # Cast to correct type
    dtypes = {
        'ts_id':            'int64',
        'key':              'object',
        'text_field':       'object',
        'timestamp':        'int64',
        'input_len':        'uint8',
        'lev_dist':         'uint8',
        'swype':            'bool',
        'predict':          'bool',
        'autocorr':         'bool',
    }
    log = log.astype(dtypes)
    
    return log

def log_validate(log):
    log = log.copy()
    
    # Filter large iki
    log = filter_iki(log,5000)
    
    # Replace undefined keys
    ## Case 1: No change
    mask = (log.key == 'undefined') & (log.lev_dist == 0)
    log.loc[mask,'key'] = ''
    
    ## Case 2: First word of sentence
    first_key = log.groupby('ts_id').head(1).text_field
    log.loc[first_key.index,'key'] = first_key
    
    ## Case 3: Characters added to the end of the sentence
    def find_key(x):
        # Check that the end of the sentence is what was added to (not the middle of the sentence)
        if x.text_field[:-x.lev_dist] == x.text_field_prev:
            return x.text_field[-x.lev_dist:]
        else:
            return 'undefined'
    log['text_field_prev'] = log.text_field.shift(1)
    ## Preliminary conditions: Positive LD, and LD is accounted for by additions at the end of the text field
    mask = (log.key == 'undefined') & (log.lev_dist > 0) & (log.len_diff == log.lev_dist)
    log.loc[mask,'key'] = log.loc[mask].apply(find_key,axis=1)
    
    # Squash empty entries
    log['is_rep'] = False

    ## Case 1: If repeated AND is multiple characters AND is fast
    mask = (log.key.shift(-1) == log.key) & (log.lev_dist.shift(-1) == 0)
    mask &= (log.key.shift(-1).str.len() > 1)
    mask &= (log.iki.shift(-1) < 30)
    log.loc[mask,'iki'] += log.shift(-1).loc[mask,'iki']
    log.loc[mask,'is_rep'] = True
    log.drop(log.loc[mask].index + 1, inplace=True)
    log.reset_index(drop=True,inplace=True)

    ## Case 2: Empty key (this is due to poor key inference of undefined keys)
    mask = (log.key.shift(-1) == '')
    log.loc[mask,'iki'] += log.shift(-1).loc[mask,'iki']
    log.loc[mask,'is_rep'] = True
    log.drop(log.loc[mask].index + 1, inplace=True)
    log.reset_index(drop=True,inplace=True)
    
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
        267956,
        268067,
        268085
    ]

    lab_ts = ts.loc[ts[2].isin(lab_participants)]

    logs = []
    for i in range(24,27):
        log = get_log(i*1000000,1000000)
        logs.append(log.loc[log.ts_id.isin(lab_ts[0].unique())])

    lab_log = pd.concat(logs)

    lab_log = log_process(lab_log)
    
    lab_log = pd.merge(lab_log,ts[[0,2]],left_on=['ts_id'],right_on=[0])
    lab_log.drop(0,axis=1,inplace=True)
    lab_log.rename(columns={2:'participant_id'},inplace=True)
    
    return lab_log

def get_participants_for_log(log):
    ts = get_test_sections()
    
    log = pd.merge(log,ts[['TEST_SECTION_ID','PARTICIPANT_ID']],left_on='ts_id',right_on='TEST_SECTION_ID')
    log.drop('TEST_SECTION_ID',axis=1,inplace=True)
    log.rename(columns={'PARTICIPANT_ID':'participant_id'},inplace=True)
    
    return log
    
def calculate_iki(log):
    return log.groupby('ts_id').timestamp.diff()


def filter_iki(log, thresh):
    return log.groupby('ts_id').filter(lambda x: (x.iki.dropna() < thresh).all()).copy()


def calculate_len_diff(log):
    return log.groupby('ts_id').input_len.diff().fillna(-1).astype('int8')


def calculate_iki_norm(log):
     return log.iki / log.lev_dist

    
def describe_ite(log):
    ite = log[['swype','predict','autocorr']].idxmax(axis=1)
    ite[log[['swype','predict','autocorr']].sum(axis=1) == 0] = 'none'
    return ite


def get_test_sections():
    return pd.read_csv('./data/test_sections.csv',sep='\t')

def get_participants():
    return pd.read_csv('./data/participants.csv', sep='\t')


def mark_entries(log):
    log = log.copy()
    
    # 1. Mark forward entries
    ## Default
    log['is_forward'] = False

    ## Case 1: Zero LD. We assume this is always forward.
    mask = (log.key != 'undefined') & (log.lev_dist == 0)
    log.loc[mask,'is_forward'] = True
    
    ## Case 2: Beginning of a sentence
    first_key = log.groupby('ts_id').head(1).text_field
    log.loc[first_key.index,'is_forward'] = True
    
    ## Case 3: LD > 0. Only if the LD is accounted for by characters added at the end of a sentence.
    '''
    We are currently using the length of the key. Another way to do this is to look at the lev_dist (i.e. check that
    the text field minus [LD amount of characters] is equal to the previous text field). However, this is flawed
    since it would result in corrective inputs (e.g. giulty --> guilty) not being recognized as forward entries.
    '''
    mask = (log.key != 'undefined') & (log.lev_dist > 0) & (~log.is_forward)
    log.loc[mask,'is_forward'] = log.loc[mask].apply(lambda x: x.text_field[-len(x.key):] == x.key, axis=1)
    
    ## Case 4: Backspace at the current word. Double check that the difference is the character at the end of the text field
    mask = log.key == '_'
    mask &= (log.text_field == log.text_field_prev.str[:-1])
    log.loc[mask,'is_forward'] = True

    # 2. Define entries
    ## Assign entry id based on the number of spaces and non-forward backspaces
    log['entry_id'] = log.text_field.str.findall(' ').apply(len)
    # Multi-character keys ending in a space actually belong to the previous entry
    log.loc[(log.key.str[-1] == ' ') & (log.key.str.len() > 1),'entry_id'] -= 1

    ## Negative entries for separators
    log.loc[(log.key == ' '),'entry_id'] = -1
    log.loc[(log.key == '_') & (~log.is_forward),'entry_id'] = -2
    log.loc[~log.is_forward,'entry_id'] = -3


    # Reset, to be safe
    log.reset_index(drop = True,inplace= True)
    
    return log


def infer_ite(log):
    log = log.copy()

    # Assume no ite by default
    log['ite'] = 'none'

    # 1. Infer swype

    ## Case 1: Has leading spaces AND multiple characters
    mask = (log.key.str[0] == ' ') & (log.key.str.len() > 2)
    log.loc[mask,'ite'] = 'swype'

    ## Case 2: The first action of an entry is multicharacter (excl. spaces) AND there's multiple actions
    index_first = log.groupby(['ts_id','entry_id']).head(1).index
    mask = (log.index.isin(index_first)) & (log.key.str.strip(' ').str.len() > 1)
    mask &= (log.entry_id == log.entry_id.shift(-1))
    log.loc[mask,'ite'] = 'swype'

    ## Case 3: The first action of the very first entry has multiple characters (excluding spaces)
    index_first = log.groupby(['ts_id']).head(1).index
    mask = (log.index.isin(index_first)) & (log.key.str.strip(' ').str.len() > 1)
    log.loc[mask,'ite'] = 'swype'

    ## Case 3: The first action of a new word has multiple characters (excluding spaces) AND it's slow
    mask = log.text_field.shift(1).str[-1] == ' '
    index_first = log.loc[mask].groupby(['ts_id','entry_id']).head(1).index
    mask = (log.index.isin(index_first)) & (log.key.str.strip(' ').str.len() > 1) & (log.iki_norm > 150)
    log.loc[mask,'ite'] = 'swype'

    ## Case 4: The first action of a new word has multiple characters (excluding spaces) AND it's long
    mask = log.text_field.shift(1).str[-1] == ' '
    index_first = log.loc[mask].groupby(['ts_id','entry_id']).head(1).index
    mask = (log.index.isin(index_first)) & (log.key.str.strip(' ').str.len() > 1)
    mask &= (log.key.str.len() > 5)
    log.loc[mask,'ite'] = 'swype'

    ## Case 5: Fill in the same entry as a swype
    log.set_index(['ts_id','entry_id'],inplace=True)
    log.loc[log.loc[log.ite == 'swype'].index,'ite'] = 'swype'
    log.reset_index(inplace=True)

    # TODO Case 3 could also mean a prediction
    # TODO what about swype followed by a prediction correction?
    # TODO what about backspace followed by a prediction?

    # 2. Infer Prediction

    ## Case 1: The last action of an entry has multiple characters AND is slow
    mask = (log.key.str.len() > 1)
    mask &= (log.lev_dist > 0) & (log.ite != 'swype') & (log.iki > 500)
    log.loc[mask,'ite'] = 'predict'

    # 3. Infer Autocorrect
    
    ## Case 1: The last action of an entry has multiple characters AND there are multiple entries AND is fast
    mask = (log.key.str.len() > 1)
    mask &= (log.entry_id == log.entry_id.shift(1))
    mask &= (log.lev_dist > 0) & (log.ite != 'swype') & (log.iki < 400)
    log.loc[mask,'ite'] = 'autocorr'

    # Reset negative entries
    log.loc[log.entry_id < 0,'ite'] = 'none'

    return log

def infer_ite_no_swype(log):
    log = log.copy()

    # Assume no ite by default
    log['ite'] = 'none'

    # 1. Infer Prediction

    ## Case 1: The last action of an entry has multiple characters AND is slow
    mask = (log.key.str.len() > 1)
    mask &= (log.lev_dist > 0) & (log.ite != 'swype') & (log.iki > 500)
    log.loc[mask,'ite'] = 'predict'

    # 2. Infer Autocorrect
    
    ## Case 1: The last action of an entry has multiple characters AND there are multiple entries AND is fast
    mask = (log.key.str.len() > 1)
    mask &= (log.entry_id == log.entry_id.shift(1))
    mask &= (log.lev_dist > 0) & (log.ite != 'swype') & (log.iki < 400)
    log.loc[mask,'ite'] = 'autocorr'

    # Reset negative entries
    log.loc[log.entry_id < 0,'ite'] = 'none'

    return log