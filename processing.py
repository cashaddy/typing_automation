import pandas as pd
import json

def get_log(path, process = True, filtering = True, infer_ite = True):
    log = pd.read_csv(path)
    if process:
        log = get_participants_for_log(log)
        log = log_process(log)
    if filtering or infer_ite:
        log = mark_entries(log)
    if filtering:
        log = filter_log(log)
    if infer_ite:
        log = infer_ite_no_swype(log)
        log = infer_sub_strategy(log)
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
    log = log.loc[~log.key.str.contains('\.|,|\?|\!')].copy()
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
    
    # Replace bad apostrophes
    log.key  = log.key.str.replace("`","'")
    log.key  = log.key.str.replace("’","'")
    log.text_field  = log.text_field.str.replace("`","'")
    log.text_field  = log.text_field.str.replace("’","'")
    
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
    mask &= (log.text_field.str.strip() == log.text_field_prev.str.strip().str[:-1])
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


def filter_log(log):
    # Remove all non-forward entries
    log = log.loc[log.is_forward].copy()

    # Remove participants with the cumulative multichar behaviour
    # Action of an entry is single letter AND following two actions in the entry have increasing # of chars
    mask = log.key.str.len() == 1
    mask &= log.key.shift(-1).str.len() == 2
    mask &= log.key.shift(-2).str.len() == 3
    mask &= log.entry_id.shift(-1) == log.entry_id
    mask &= log.entry_id.shift(-2) == log.entry_id.shift(-1)
    participants_invalid = log.loc[mask].participant_id.unique()
    log = log.loc[~log.participant_id.isin(participants_invalid)]

    # Remove participants who use swipe
    participants_ = get_participants().set_index('PARTICIPANT_ID')
    participants_ = participants_.loc[log.participant_id.unique()].copy()
    participants_swipe = participants_.loc[participants_.USING_FEATURES.str.contains('swipe')].index
    log = log.loc[~log.participant_id.isin(participants_swipe)].copy()
    participants_ = participants_.loc[log.participant_id.unique()].copy()

    # Remove heavy swype users that managed to get past the earlier swipe filtering stage
    # We do this by detect users where swipe was used more than 10 percent of the time 
    log = infer_ite_swype(log)
    participants_swipe = log.groupby('participant_id').ite.value_counts(
        normalize=True
    ).unstack()['swype'].sort_values(ascending=False)
    participants_swipe = participants_swipe.loc[participants_swipe >= 0.1].index
    log = log.loc[~log.participant_id.isin(participants_swipe)].copy()

    # Remove participants where every entry before a space is a multichar
    # Entry is multicharacter AND is followed by a space AND is zero LD
    mask = log.key.str.strip(' ').str.len() > 1
    mask &= log.key.shift(-1) == ' '
    mask &= log.lev_dist == 0
    log['tmp'] = mask
    # Find participants who have a large percentage of these 0-LD, multichar entries
    participants_multichar = log.groupby('participant_id').tmp.value_counts().unstack()[True].fillna(0)
    participants_multichar /= log.groupby(['participant_id','ts_id']).entry_id.nunique().sum(level=0)
    participants_multichar = participants_multichar.loc[participants_multichar > 0.2].index
    # Remove them
    log = log.loc[~log.participant_id.isin(participants_multichar)].copy()

    # Remove non-native speakers
    participants_nonnative = participants_.loc[participants_.NATIVE_LANGUAGE != 'en'].index
    log = log.loc[~log.participant_id.isin(participants_nonnative)].copy()
    
    return log


def infer_ite_swype(log):
    log = log.copy()
    
    log['iki_norm'] = log.iki / log.key.str.len()

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

    return log

def infer_ite_no_swype(log):
    log = log.copy()

    # Assume no ite by default
    log['ite'] = 'none'

    # 1. Infer Prediction

    ## Case 1: The action of an entry has multiple characters AND is slow
    mask = (log.key.str.len() > 1)
    mask &= (log.lev_dist >= 0) & (log.iki > 500)
    log.loc[mask,'ite'] = 'predict'

    # 2. Infer Autocorrect
    
    ## Case 1: The action of an entry has multiple characters AND there are multiple entries AND is fast
    mask = (log.key.str.len() > 1)
    mask &= (log.entry_id == log.entry_id.shift(1))
    mask &= (log.lev_dist > 0) & (log.iki < 400)
    log.loc[mask,'ite'] = 'autocorr'

    # Reset negative entries
    log.loc[log.entry_id < 0,'ite'] = 'none'

    return log

def infer_sub_strategy(log):
    log = log.copy()
    
    # Default null
    log['ite2'] = None
    # Default for predict is 'other'
    log.loc[log.ite == 'predict','ite2'] = 'other'

    # Prediction: The ITE action is the only action in the entry AND is new word
    mask = log.ite == 'predict'
    mask &= log.entry_id != log.entry_id.shift(-1)
    mask &= log.entry_id != log.entry_id.shift(1)
    mask &= log.text_field.shift(1).str[-1] == ' '
    log.loc[mask,'ite2'] = 'prediction'

    # Correction: Not pure forward motion AND mutliple actions in the entry
    mask = log.ite == 'predict'
    mask &= log.len_diff != log.lev_dist
    mask &= log.entry_id == log.entry_id.shift(1)
    log.loc[mask,'ite2'] = 'correction'

    # Fixup: Preceded by backspace AND not a new word
    mask = log.ite == 'predict'
    mask &= log.key.shift(1) == '_'
    mask &= log.entry_id.shift(1) == log.entry_id
    mask &= log.text_field.shift(1).str[-1] != ' '
    log.loc[mask,'ite2'] = 'fixup'

    # No-change prediction: multiple actions, LD is 0, len_diff is 0, 
    mask = log.ite == 'predict'
    mask &= log.len_diff == 0
    mask &= log.lev_dist == 0
    mask &= log.entry_id == log.entry_id.shift(1)
    log.loc[mask,'ite2'] = 'no_change'

    # Completion: Pure forward motion AND mutliple actions in the entry AND there is a change
    mask = log.ite == 'predict'
    mask &= log.len_diff == log.lev_dist
    mask &= log.lev_dist > 0
    mask &= log.entry_id == log.entry_id.shift(1)
    log.loc[mask,'ite2'] = 'completion'
    '''
    Edge case: It's quite common for correction to cause len_diff of 1. So double check that the last letter
    changes between the inputs. We also check against double letters, since this could still be a completion.
    '''
    mask = log.ite2 == 'completion'
    mask &= log.len_diff == 1
    mask &= log.text_field.str[-1] == log.text_field_prev.str[-1]
    mask &= log.text_field.str[-1] != log.text_field.str[-2]
    log.loc[mask,'ite2'] = 'correction'
    
    return log


def log_to_words(log):
    # All valid keystrokes
    groupby = log.loc[log.entry_id >= 0].groupby([
        'participant_id',
        'ts_id',
        'entry_id'
    ])

    words = groupby.last().text_field.str.split(' ').str[-1]
    words = words.to_frame('word')
    # For keys ending with a space, look two spaces back instead of one 
    mask = groupby.last().key.str[-1] == ' '
    words.loc[mask,'word'] = groupby.last().loc[mask].text_field.str.split(' ').str[-2]

    words['word_length'] = words.word.str.len()

    # Time per keystroke
    groupby = log.groupby([
        'participant_id',
        'ts_id',
        'entry_id'
    ])
    words['iki'] = groupby.iki.sum()
    words.iki /= words.word_length

    # Only letter keystrokes
    mask = (log.len_diff == 1) & (log.is_forward)
    mask &= (log.key.str.contains('[a-z]')) & (log.key.shift(1).str.contains('[a-z]'))
    groupby = log.loc[mask].groupby([
        'participant_id',
        'ts_id',
        'entry_id'
    ])
    words['iki_letters'] = groupby.iki.mean()
    
    # Letter keystrokes AND ite time
    mask = (log.len_diff == 1) & (log.is_forward) & (log.ite == 'none')
    mask &= (log.key.str.contains('[a-z]')) & (log.key.shift(1).str.contains('[a-z]'))
    mask |= (log.entry_id >= 0) & (log.ite != 'none')
    groupby = log.loc[mask].groupby([
        'participant_id',
        'ts_id',
        'entry_id'
    ])
    words['iki_letters_and_ite'] = groupby.iki.mean()


    # Only ite keystrokes
    groupby = log.loc[(log.entry_id >= 0) & (log.ite != 'none')].groupby([
        'participant_id',
        'ts_id',
        'entry_id'
    ])

    words['ite'] = groupby.last().ite
    words.ite.fillna('none',inplace=True)

    words['ite2'] = groupby.last().ite2

    words['ite_input'] = groupby.last().text_field.str.split().str[-1]
    words['ite_input_key'] = groupby.last().key
    words['ite_input_len'] = words.ite_input.str.len()

    words['ite_input_prev'] = groupby.last().text_field_prev.str.split().str[-1]
    words.loc[(words.ite != 'none') & (words.ite_input_prev.isna()),'ite_input_prev'] = '' # Replace prev text with empty string  
    words['ite_lev_dist'] = groupby.last().lev_dist

    words['ite_len_diff'] = groupby.last().len_diff

    words['ite_iki'] = groupby.last().iki

    groupby = log.loc[(log.entry_id >= 0) & (log.key == '_')].groupby([
        'participant_id',
        'ts_id',
        'entry_id'
    ])
    words['n_backspace'] = groupby.size()
    words.n_backspace = words.n_backspace.fillna(0)

    words.reset_index(inplace=True)
    words = words.loc[words.word.notna()].copy()
    words = words.loc[words.word != ''].copy()
    
    return words

def process_words(words):
    words = words.copy()
    words['type'] = None

    # Get the type of word
    # First letter is capital AND not the first word in the sentence AND not "I"
    mask = (words.word.str.contains("[A-Z]")) & (words.entry_id != 0) & (words.word != 'I')
    words.loc[mask,'type'] = 'proper'

    # Contains an apostrophe
    mask = (words.word.str.contains("'"))
    words.loc[mask,'type'] = 'contraction'

    # The rest are normal words
    words.loc[words.type.isna(),'type'] = 'generic'
    
    
    # Get the frequency of the word
    ## https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Contemporary_fiction
    word_freq = pd.read_csv('./data/word_frequency_2.csv')
    word_freq.word = word_freq.word.str.lower()

    freq_map = word_freq.set_index('word').frequency
    words['freq'] = words.word.str.lower().map(freq_map)
    words['freq_category'] = None

    words.loc[(words.freq < 100000) & (words.freq > 0),'freq_category'] = 'common'
    words.loc[(words.freq > 100000),'freq_category'] = 'very_common'

    ## The rest are uncommon
    mask = (words.freq_category.isna())
    words.loc[mask,'freq_category'] = 'uncommon'
    
    return words