import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pm4py

from sklearn.model_selection import train_test_split

def add_deal_outcome_to_df(group):
    entry = group.tail(1)
    entry.activity_added_date = entry.closed_date
    entry.activity_type = entry.case_outcome_string
    entry.activity_seq += 1
    entry.activity_active_days = 0
    entry.activity_count += 1
    return pd.concat([group, entry], ignore_index=True)

def create_list_of_lists(grp):
    return [grp[0:i+1] for i in range(len(grp))]

def decode_activity_types(array):
    arr = [activity_types[x] for x in array if x]
    return json.dumps(arr, separators=(', ', ':'), ensure_ascii=False).replace('"', "'")

def prepare_sequences(df):
    activity_types = dict(enumerate(df.activity_type.unique()))
    sequences = df[['activity_type_list', 'case_outcome']]
    sequences['activity_type_list'] = sequences.activity_type_list.astype(str)

    days = df['activity_active_days_list']
    acts = df['activity_type_list'].apply(lambda x: np.array(x[:-1]))
    y = df['activity_type_list'].apply(lambda x: np.array(x[-1]))
    days_train, days_test, acts_train, acts_test, y_train, y_test = train_test_split(days, acts, y, test_size=0.3, random_state=42)
    return days_train, days_test, acts_train, acts_test, y_train, y_test, sequences

def split_dataset(days_train, days_test, acts_train, acts_test, y_train, y_test, num_activity_types, max_seq_length):
    X_train = pad_set(days_train, acts_train, num_activity_types, max_seq_length)
    X_test = pad_set(days_test, acts_test, num_activity_types, max_seq_length)
    y_train = to_categorical(y_train, num_classes=num_activity_types)
    y_test = to_categorical(y_test, num_classes=num_activity_types)
    return X_train, y_train, X_test, y_test

def pad_set(days, acts, num_activity_types, max_seq_length):
    act_types = pad_sequences(acts, maxlen=max_seq_length - 1)
    
    x1 = to_categorical(act_types, num_classes=num_activity_types)
    x2 = pad_sequences(days, maxlen=max_seq_length - 1)
    x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
    x = np.concatenate([x1, x2], axis=2)
    return x

def build_seq_freq(df):
    seq_df = df.groupby('case_id').agg(list)
    seq_df['single_case_outcome'] = seq_df.case_outcome.apply(lambda x: x[-1])
    seq_df['activity_type_str'] = seq_df.activity_type.apply(lambda x: x[:-1]).astype(str)
    return seq_df.groupby(['activity_type_str'])['single_case_outcome'].value_counts(normalize=True).round(decimals=2)

def append_outcome_to_seq(df):
    df['case_outcome_string'] = df.case_outcome.apply(lambda x: utils.case_outcome_types[x])
    # Add deal outcome
    print("Adding outcome to sequence...")
    df = df.groupby('case_id').apply(lambda x: add_deal_outcome_to_df(x)).reset_index(drop=True)
    encode_activity_types = utils.get_encode_activity_types(df)
    df['le_activity_type'] = df.activity_type.apply(lambda x: encode_activity_types[x])
    return df

def collect_sequence_traces(df):
    print("Preparing sequence traces...")
    df['activity_type_list'] = None
    df['activity_active_days_list'] = None

    for caseId in tqdm(df['case_id'].unique()):
        case_acts = df.loc[df.case_id == caseId]
        for idx, case_act in case_acts.iterrows():
            new_part = df.loc[
                (df.case_id == case_act.case_id) & (df.activity_seq <= case_act.activity_seq)
            ]
            seq_list = new_part.groupby('case_id').le_activity_type.agg(lambda x: x.tolist())
            active_days_list = new_part.groupby('case_id').activity_active_days.agg(lambda x: x.tolist())
            df.at[idx, 'activity_type_list'] = seq_list.values[0]
            df.at[idx, 'activity_active_days_list'] = active_days_list.values[0]
    num_activity_types = len(df['le_activity_type'].unique()) + 1
    max_seq_length = df['activity_count'].max()

    return df, num_activity_types, max_seq_length

def xes_to_dataframe(filename):
    logs = pm4py.read_xes(filename)
    df = pm4py.convert_to_dataframe(logs)

    df = df.rename(columns={'concept:name': 'activity_type', 'case:concept:name': 'case_id', 'time:timestamp':'timestamp'})
    df = df['activity_type', 'case_id', 'timestamp']
    return df

def build_active_days_colummn():
    for caseId in tqdm(df['case_id'].unique()):
        case_acts = df.loc[df.case_id == caseId]
        case_id = None
        for idx, case_act in case_acts.iterrows():
            if case_id == case_act['case_id']:
                df.at[idx, 'activity_active_days'] = (case_act['timestamp'] - df.at[idx - 1, 'timestamp']).days
            else:
                case_id = case_act['case_id']
                df.at[idx, 'activity_active_days'] = 0
    
    success_case_ids = df.query('activity_type == "O_Accepted"')['case_id'].unique()

    df['case_outcome'] = 0

    df.loc[df['case_id'].isin(success_case_ids), 'case_outcome'] = 1

    df['le_activity_type'] = df.activity_type
    df['activity_count'] = df.groupby('case_id')['case_id'].transform('count')
    
    return df
