import pickle
import re
import numpy as np
import torch
import sparse
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sparse
from transformers import BertTokenizer
from pandarallel import pandarallel


class MyDataset(Dataset):
    def __init__(self, x1, x2, s, input_ids, token_type_ids, attention_mask, label, device):
        self.x1 = x1
        self.x2 = x2
        self.s = s
        self.input_ids = input_ids
        self.token_type = token_type_ids
        self.attention_mask = attention_mask
        self.label = label
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.FloatTensor(self.x1[idx]).to(self.device), torch.FloatTensor(self.x2[idx]).to(self.device), \
               torch.FloatTensor(self.s[idx]).to(self.device), torch.tensor(self.input_ids[idx]).to(self.device), \
               torch.tensor(self.token_type[idx]).to(self.device), torch.tensor(self.attention_mask[idx]).to(self.device), \
               torch.tensor(self.label[idx], dtype=torch.float).to(self.device)


class DataModule(object):
    def __init__(self, device, #mimic_dir: str = './data/mimic_csv',
                 data_dir: str = './data',
                 mimic_dir: str = './mimic',
                 task: str = 'mortality',
                 duration: float = 48.0,
                 timestep: float = 1.0,
                 ):
        # self.mimic_dir = Path(mimic_dir)
        self.device = device
        self.data_dir = Path(data_dir)
        self.task = task
        self.duration = duration
        self.timestep = timestep
        self.mimic_dir = mimic_dir
        self.label_dir = self.data_dir / f'population/{task}_{duration}h.csv'
        self.label_name = f'{task}_{duration}h'
        self.subject_dir = self.data_dir / 'prep/icustays_MV.csv'
        self.task_dir1 = self.data_dir / f'features/outcome={task},T={duration},dt={timestep},1'
        self.task_dir2 = self.data_dir / f'features/outcome={task},T={duration},dt={timestep},2'
        self.X_dir1 = self.task_dir1 / 'X.npz'
        self.X_dir2 = self.task_dir2 / 'X.npz'
        self.s_dir = self.task_dir1 / 'S.npz'
        self.train, self.val, self.test = self.split()


    def note_feats(self):
        labels = pd.read_csv(self.label_dir, usecols=['ID', f'{self.task}_LABEL']). \
            rename(columns={'ID': 'ICUSTAY_ID', f'{self.task}_LABEL': 'LABEL'})

        notes = pd.read_csv(self.mimic_dir+'/'+'NOTEEVENTS.csv', parse_dates=['CHARTDATE', 'CHARTTIME', 'STORETIME'])

        icus = pd.read_csv(self.mimic_dir+'/'+'ICUSTAYS.csv', parse_dates=['INTIME', 'OUTTIME']).sort_values(
            by=['SUBJECT_ID']).reset_index(drop=True)

        df = pd.merge(icus[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME']], notes, on=['SUBJECT_ID', 'HADM_ID'],
                      how='inner')
        df = df.drop(columns=['SUBJECT_ID', 'HADM_ID'])

        df = pd.merge(labels['ICUSTAY_ID'], df, on='ICUSTAY_ID', how='left')

        df = df[df['ISERROR'].isnull()]

        df = df[df['CHARTTIME'].notnull()]

        df['TIME'] = (df['CHARTTIME'] - df['INTIME']).apply(lambda x: x.total_seconds()) / 3600
        df = df[(df['TIME'] <= self.duration) & (df['TIME'] >= 0.0)]

        df['VARNAME'] = df[['CATEGORY', 'DESCRIPTION']].apply(lambda x: x.CATEGORY + '/' + x.DESCRIPTION, axis=1)
        df = df.groupby(['ICUSTAY_ID', 'VARNAME'])[['TIME', 'TEXT']].apply(lambda x: x.iloc[x['TIME'].argmax()])

        id_texts = df.groupby('ICUSTAY_ID')['TEXT'].apply(lambda x: '\n'.join(x)).reset_index()
        return self.tokenize(id_texts)

    @staticmethod
    def tokenize(id_texts):
        pandarallel.initialize(nb_workers=12)

        def preprocess1(x):
            y = re.sub('\\[(.*?)]', '', x)
            y = re.sub('[0-9]+\.', '', y)
            y = re.sub('dr\.', 'doctor', y)
            y = re.sub('m\.d\.', 'md', y)
            y = re.sub('admission date:', '', y)
            y = re.sub('discharge date:', '', y)
            y = re.sub('--|__|==', '', y)
            return y

        def preprocessing(df_less_n):
            df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
            df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ')
            df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ')
            df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
            df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()

            df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))
            return df_less_n

        id_texts = preprocessing(id_texts)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        id_texts['FEATS'] = id_texts['TEXT'].parallel_apply(
            lambda x: tokenizer.encode_plus(x, padding='max_length', truncation=True,
                                            return_token_type_ids=True, return_attention_mask=True))

        id_texts = id_texts.drop(columns='TEXT')

        id_texts['input_ids'] = id_texts['FEATS'].apply(lambda x: x['input_ids'])

        id_texts['token_type_ids'] = id_texts['FEATS'].apply(lambda x: x['token_type_ids'])

        id_texts['attention_mask'] = id_texts['FEATS'].apply(lambda x: x['attention_mask'])

        id_texts = id_texts.drop(columns='FEATS')
        return id_texts


    def split(self):
        df_label = pd.read_csv(self.label_dir, usecols=['ID', f'{self.task}_LABEL']). \
            rename(columns={'ID': 'ICUSTAY_ID', f'{self.task}_LABEL': 'LABEL'})

        note_feats = self.note_feats()
        df_label = pd.merge(df_label, note_feats, on='ICUSTAY_ID', how='left')

        df_label = df_label[df_label['input_ids'].notnull()]

        df_subjects = df_label.reset_index().merge(pd.read_csv(self.subject_dir), on='ICUSTAY_ID',
                                                   how='left').set_index('index')

        train_idx = df_subjects[df_subjects['partition'] == 'train'].index.values
        val_idx = df_subjects[df_subjects['partition'] == 'val'].index.values
        test_idx = df_subjects[df_subjects['partition'] == 'test'].index.values

        X1 = sparse.load_npz(self.X_dir1).todense()
        X2 = sparse.load_npz(self.X_dir2).todense()
        s = sparse.load_npz(self.s_dir).todense()

        X1_train, X2_train, s_train, label_train = X1[train_idx], X2[train_idx], s[train_idx], np.stack(df_label.loc[train_idx]['LABEL'].to_numpy())
        X1_val, X2_val, s_val, label_val = X1[val_idx], X2[val_idx], s[val_idx], np.stack(df_label.loc[val_idx]['LABEL'].to_numpy())
        X1_test, X2_test, s_test, label_test = X1[test_idx], X2[test_idx], s[test_idx], np.stack(df_label.loc[test_idx]['LABEL'].to_numpy())

        input_ids_tr = np.stack(df_label.loc[train_idx]['input_ids'].to_numpy())
        token_type_ids_tr = np.stack(df_label.loc[train_idx]['token_type_ids'].to_numpy())
        attention_mask_tr = np.stack(df_label.loc[train_idx]['attention_mask'].to_numpy())

        input_ids_d = np.stack(df_label.loc[val_idx]['input_ids'].to_numpy())
        token_type_ids_d = np.stack(df_label.loc[val_idx]['token_type_ids'].to_numpy())
        attention_mask_d = np.stack(df_label.loc[val_idx]['attention_mask'].to_numpy())

        input_ids_t = np.stack(df_label.loc[test_idx]['input_ids'].to_numpy())
        token_type_ids_t = np.stack(df_label.loc[test_idx]['token_type_ids'].to_numpy())
        attention_mask_t = np.stack(df_label.loc[test_idx]['attention_mask'].to_numpy())

        trainset = MyDataset(X1_train, X2_train, s_train, input_ids_tr, token_type_ids_tr, attention_mask_tr, label_train, self.device)
        valset = MyDataset(X1_val, X2_val, s_val, input_ids_d, token_type_ids_d, attention_mask_d, label_val, self.device)
        testset = MyDataset(X1_test, X2_test, s_test, input_ids_t, token_type_ids_t, attention_mask_t, label_test, self.device)
        return trainset, valset, testset

    def get_data(self):
        return self.train, self.val, self.test


def collate_fn(batch):
    x1, x2, s, input_ids, tokens, attention_mask, label = [], [], [], [], [], [], []
    for data in batch:
        x1.append(data[0])
        x2.append(data[1])
        s.append(data[2])
        input_ids.append(data[3])
        tokens.append(data[4])
        attention_mask.append(data[5])
        label.append(data[6])
    return torch.stack(x1, 0), torch.stack(x2, 0), torch.stack(s, 0), torch.stack(input_ids, 0), torch.stack(tokens, 0),\
           torch.stack(attention_mask, 0), torch.stack(label, 0)
