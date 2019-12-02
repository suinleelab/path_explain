import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data():
    X_df = pd.read_csv('/projects/leelab2/data/AD_DATA/Nicasia/processed' + \
                       '/PCG_normalized/no_covar_correction/MSBB_RNA.tsv', 
                 sep='\t')
    y_df = pd.read_csv('/projects/leelab2/data/AD_DATA/Nicasia/processed' + \
                       '/samples_neuropath_prenorm/MSBB_RNA.tsv',
                  sep='\t')
    
    X_df = X_df.T
    X_df.columns = X_df.iloc[0]
    X_df.drop('PCG', axis=0, inplace=True)
    X_df.dropna(how='any', axis=1, inplace=True)
    X_df.index = X_df.index.astype(int)
    
    y_df.set_index('sample_name', inplace=True)
    y_df = y_df.loc[X_df.index]
    y_df.dropna(how='any', subset=['AD'], inplace=True)
    X_df = X_df.loc[y_df.index]
    y = y_df['AD'].values.astype(int)
    X_df = X_df.astype(float)
    
    X_train_total, X_test, y_train_total, y_test = train_test_split(X_df, y, test_size=0.15, random_state=0, stratify=y)
    X_train, X_vald, y_train, y_vald = train_test_split(X_train_total, y_train_total, test_size=0.15, random_state=0, stratify=y_train_total)
    return X_train_total, y_train_total, \
           X_train, y_train, \
           X_vald,  y_vald, \
           X_test,  y_test