# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 00:00:17 2018

@author: Jochen Binder
"""

# Load packages
import os as os
import sys as sys
import pandas as pd
import numpy as np
import Raking as Rake
import Verticalize as Verticalize

# Set working directory for the session
os.chdir('C:/Users/Jochen Binder/Box Sync/eFellows/Daten')
# Read in the df from filepath
df = pd.read_table('01.3 Data gewichtet.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])

df_b = pd.read_table('most_wanted18_berufstaetig_N1484_flat.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])
df_s = pd.read_table('most_wanted18_studierende&absolventen_N5263_flat.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])
df = pd.concat([df_b, df_s])
del [df_b, df_s]
df['total'] = 1


dic_varnames = pd.read_excel('Lookup.xlsx', sheet_name='Vardict', usecols=('A,B'), index_col=0).to_dict()
dic_varnames = dic_varnames['Newname']

df = df.rename(columns=dic_varnames)

dic_var = pd.read_excel('Lookup.xlsx', sheet_name='Variables', usecols=('A,E'), index_col=0).to_dict()
dic_var = dic_var['Label']

dic_brand = pd.read_excel('Lookup.xlsx', sheet_name='Brands', usecols=('A,B'), index_col=0).to_dict()
dic_brand = dic_brand['Brand']

dic_subject = pd.read_excel('Lookup.xlsx', sheet_name='Subjects', usecols=('A,B'), index_col=0).to_dict()
dic_subject = dic_subject['Subject']

# Nested dict from data frame
dic_val_labels = pd.read_excel('Vars_Labels.xlsx', sheet_name='Value_Labels', usecols=('A,B,C'))
d = dic_val_labels.groupby(['Value', 'Label_num'], as_index=False).sum()
dic_val_labels = {}
for i in d['Value'].unique():
    dic_val_labels[i] = {d['Label_num'][j]: d['Label'][j] for j in d[d['Value']==i].index}
del d

df = df.replace({'qd1': dic_val_labels['qd1']})


# Weighting
aggregates = pd.read_excel('Poolzahlen_20180425_test.xlsx', sheet_name='crosstab', index_col=None, header=0)
aggregates = pd.melt(aggregates, id_vars=['STATUS_TN'], value_vars=['Männlich','Weiblich'], var_name='qd1', value_name='total')
agg2 = aggregates.groupby(['STATUS_TN', 'qd1'])['total'].sum()
agg0 = aggregates.groupby(['STATUS_TN'])['total'].sum()
agg1 = aggregates.groupby(['qd1'])['total'].sum()

data = df.copy()

ipfn_k = Rake.ipfn(data, [agg0, agg1, agg2],
               [['STATUS_TN'], ['qd1'], ['STATUS_TN', 'qd1']], 'total')
df = ipfn_k.iteration()
df = ipfn_k.weighting()

vars = df.columns.tolist()



metric = ['qft2_2']

test = df.groupby(['qd1'])['total']
test = df.total * df.weight
test = df['total'].multiply(df['weight'])
test = df.groupby(['qd1'])['total'].multiply(df['weight'])

test = df.groupby(['qd1'])['weight', 'total'].apply(lambda x: df['weight']*8, axis=0)

test = df.groupby(['qd1'])['weight'] * df.groupby(['qd1'])['total']

group = df.groupby(['qd1'])['total'].apply(lambda x: x/sum())

group['weight']/group['total']

def percentage(data, col1, col2):
    percent = data[col1]*data[col2]/(data[col1]*data[col2]).sum()
    return percent

def percentage(data, col):
    percent = data[col]/(data[col].sum())
    return percent

grgend = df[df['qd1']!='Männlich'].groupby(['STATUS_TN'])
grgend = df.groupby(['STATUS_TN', 'qd1'], as_index=False)
test=percentage(grgend.sum(), 'weight', 'total').reset_index()
test= grgend.apply(lambda x: percentage(x, 'weight', 'total')).reset_index().rename(columns={0: 'Percentage'})

test= grgend.agg(lambda x: percentage(x, 'weight', 'total')).reset_index().rename(columns={0: 'Percentage'})

test= grgend.sum().groupby(['STATUS_TN']).apply(lambda x: percentage(x, 'weight', 'total')).reset_index().rename(columns={0: 'Percentage'})

test= pd.DataFrame(grgend.sum().apply(lambda x: percentage(x, 'weight', 'total')))








def Distribution(x):
    d = {}
    d['KPI'] = 'distribution in %'
    d['Value'] = 100*x['weight']/x['weight'].sum()
    d['N_weighted'] = x['weight']
    d['N_unweighted'] = x['total']
    d['Rank'] = d['Value'].rank(method='min', ascending=False)
    d = pd.DataFrame(d)[['KPI','Value','Rank','N_weighted','N_unweighted']]
    return pd.DataFrame(d); del d

# Code for Split Groups
def Iter_Splitgroups(df, Object, Split, Method=Distribution):
    i = Object
    j = Split
    df['TOTAL'] = 'TOTAL'
    
    tmp = df.groupby([i, j], as_index=True).agg({'weight': 'sum', 'total': 'sum'})
    tmp = tmp.groupby(level=1, as_index=True).apply(Method).reset_index()
    tmp['Split'] = j; tmp['Object'] = i
    tmp = tmp.rename(index=str, columns={j: 'Split_Label', i: 'Object_Label'})
    tmp = pd.DataFrame(tmp)[['Object', 'Object_Label',
                          'Split', 'Split_Label','KPI','Value','Rank',
                          'N_weighted','N_unweighted']]
    return tmp; del tmp

# Code no Split Groups
test = df.groupby(['STATUS_TN'], as_index=True).agg({'weight': 'sum', 'total': 'sum'})
test = Distribution(test).reset_index()


test.to_csv('Test.csv')

test[['total', 'weight']]




test = test.groupby(level=0).apply(lambda x: percentage(x, 'weight'))

test = test.groupby(level=0).apply(lambda x: round(100*x/x.sum(),0))
test = test.groupby(level=0)['weight'].apply({'a': lambda x: round(100*x/x.sum(),0),
                                    'N': lambda x: x.sum()})


df.loc[:,['weight']]/df.loc[:,['total']]

new = grgend.apply(lambda x: x['weight'].rank(axis=0, ascending=True)).reset_index()
test['Rank'] = test.groupby(['qd1', 'STATUS_TN'])['Percentage'].rank(ascending=False, method='min')