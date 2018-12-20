# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 00:00:17 2018

@author: Jochen Binder
"""

# Load packages
import os as os
import pandas as pd
import numpy as np
import Raking as Rake

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

dic_gender = {1: 'männlich', 2: 'weiblich'}

df = df.replace({'qd1': dic_gender})

vars = df.columns.tolist()




aggregates = pd.read_excel('Poolzahlen_20180425_test.xlsx', sheet_name='crosstab', index_col=None, header=0)
aggregates = pd.melt(aggregates, id_vars=['STATUS_TN'], value_vars=['männlich','weiblich'], var_name='qd1', value_name='total')
agg2 = aggregates.groupby(['STATUS_TN', 'qd1'])['total'].sum()
agg0 = aggregates.groupby(['STATUS_TN'])['total'].sum()
agg1 = aggregates.groupby(['qd1'])['total'].sum()


data = df.copy()

ipfn_k = Rake.ipfn(data, [agg0, agg1, agg2],
               [['STATUS_TN'], ['qd1'], ['STATUS_TN', 'qd1']], 'total')
df = ipfn_k.iteration()
df = ipfn_k.weighting()


data = df.copy()

ipfn_k = Rake.ipfn(data, [agg1, agg0],
               [['qd1'], ['STATUS_TN']], 'total')
df = ipfn_k.iteration()
df = ipfn_k.weighting()


def categorization(data, varnames):
    for i in varnames:
        data[i] = data[i].astype('category').copy()
    return data.describe(include='all'), data

test = verticalize(df)
new = test.melt(df, melt_cols)
newp = test.pivot(new)

df = newp.copy()

vars = df.columns.tolist()
new_vars = vars[0:11] + vars[415:]

dt = df[df.columns.intersection(new_vars)]
   
    df['status'].dtype
    df['qd5'].astype('category').describe()
    
df['status'].astype('category')
df['status'].describe(include='all')
df['status'].dtype
df['qd5'].astype('category').describe()

df['new'] = df.loc[df['fach'] == 2]

conditions = [
    (df['qd4'] == 2) & (df['fach'] == 1),
    (df['qd4'] == 1) & (df['fach'] == 1),
    (df['fach'] == 2)]
choices = ['Studierende', 'Absolventen', 'Juristen']
df['Teilnehmer'] = np.select(conditions, choices, default='Mitglieder')
print(df)

test = df.groupby(['STATUS_TN', 'qf11a_1'])['total'].sum()

new = df[df['qf11a_1']==2].groupby(['STATUS_TN', 'qd1'])['total'].sum()/df[df['qf11a_1'].notnull()].groupby(['STATUS_TN', 'qd1'])['total'].sum()

newk.replace({'})

data.to_csv('eFellows.csv')

dt['brand'] = dt['brand'].astype(int)
dt = dt.replace({'brand': dic_brand})


dt.to_csv('eFellows.csv')
dt.to_csv('eFellows_PowerBI.csv', sep=";", decimal=",")
df.to_csv('eFellows_II.csv')

test1 = df.loc[df['qd1']=='männlich'].groupby(['STATUS_TN'])['total'].sum()
test2 = df.groupby(['STATUS_TN'])['total'].sum()
test3 = round((test1/test2)*100,0)

: df_merged = pd.merge(test1, test3['total_new'], left_on=['STATUS_TN'], right_on=['STATUS_TN'])