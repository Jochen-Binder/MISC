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
import itertools as itertools

# Set working directory for the session
os.chdir('C:/Users/Jochen Binder/Box Sync/eFellows/Daten')
# Read in the df from filepath
df = pd.read_table('01.3 Data gewichtet.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])

df_b = pd.read_table('most_wanted18_berufstaetig_N1484_flat.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])
df_s = pd.read_table('most_wanted18_studierende&absolventen_N5263_flat.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])
df = pd.concat([df_b, df_s])
del [df_b, df_s]
df['total'] = 1


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

variables = df.columns.tolist()



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

# Assign labels to variables
varlabels = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,J'))
varlabels = varlabels.loc[varlabels['Assign_Label']=='x', 'Variable'].tolist()
df = df.replace({i:dic_val_labels[i] for i in varlabels})
del varlabels

# Define split variables
splitvars = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,K'))
splitvars = splitvars.loc[splitvars['Assign_Split']=='x', 'Variable'].tolist()
if splitvars.count('TOTAL') == 0 :
   splitvars.insert(0,'TOTAL')

# Define numerical variables
num_var = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,L'))
num_var = num_var.loc[num_var['Assign_Numerical']=='x', 'Variable'].tolist()

# Crate iteration-matrix
#iteration_tuples = list(list(zip(r, p)) for (r, p) in zip(repeat(splitvars), permutations(variables)))
testlist = [splitvars, variables]
iteration_tuples = list(itertools.product(*testlist))
iteration_tuples = [i for i in iteration_tuples if i[0]!=i[1]]




def Mean(x):
    d = {}
    d['KPI'] = 'mean'
    d['Value'] = sum(x['weight']*x[i])/len(x['weight']*x[i])
    d['N_weighted'] = sum(x['weight'])
    d['N_unweighted'] = sum(x['total'])
    d = pd.Series(d).to_frame().T[['KPI','Value', 'N_weighted','N_unweighted']]
    return d; del d

def TopX(x):
    d = {}
    d['KPI'] = 'top'+str(t)
    d['Value'] = x.loc[x[i] >= max(x[i]-t), i].sum()/sum(x['weight']*x[i])
    d['N_weighted'] = sum(x['weight'])
    d['N_unweighted'] = sum(x['total'])
    d = pd.Series(d).to_frame().T[['KPI','Value', 'N_weighted','N_unweighted']]
    return d; del d

def Distribution(x):
    x = x.groupby(i+j, as_index=False).agg({'weight': 'sum', 'total': 'sum'})
    d = {}
    d['KPI'] = 'distribution in %'
    d['Value'] = 100*x['weight']/x['weight'].sum()
    d['N_weighted'] = sum(x['weight'])
    d['N_unweighted'] = sum(x['total'])
    d = pd.DataFrame(d)[['KPI','Value','N_weighted','N_unweighted']]
    return pd.DataFrame(d); del d

# Code for Split Groups
def Iter_Splitgroups(df, Object, Split, Method=Distribution, Top=2):
    i = Object
    j = Split
    t = Top
    df['TOTAL'] = 'TOTAL'    
    
    tmp = df.groupby(j).apply(lambda x: Method(x)).reset_index()   
    tmp['Split'] = j[0]; tmp['Object'] = i
    indices = [i for i, s in enumerate(tmp.columns.tolist()) if 'level_' in s]
    
    if tmp.iloc[1,indices[0]] == 0: tmp.iloc[:,indices[0]] = 'n.a.'  
    tmp['Rank'] = tmp['Value'].rank(method='min', ascending=False)
    
    if len(Split)==1:
        tmp = tmp.rename(index=str, columns={j: 'Split_Label', tmp.columns.tolist()[indices[0]]: 'Object_Label'})  
    if len(Split)>1:
        tmp = tmp.rename(index=str, columns={j[0]: 'Split_Label', j[1]: 'Object_Label'}) 
    
    tmp = pd.DataFrame(tmp)[['Object', 'Object_Label',
                          'Split', 'Split_Label','KPI','Value','Rank',
                          'N_weighted','N_unweighted']]
    return tmp; del tmp


test = Iter_Splitgroups(df, Object = ['GESAMTSEMESTER'], Split = ['STATUS_TN'], Method=Distribution)

for a in iteration_tuples:
    tmp_df = Iter_Splitgroups(df, Object = a[0], Split = a[1], Method=Distribution)











list(filter(re.compile('level_*').match, tmp.columns.tolist()))
test.to_csv('Test.csv')

test[['total', 'weight']]

a = {'weight': 'sum', 'total': 'sum'}

df.groupby(j).apply(lambda x: sum(x['weight']*x[i])/len(x['weight']*x[i]))
df.groupby(j).apply(lambda x: sum(x['weight']))


tmp = df.groupby(j).agg({'weight': 'sum', 'total': 'sum'})
tmp['new'] = df.groupby(j).apply(lambda{'weight': 'sum', 'total': 'sum'} x: sum(x['weight']*x[i])/len(x['weight']*x[i]))


test = test.groupby(level=0).apply(lambda x: percentage(x, 'weight'))

test = test.groupby(level=0).apply(lambda x: round(100*x/x.sum(),0))
test = test.groupby(level=0)['weight'].apply({'a': lambda x: round(100*x/x.sum(),0),
                                    'N': lambda x: x.sum()})


df.loc[:,['weight']]/df.loc[:,['total']]

new = grgend.apply(lambda x: x['weight'].rank(axis=0, ascending=True)).reset_index()
test['Rank'] = test.groupby(['qd1', 'STATUS_TN'])['Percentage'].rank(ascending=False, method='min')