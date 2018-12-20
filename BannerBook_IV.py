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
import Verticalize as Verticalize
import itertools as itertools
import re

# Set working directory for the session
os.chdir('C:/Users/Jochen Binder/Box Sync/eFellows/Daten')
# Read in the df from filepath
df = pd.read_table('01.3 Data gewichtet.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])

df_b = pd.read_table('most_wanted18_berufstaetig_N1484_flat.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])
df_s = pd.read_table('most_wanted18_studierende&absolventen_N5263_flat.csv', sep=';', decimal=',', encoding='latin1', na_values=[' '])
df = pd.concat([df_b, df_s])
del [df_b, df_s]
df['total'] = 1
df['respid'] = range(1,len(df)+1)


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
varlabels = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,K'))
varlabels = varlabels.loc[varlabels['Assign_Label']=='x', 'Variable'].tolist()
df = df.replace({i:dic_val_labels[i] for i in varlabels})
del varlabels

# Define variables to include in bannerbook
variables = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,J'))
variables = variables.loc[variables['Include_Variable']=='x', 'Variable'].tolist()
if variables.count('TOTAL') == 0 :
   variables.insert(0,'TOTAL')

# Define split variables
splitvars = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,L'))
splitvars = splitvars.loc[splitvars['Assign_Split']=='x', 'Variable'].tolist()
if splitvars.count('TOTAL') == 0 :
   splitvars.insert(0,'TOTAL')

# Define numerical variables
num_var = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,M'))
num_var = num_var.loc[num_var['Assign_Numerical']=='x', 'Variable'].tolist()

# Crate iteration-matrix
# iteration_tuples = list(list(zip(r, p)) for (r, p) in zip(repeat(splitvars), permutations(variables)))
testlist = [splitvars, variables]
iteration_tuples = list(itertools.product(*testlist))
iteration_tuples = [i for i in iteration_tuples if i[0]!=i[1]]

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

def Mean(x):
    d = {}
    d['KPI'] = 'mean'
    d['Value'] = np.sum(x['weight']*x[i])/(x['weight']*x[i]).count()
    d['N_weighted'] = np.sum(x['weight'])
    d['N_unweighted'] = np.sum(x['total'])
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
    x = x.groupby([i,j], as_index=False).agg({'weight': 'sum', 'total': 'sum'})
    d = {}
    d['KPI'] = 'distribution in %'
    d['Value'] = 100*x['weight']/x['weight'].sum()
    d['N_weighted'] = sum(x['weight'])
    d['N_unweighted'] = sum(x['total'])
    d['Object_Label'] = x[i]
    d = pd.DataFrame(d)[['Object_Label', 'KPI','Value','N_weighted','N_unweighted']]   
    
    return pd.DataFrame(d); del d

# Code for Split Groups
def Iter_Splitgroups(df, Object, Split, Method=Distribution, Top=2, Vertical_Split = ''):
    global i
    global j
    global t
    i = Object
    j = Split
    t = Top
    v = Vertical_Split
    df['TOTAL'] = 'TOTAL'
    
    g=j if v=='' else [j,v]
    
    tmp = df.groupby(g).apply(lambda x: Method(x)).reset_index()   
    tmp['Split'] = j; tmp['Object'] = i
    indices = [i for i, s in enumerate(tmp.columns.tolist()) if 'level_' in s]
    
    if (Method != Distribution): tmp.iloc[:,indices[0]] = 'n.a.'  
    tmp['Rank'] = tmp['Value'].rank(method='min', ascending=False)
   
    if Method != Distribution:
            if v=='':
                tmp = tmp.rename(index=str, columns={j: 'Split_Label', tmp.columns.tolist()[indices[0]]: 'Object_Label'})  
            if len(g)>1:
                tmp = tmp.rename(index=str, columns={g[0]: 'Split_Label', g[1]: 'Object_Label'}) 
    
    if Method == Distribution:
            if v=='':
                tmp = tmp.rename(index=str, columns={j: 'Split_Label'})#, tmp.columns.tolist()[indices[0]]: 'Object_Label'})  
            if len(g)>1:
                tmp = tmp.rename(index=str, columns={g[0]: 'Split_Label'}) 

    
    tmp = pd.DataFrame(tmp)[['Object', 'Object_Label', 'Split', 'Split_Label',
                          'KPI','Value','Rank',
                          'N_weighted','N_unweighted']]
    
   # if Method == Distribution:
       # tmp = tmp.replace({'Object_Label':dic_val_labels[i] for k in tmp['Object_Label']})

    return tmp; del tmp


test = Iter_Splitgroups(df, Object = 'GESAMTSEMESTER', Split = 'STATUS_TN', Method=Mean)
test = Iter_Splitgroups(df, Object = 'qf11b_46', Split = 'fach', Method=Distribution)
test = Iter_Splitgroups(df, Object = 'fach', Split = 'STATUS_TN', Method=Distribution)

tmp = pd.DataFrame()
counter = 0
for a in iteration_tuples:
    counter = counter + 1
    tmp_df = Iter_Splitgroups(df, Object = a[1], Split = a[0], Method=Distribution)
    tmp = pd.concat([tmp, tmp_df], axis=0)
    print ("{:.0f}".format(counter/len(iteration_tuples)*100),'%')
tmp = tmp.reset_index(drop=True)

tmp.to_csv('eFellows_bannerbook.csv')
test = tmp.replace({'Object_Label':dic_val_labels[i] for k in tmp['Object_Label']})




# Rename comp columns
df.rename(columns={'comp1': 'comp_1', 'comp2': 'comp_2',
                   'comp3': 'comp_3', 'comp4': 'comp_4',
                   'comp5': 'comp_5'}, inplace=True)

df = df.rename(columns=lambda x: re.sub('qb1_', 'qb1',x))

coldict = {'comp1': 'comp_1', 'comp2': 'comp_2',
                   'comp3': 'comp_3', 'comp4': 'comp_4',
                   'comp5': 'comp_5'}

# Define variables to verticalize
melt_cols = pd.read_excel('Vars_Labels.xlsx', sheet_name='Variable_Information', usecols=('A,N'))
melt_cols = melt_cols.loc[melt_cols['Funnel']=='x', 'Variable'].tolist()
melt_cols = [coldict.get(n, n) for n in melt_cols]
melt_cols[:] = [s.replace('qb1_', 'qb1') for s in melt_cols]

#test = verticalize(df)
test = Verticalize.verticalize(df)
new = test.melt(df, melt_cols)
newp = test.pivot(new)

newt = newp.head(10).append(new.tail(10))

new.loc[new['variable'] != 'comp', 'value'] =  new['brand']

df = newp.copy()

melt_cols[:] = [s.replace('qb1_', 'qb1') for s in melt_cols]
for i in melt_cols:
   melt_cols[i] = re.sub('qb1_','qb1', i)




x= 42

def foo():
    x=3
    
    def baz():
        print(x)
        print(locals())
    baz()
    
    


def foo():
    global x
    x=3
    
def baz():
    foo()
    print(x)
    print(locals())
baz()



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