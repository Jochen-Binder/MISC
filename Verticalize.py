#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from itertools import product
import copy


class verticalize(object):

    def __init__(self, df):
        """
        Initialize the verticalize class
        original: numpy darray matrix or dataframe to perform the ipfn on.
        aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
        They are the target values that we want along one or several axis when aggregating along one or several axes.
        dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
        Preserved dimensions along which we sum to get the corresponding aggregates.
        convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.
        max_iteration: Integer. Maximum number of iterations allowed.
        verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.
        0: Updated matrix returned. 
        1: Flag with the output status (0 for failure and 1 for success).
        2: dataframe with iteration numbers and convergence rate information at all steps. 
        rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value
        For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
        """
        self.original = df
        self.original['tmp_ID'] = range(len(self.original))
        self.cols = df.columns

        
    def melt(self, df, columns):
        melt_cols = columns 
        cols = self.cols
        diff = cols.difference(melt_cols)
        df['tmp_ID']=range(len(df))
    
        # Keeping the order of the original df frame
        ordered_cols = [item for item in self.cols if item in diff]
    
        # Verticalize the df set
        df = pd.melt(df, id_vars=ordered_cols, value_vars=melt_cols, var_name='variable_original')
    
        # Deleting unneded helper variables
        del(ordered_cols, melt_cols, diff, cols)
    
        # Create 'brand' and 'item' variable and strip unnecessary strings
        df['brand'] = df['variable_original'].apply(lambda x: x.split(sep="_")[1])
        df['variable'] = df['variable_original'].map(lambda x: x.split(sep="_")[0] 
                                                        if x.count('_')<=1 
                                                        else x.split(sep="_")[0]+'_'+x.split(sep="_")[2])
        del df['variable_original']
        
        return df
    
    def pivot(self, df, variable='variable', value='value', ID = 'tmp_ID'):
        
        # sort the dataframe
        df.sort_values(ID, inplace=True)
        # set the index to be this and don't drop
        df.set_index(keys=[ID], drop=False,inplace=True)
        # get a list of names
        names=df[ID].unique().tolist()        
        
        cols = [c for c in df.columns if c not in {variable, value}]
        
        tmp = pd.DataFrame()
        counter = 0
        
        for k in names:
            df_tmp = df.loc[df[ID]==k,:]
            df_tmp = df_tmp.set_index(cols + [variable]).unstack(variable)
            df_tmp.columns = df_tmp.columns.levels[1]
            df_tmp.columns.name = None
            tmp = pd.concat([tmp, df_tmp], axis=0)
            counter += 1
            print ("{:.0f}".format(counter/len(names)*100),'%')
        df = tmp.copy()
        del tmp
        df = df.reset_index()
        
        
        #https://stackoverflow.com/questions/17333644/pandas-dataframe-transforming-frame-using-unique-values-of-a-column
        
        #cols = [c for c in new.columns if c not in {'variable', 'value'}]
        #df2 = new.set_index(cols + ['variable']).unstack('variable')
        #df2.columns = df2.columns.levels[1]
        #df2.columns.name = None
        #df2.reset_index()
        
        
        return df
