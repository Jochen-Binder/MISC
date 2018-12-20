# Load packages
import os as os
import pandas as pd
import numpy as np

# Set working directory for the session
os.chdir('C:/Users/Jochen Binder/Box Sync/eFellows/Daten')
# Read in the df from filepath
#df = pd.read_table('TEST_PYTHON.csv', sep=';', encoding='latin1')

# Define variables to be verticalized & all remaining columns
melt_cols = list(df.iloc[:,101:287].columns)
cols = df.columns
diff = cols.difference(melt_cols)

# Keeping the order of the original df frame
ordered_cols = [item for item in cols if item in diff]

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

# Create dicts of variable labels and brands
var_dict = {'f201c': 'aided_awareness', 'f202': 'brand_usage', 'f204': 'recommend', 
            'f205': 'sales_dist', 'f206': 'f_use', 'f207': 'trust', 'f208': 'uniqueness', 
            'f209': 'price_image', 'f214_1': 'innovative', 'f214_2': 'technology', 'f214_3': 'value_for_money', 
            'f214_4': 'energy_efficient', 'f214_5': 'easy_to_install', 'f214_6': 'reliability', 'f214_7': 'quality_general', 
            'f214_8': 'renewable_technologies', 'f214_9': 'uncomplicated', 'f214_10': 'customer_oriented', 
            'f214_11': 'social_envir_responsibility', 'f214_12': 'professional_client_treatment', 'f214_13': 'treated_equally', 
            'f215_1': 'training_offers', 'f215_2': 'field_support', 'f215_3': 'product_range', 'f215_4': 'delivery_capacity', 
            'f215_5': 'aftersales_service', 'f215_6': 'reachable_hotline', 'f215_7': 'loyalty_programme', 
            'f215_8': 'marketing_support', 'f215_9': 'quality_product', 'f215_10': 'good_margins', 'f215_11': 'internet_presence'}

brand_dict = {'1': 'Agpo', '4': 'ATAG', '5': 'AWB', '8': 'Bosch', '23': 'Ferroli', '31': 'Intergas', '32': 'Itho | Daalderop', 
			'37': 'Nefit', '41': 'Nemeha', '48': 'Vaillant', '89': '89', 'newa': 'newa', 'newb': 'newb', '989': '989', '996': '996', '998': '998'}

# Replace brand and variable with labels (instead of numbers)
df['brand'] = df['brand'].astype('str').apply(lambda x: brand_dict[x])
df['f203'] = df['f203'].astype('str').apply(lambda x: brand_dict[x])
df['variable'] = df['variable'].astype('str').apply(lambda x: var_dict[x])

# Replace '98' and '99' as NaN in 'variable'. The same for Non-brands in 'brand'
df['value']=df['value'].replace(to_replace=[99,98,999,998], value=np.NaN)
df['value']=df['value'].replace(to_replace=[2], value=0)
df['brand']=df['brand'].replace(to_replace=['newa', 'newb', '89'], value=np.NaN)

# Preparing list of variables to be kept during pivoting
names = df.columns.values.tolist()
names.remove('variable')
names.remove('value')

# Pivot long df back to wide data by variables
df = df.pivot_table(index=names, columns=['variable'], values='value')
df.reset_index(inplace=True)
df = df.drop_duplicates(subset=None, keep='first', inplace=False)

# Conditionally defining main brand variable
df['main_brand'] = np.where(df['brand_usage']==4,0,np.NaN)
df['main_brand'] = np.where(df['brand']==df['f203'],1,df['main_brand'])

# Creating funnel variables
df['stage23'] = np.where(df['brand_usage']==1,0,np.where(df['brand_usage']>1,1,np.NaN))
df['stage34'] = np.where((df['brand_usage']==2) | (df['brand_usage']==3),0,np.where(df['brand_usage']==4,1,np.NaN))
df['stage45'] = np.where(df['main_brand']==0,0,np.where(df['main_brand']==1,1,np.NaN))

# Saving data frame as csv-file
df.to_csv('example.csv')
