from pandas import json_normalize
import ndjson
import numpy as np
import pandas as pd
import json
import torch
import spacy
from sklearn.preprocessing import OneHotEncoder
import re
import datetime
from sklearn.impute import SimpleImputer
import warnings
import torch.nn.functional as F
warnings.simplefilter(action='ignore', category=FutureWarning)
import uuid

def flatten_json(nested_json):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out

def unnest_row(row):
    unnested = {}
    for col in row.index:
        keys = col.split('_')
        current = unnested
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = row[col]
    return unnested

# Load Spacy's pre-trained model for embedding generation
nlp = spacy.load('en_core_web_md')

def is_valid_fhir_datetime(date_string):
    # Regular expression for FHIR dateTime format
    regex = re.compile(r'^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[1-2][0-9]|3[0-1])(T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]+)?(Z|(\+|-)((0[0-9]|1[0-3]):[0-5][0-9]|14:00)))?)?)?$')
    return bool(regex.match(date_string))
# Use the function
#print(is_valid_fhir_datetime("2023-07-04"))  # True
#print(is_valid_fhir_datetime("2023/07/04"))  # False

#Function to convert datafram to a Torch ready datframe    
def process_dataframe(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = df[column].replace(['Yes', 'No', 'True', 'False'], [1, 0, 1, 0]).astype(float)
                #print(f'Column {column} converted to float, naturally')
            except ValueError:
                if df[column].str.contains(' ').any():
                    df[column] = df[column].apply(lambda x: nlp(x).vector.mean())
                    #print(f'Column {column} embedded with spacy')
                else:
                    # Only convert to datetime if all values are valid FHIR datetimes
                    if df[column].apply(is_valid_fhir_datetime).all():
                        df[column] = pd.to_datetime(df[column])
                        df[column+'_year'] = df[column].dt.year
                        df[column+'_month'] = df[column].dt.month
                        df[column+'_day'] = df[column].dt.day
                        #print(f'Date Column {column} converted to Panda datetime')
                        #df[column+'_weekday'] = df[column].dt.weekday
                        #df[column+'_week'] = df[column].dt.week
                        df = df.drop(columns=column)
                 
                    else:
                        # If not all values are valid FHIR datetimes, treat as categorical
                        #print(f'Categorical Column {column} is one hot encoded')
                        encoder = OneHotEncoder(sparse=False)
                        transformed = encoder.fit_transform(df[column].values.reshape(-1, 1))
                        transformed_df = pd.DataFrame(transformed, columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
                        df = pd.concat([df.drop(columns=column), transformed_df], axis=1)
    
    df = numerically_encode_column_names(df)    
    return df
    
def numerically_encode_column_names(df):
    df.columns = range(len(df.columns))
    return df
    
def dataframe_to_tensor(df):
    processed_df = process_dataframe(df)
    for column in processed_df.columns:
        #print(processed_df[column].dtype.name) 
        if processed_df[column].isna().any():
            # Impute numerical columns
            #this contain did not happen    
            num_imputer = SimpleImputer(strategy='mean')
            processed_df[column] = num_imputer.fit_transform(processed_df[[column]])
            #print(f'#######################################################Imputed Column {column}')
    
    # Converting all numerical and boolean columns to float64
    for column in processed_df.columns:
        if processed_df[column].dtype in ['int32', 'float32', 'bool']:
            processed_df[column] = processed_df[column].astype('float64')

    #print_dataframe(processed_df)
    #empty_strings = (processed_df == '').any()
    #print(empty_strings)

    try:
        tensor = torch.tensor(processed_df.values, dtype=torch.float)
        #dcolumns = processed_df.columns
        #dindex = processed_df.index
        #df_from_tensor = pd.DataFrame(tensor.numpy(), columns=dcolumns, index=dindex)
        #print('You can use the above to test whether the ##############################################')
        #print_dataframe(df_from_tensor)
    except:
        print(processed_df.dtypes)

    return tensor

def normalize_fhir(line):
        # json_data is the JSON object
        df = json_normalize(line)
        flat = flatten_json(line)
        df = pd.json_normalize(flat)
        # Store column names and index
        #columns = df.columns
        #index = df.index
        # Convert the DataFrame back to a JSON object
        #json_data = df.to_json(orient='records')
        #df = df.replace({'true': 1, 'false': 0})
        #print(json_data)
        # Assuming df is your DataFrame
        tensor = dataframe_to_tensor(df)
        #(tensor)
        # Convert tensor back to DataFrame

def print_dataframe(processed_df):
    print(processed_df.values)
    non_numeric_cols = processed_df.select_dtypes(include=['object']).columns
    print(non_numeric_cols)
    for col in non_numeric_cols:
        print(f"Unique values in {col}:", processed_df[col].unique())
    print("DF Types")
    print(processed_df.dtypes)
    print("DF Info")
    print(processed_df.info())
    print("DF Summary")
    print(processed_df.describe(include='all'))
    print("DF Is Null")
    print(processed_df.isnull().sum())
    print("Check for INF")
    print(processed_df.isin([np.inf, -np.inf]).sum())
    print("check for mixted type_cols")
    #pd.set_option('display.max_columns', None)  # None means unlimited
    mixed_type_cols = processed_df.applymap(type).nunique() > 1
    print(mixed_type_cols)

def unnest_json(df):        

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        #print(f"Index: {index}")
        #print(f"Row data: \n{row}")
        #print("-----")
        # Unnest each row and convert back to JSON
        unnested_json = df.apply(unnest_row, axis=1).to_json()
        #print(json.dumps(json.loads(unnested_json), separators=(',', ':')))
        return json.dumps(json.loads(unnested_json), separators=(',', ':'))

def fhir_to_tensor(fhir):
    df = json_normalize(fhir)
    flat = flatten_json(fhir)
    df = pd.json_normalize(flat)
    # Store column names and index
    # columns = df.columns
    #index = df.index
    # Convert the DataFrame back to a JSON object
    #json_data = df.to_json(orient='records')
    #df = df.replace({'true': 1, 'false': 0})
    #print(json_data)
    # Assuming df is your DataFrame
    tensor = dataframe_to_tensor(df)
    #print(tensor)
    #tensor = F.pad(tensor, pad=(0, 300), mode='constant', value=0)
    #print(tensor)
    # Convert tensor back to DataFrame
    return tensor

def tensor_to_fhir(tensor):
    df = tensor_to_dataframe(tensor)
     # Iterate through each row in the DataFrame
    csvuuid = uuid.uuid4()
    df.to_csv('./output/csv/' + str(csvuuid) + '.csv', sep='\t', encoding='utf-8') 
    #for index, row in df.iterrows():
        #print(f"Index: {index}")
        #print(f"Row data: \n{row}")
        #print("-----")
        # Unnest each row and convert back to JSON
        #fhiruuid = uuid.uuid4()
        #unnested_json = df.apply(unnest_row, axis=1).to_json()
        #fhir = json.dumps(json.loads(unnested_json), separators=(',', ':'))
        # Writing to sample.json
        #with open("./output/fhir/Patient/" + str(fhiruuid), "w", encoding='utf-8') as outfile:
            #outfile.write(fhir)
            #outfile.close()
def tensor_to_dataframe(tensor):
    df = pd.DataFrame(tensor.detach().numpy())
    df.columns = range(len(df.columns))
    return df
