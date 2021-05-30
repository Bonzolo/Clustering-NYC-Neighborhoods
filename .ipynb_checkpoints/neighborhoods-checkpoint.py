def demographics(features, filepath):

    ''' 
    Inputs:
    features(list): list of features in str format to be obtained from the data within the excel file 'acs_combined.xlsx'
    filepath(string): filepath to excel file 'acs_combined.xlsx'
    
    Outputs: 
    DataFrame(df) with features passed listed as the columns and NTA Codes of neighborhoods as the index. 
    '''
    import pandas as pd
    
    #Creating list to access specific columns within the Excel dataset.
    est_range= list(range(1,780,4))
    est_range.insert(0,0)

    # Read excel files, dropping all empty cells and duplicates. 
    # Indexes are set to 'Unnamed: 0' as this is the column name that contains all the feature labels.
    # Dataset created has feature labels on the index and neighborhoods as the column headers.
    estimates = pd.read_excel(filepath,header = 6,usecols = est_range).dropna().set_index('Unnamed: 0').drop_duplicates()

    # Removing unncessary 'Unnamed 0' index header and removing all unwanted trailing and preceding spaces within the strings of the column and index labels. 
    estimates.index.name = None
    estimates.index = estimates.index.str.strip()
    estimates.columns = estimates.columns.str.strip()
    
    # Reading Excel file to obtain neighborhood names and NTA codes.
    # Set index to NTA Codes
    neighborhoods = pd.read_excel(filepath,sheet_name = 'Heading')
    neighborhoods.drop_duplicates(inplace = True)
    neighborhoods.index = neighborhoods['NTA Code']
    neighborhoods.index.name = None
    neighborhoods.drop('NTA Code',axis = 1,inplace=True)
    
    #matching index of estimates with neighborhoods to prepare for merging.
    estimates = estimates.T
    estimates = estimates[features]
    estimates.index = neighborhoods.index
    
    return neighborhoods.merge(estimates, left_index=True,right_index = True)