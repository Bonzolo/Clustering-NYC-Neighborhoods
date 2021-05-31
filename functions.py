#import libraries for functions
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS

def demo_data(features, filepath):

    ''' 
    Accesses data within excel file 'acs_combined.xlsx' to return requested features.
    
    Parameters:
    features(list): list of features in str format to be obtained from the data within the excel file 'acs_combined.xlsx'
    filepath(string): filepath to excel file 'acs_combined.xlsx'
    
    Returns: 
    DataFrame(df) with features passed listed as the columns and NTA Codes of neighborhoods as the index. 
    '''
    
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

# define Geographic 2D Coodinate system (CRS: EPSG:4326)
crs_4326 = CRS("WGS84")

# define neighborhood (nb_df) GeoDataFrame
neighborhood = gpd.read_file('Raw Data/Neighborhood Tabulation Areas (NTA)')
nb_df = neighborhood[['ntacode', 'ntaname', 'geometry']]

def identify_neighborhood(df):
    '''
    Returns a GeoDataFrame with features labeled by NYC neighborhood.
    
        Parameters:
            df (DataFrame): a pandas dataframe with feature and corresponding geographic info.
                            must include columns 'Latitude' and 'Longitude'.
            
        Returns:
            df_with_neighborhood (GeoDataFrame): new GeoDataFrame with neighborhood code and name added as columns to label feature location within neighborhood.
    '''
    # drop null columns if pressent
    df = df.dropna()
    
    # create shapely Point info from Lat & Long
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    # add Points column
    geo_df = gpd.GeoDataFrame(df, crs=crs_4326, geometry=geometry)
    
    # join feature with neighborhood GeoDataFrame 
    df_with_neighborhood = gpd.sjoin(geo_df, nb_df, how="inner", op='intersects')
    
    # set ntacode as index
    df_with_neighborhood.index = df_with_neighborhood['ntacode']
    df_with_neighborhood.index.name = 'ntacode'
    df_with_neighborhood.drop('ntacode',axis = 1,inplace=True)
    
    # ask for input value to name feature output column
    column_name = input('Feature column name: ')
    
    #output feature count with neighborhood code
    df_with_neighborhood = df_with_neighborhood.sort_values('ntacode')[['ntaname']].groupby('ntacode').count()
    df_with_neighborhood = df_with_neighborhood.rename(columns={'ntaname': column_name})
    
    return df_with_neighborhood