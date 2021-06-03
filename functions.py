#import libraries for functions
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import functions as f
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

##Global Variables
# define Geographic 2D Coodinate system (CRS: EPSG:4326)
crs_4326 = CRS("WGS84")

# define neighborhood (nb_df) GeoDataFrame
neighborhood = gpd.read_file('Raw Data/Neighborhood Tabulation Areas (NTA)')
nb_df = neighborhood[['ntacode', 'ntaname', 'geometry']]





def demo_data(features, filepath):

    ''' 
    Accesses data within excel file 'acs_combined.xlsx' to return requested features.
    
    Parameters:
    features(list): list of features in str format to be obtained from the data within the excel file 'acs_combined.xlsx'.
    filepath(string): filepath to excel file 'acs_combined.xlsx'.
    
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





def identify_neighborhood(df):
    '''
    Returns a GeoDataFrame of feature counts labeled by NYC neighborhood.
    
        Parameters:
            df (DataFrame): a dataframe for feature that includes corresponding geographic info.
                            must include columns 'Latitude' and 'Longitude'.
            
        Returns:
            df_with_neighborhood (GeoDataFrame): new GeoDataFrame with neighborhood code and name added as columns to label feature location within neighborhood.
    '''
    # make all column names lower case, and remove all irrelevant columns
    df.columns = df.columns.str.lower()
    df = df[['longitude', 'latitude']]
    # now drop null columns if pressent
    df = df.dropna()
    
    # create shapely Point info from Lat & Long
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
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





def visualize_clusters(df, clusters_array):
    """
    Returns a plot of color-coded clustered neighborhoods on map of NYC.
    
        Parameters:
            df (DataFrame): dataframe used for clustering, prior to scaling.
            clusters_array (list): list of assigned cluster values.
            
        Returns:
            Plot of clustered NYC neighborhoods.
    """
    # copy df
    cluster_result_df = df.copy()
    # add column of cluster values to cluster result DataFrame
    cluster_result_df['cluster'] = clusters_array
    
    # creat list to match neighborhood with each observation 
    nb_polygon = []
    for i in range(len(cluster_result_df)):
        try:
            for j in range(len(nb_df)):
                if cluster_result_df.index[i] == nb_df.ntacode[j]:
                    nb_polygon.append(nb_df.geometry[j])
                    break
        except:
            pass
        
    # add column of neighborhoods to cluster result DataFrame
    cluster_result_df['geometry'] = nb_polygon
    
    # convert cluster result to GeoDataFrame
    gdf = gpd.GeoDataFrame(cluster_result_df, geometry=cluster_result_df.geometry, crs="EPSG:4326")
    
    # plot on NYC map
    fig, ax = plt.subplots(figsize=(15,15))
    gdf.plot(ax=ax, column='cluster', cmap='Dark2', categorical=True, legend=True)
    plt.show()
    
    fig.clf()
    

    
    
    
def plotter(df,method):
    """
    Returns a color-coded clustered neighborhoods on map of NYC labled via colors followed with a Radar Chart identifying the characteristics of each cluster.
    
        Parameters:
            df (DataFrame): Unscaled df containing the features required to be passed to create the clusters.
            method (str): method to clsuter neighborhoods by. Valid choices are 'kmeans', 'heirarchical' and 'dbscan'.
            
        Outputs:
            Plot of clustered NYC neighborhoods and radar chart. 
    """
    
    # Create a new scaled dataset
    new_df = pd.DataFrame(StandardScaler(with_mean = False).fit_transform(df),columns = df.columns)
    features = list(df.columns)
    
    # Check method input to decide what type of clustering to undergo
    if method == 'kmeans':
        clusters = 5
        km = KMeans(n_clusters=clusters,n_init=20)
        y = km.fit_predict(new_df)
        
    elif method == 'heirarchical':
        clusters = 5
        ac = AgglomerativeClustering(affinity='euclidean', linkage='ward', n_clusters = clusters)
        y = ac.fit_predict(new_df)
        
    elif method == 'dbscan':
        # Use Knee Method to identify an optimal value for epsilon
        n = 2
        nearest_neighbors = NearestNeighbors(n_neighbors=n)
        neighbors = nearest_neighbors.fit(new_df)
        distances, indices = neighbors.kneighbors(new_df)
        distances = np.sort(distances[:,n-1], axis=0)
        x = np.arange(len(distances))
        knee = KneeLocator(x, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        eps = distances[knee.knee]
            
        db = DBSCAN(eps, min_samples=2)
        y = db.fit_predict(new_df)
        # Unlike KMeans and Heirarchical, amount of clusters varies, so amount of clusters are identified which is used later for graphing
        clusters = len(np.unique(y))
        # Have to increase the value of cluster number as DBScan number defaults to a -1 start rather than 0
        y += 1
        
    else:
        return "Not a valid method, valid choices are 'kmeans', 'heirarchical' and 'dbscan'"
    
    # Add Cluster Number to dataframe to assign each neighborhood with a Cluster, a 1 is added as default cluster numbering begins at 0
    new_df['Cluster'] = y+1
    # Call function to plot the NYC map with clusters labeled via colors
    visualize_clusters(df,y+1)

    # Radar chart generation
    fig = go.Figure()
    # Create a color list to match the Dark 2 colormap scheme, used when creating the Radar Charts to get them to match with the map
    color = ['#1B9E77','#D95F02','#7570B3','#E7298A','#66A61E','#E6AB02','#A6761D','#666666'] 
    # Iterate to plot each radar chart on top of each other
    for j in range(clusters):
        fig.add_trace(go.Scatterpolar(r=list(new_df[new_df['Cluster'] == j+1].mean()), theta=features, fill='toself', fillcolor = color[j], opacity=0.6, name='Cluster '+ str(j+1), line={'color':color[j]}))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 6])), showlegend=True)
    fig.show()