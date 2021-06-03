# Goal
The goal of this project was to explore various datasets available online to obtain features that can be used to undergo unsupervised learning and cluster the neighborhooods of New York City (NYC). The results will be interpreted with the help of additional demographic and location data to help classify the clusters generated via various clustering methods. Clustering methods explored include KMeans, DBScan and Hierarchical. 

# Folders
## Clustering Folder
After data is extracted and processed. Store datasets are placed within the Clustering folder for ease of reference.

## Raw Data Folder
All datasets from various sources are store within 'Raw Data'. The functions will be pulling data from these files. 

'acs_combined.xlsx' is a combination of 'acs_demo_08to12_ntas.xlsx' , 'acs_select_econ_08to12_ntas.xlsx', 'acs_select_housing_08to12_ntas.xlsx', 'acs_socio_08to12_ntas.xlsx' to have all the data be located within one file for easier access. 

## Visualizations Folder
All data visulaitions used for presentations are stored here.

# Function List
## functions.py
This file stores all the functions that have been user created and used within the various notebooks to conduct tasks. 

# Data Processing and Feature Engineering
## Exploration_Results.ipynb
This notebook was used to process the data and conduct some feature engineering to obtain a dataset that can be used to conduct clustering.

# Clustering
## DBSCAN.ipynb
Exploratory notebook to explore DBSCAN clustering and find ways to optimize the results obtained from DBScan clustering. 

## Exploring Clustering.ipynb 
This notebook was used as an exploratory process to observe the results of clustering methods when features are added/removed.

# Visualizations and Interpretation 

## Clustering and Mapping.ipynb 
With the assistance of functions.py, this notebook was used to generate the final visualizations of the various clustering methods. 

## Race Demographics.ipynb
This notebook extracts demographic data for the clusters generated via clustering.

## Income Demographics.ipynb
This notebook extracts economic data for the clusters generated via clustering.

# Extra 

## Supervised_Learning.ipynb
This notebook was an exploratory process to identify if a possible regression correlation exists between the features we used to predict total population of the neighborhoods. 




