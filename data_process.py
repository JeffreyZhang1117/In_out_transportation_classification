import geojson
import geopandas as gpd
import pandas as pd
import folium
import utm
from shapely import geometry
from shapely import ops
import numpy as np
import os
import warnings
from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
warnings. filterwarnings("ignore")

def utm_convert(row):
    ls = []
    for y in row['location']:
        ls.append(list(utm.from_latlon(y[1],y[0])[0:2]))
        # convert location information in latitude and longitude to Universal Transverse Mercator coordinate system (UTM). 
        # This is especially useful for large dense arrays in a small area
    return ls 

def linestring(row):
    return geometry.LineString(row.location)

def polygon(row):
    try: 
        temp = geometry.Polygon(row.location)
        return temp
    except:
        return np.nan

def load_highway(where):
    osmRoads = []
    for i in range(1,6):
        path = f'./data/geojson/{where}/map{i}.geojson'
        with open(path, encoding="utf-8") as f:
            osmlines = geojson.load(f)
        for allFeatures in osmlines.features:
            if 'highway' in allFeatures['properties']:
                roadinfo = allFeatures['properties']
        
                locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
        
                if locarr.ndim == 3:
                    locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim
        
                if locarr.ndim == 1:
                    locarr = np.array([locarr,locarr]) 
        
                roadinfo['location'] = locarr
        
                try:
                    osmRoads.append(roadinfo)
                except:
                    continue
        
    osmRoads = pd.DataFrame.from_dict(osmRoads)
    osmRoads = osmRoads.set_index('osm_id')

    osmRoads['utmLocation'] = osmRoads.apply(lambda row: utm_convert(row), axis=1) # convert to utm coordinate
    osmRoads['locationLineString'] = osmRoads.apply(lambda row: linestring(row), axis=1)
    osmRoads = osmRoads.filter(['name','highway', 'location', 'utmLocation', 'locationLineString'])

    for i in range(len(osmRoads)):
        road = osmRoads.iloc[i]
        
        if road['highway'] == 'motorway' or road['highway'] == 'motorway_junction' or road['highway'] == 'motorway_link':
            osmRoads.iloc[i]['highway'] = 'motorway'
        elif road['highway'] == 'primary' or road['highway'] == 'primary_link':
            osmRoads.iloc[i]['highway'] = 'primary_secondary'
        elif road['highway'] == 'secondary' or road['highway'] == 'secondary_link':
            osmRoads.iloc[i]['highway'] = 'primary_secondary'
        elif road['highway'] == 'tertiary' or road['highway'] == 'tertiary_link':
            osmRoads.iloc[i]['highway'] = 'tertiary'
        elif road['highway'] == 'residential' or road['highway'] == 'living_street' or road['highway'] == 'service':
            osmRoads.iloc[i]['highway'] = 'residential'
        elif road['highway'] == 'footway' or road['highway'] == 'cycleway' or road['highway'] == 'pedestrian' \
        or road['highway'] == 'path' or road['highway'] == 'steps':
            osmRoads.iloc[i]['highway'] = 'footway'
        elif road['highway'] == 'unclassified':
            osmRoads.iloc[i]['highway'] = 'unclassified'                                                       
        else:
            osmRoads.iloc[i]['highway'] = 'unknown'
       
    
    osm_roads = gpd.GeoDataFrame(osmRoads)
    osm_roads = osm_roads.rename({'locationLineString': 'geometry'}, axis = 1)
    osm_roads['geometry'] = gpd.GeoSeries(osm_roads['geometry'])
                    
    return osm_roads



def add_road_feature(osm_roads, rawdata):
    full_data = rawdata
    points_full = gpd.GeoDataFrame(full_data, geometry = gpd.points_from_xy(full_data['gpsLongitude'],full_data['gpsLatitude']))
    offset = 0.00025 # Roughly 50 meters
    bbox_full = points_full.bounds + [-offset, -offset, offset, offset]
    hits_full = bbox_full.apply(lambda row: list(osm_roads.sindex.intersection(row)), axis=1)
    
    dist_df_full = pd.DataFrame({'pt_idx':np.repeat(hits_full.index, hits_full.apply(len)), 'close_road_idx':np.concatenate(hits_full.values)})
    dist_df_full = dist_df_full.join(points_full['geometry'].rename('point'), on='pt_idx')
    dist_df_full = dist_df_full.join(osm_roads[['geometry','location', 'highway']].reset_index(drop=True), on='close_road_idx')

    dist_gdf_full = gpd.GeoDataFrame(dist_df_full)
    dist_gdf_full['distance'] = dist_gdf_full['geometry'].distance(gpd.GeoSeries(dist_gdf_full['point']))
    dist_gdf_full = dist_gdf_full.sort_values(by=['distance'])
    dist_gdf_full = dist_gdf_full.groupby('pt_idx').first()
    new_full_data = full_data.join(dist_gdf_full[['highway', 'distance', 'close_road_idx']])
    new_full_data['highway'] = new_full_data['highway'].fillna('unknown')
    new_full_data['distance'] = new_full_data['distance'].fillna(0.005)
    
    return new_full_data

def load_landuse(where):
    osmLands = []
    for i in range(1,6):
        path = f'./data/geojson/{where}/map{i}.geojson'
        with open(path, encoding="utf-8") as f:
            osmlines = geojson.load(f)
        for allFeatures in osmlines.features:
            if 'landuse' in allFeatures['properties']:
                ## OUTLAND
                if allFeatures["properties"]["landuse"] == "grass" or allFeatures["properties"]["landuse"] == "basin" \
                or allFeatures["properties"]["landuse"] == "forest" or allFeatures["properties"]["landuse"] == "greenfield" \
                or allFeatures["properties"]["landuse"] == "meadow" or allFeatures["properties"]["landuse"] == "orchard" \
      or allFeatures["properties"]["landuse"] == "plant_nursery" or allFeatures["properties"]["landuse"] == "recreation_ground"\
      or allFeatures["properties"]["landuse"] == "village_green" or allFeatures["properties"]["landuse"] == "wasteland"\
      or allFeatures["properties"]["landuse"] == "farmland" or allFeatures["properties"]["landuse"] == "farmyard":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'outland'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
            
                ## BUILDING
                elif allFeatures["properties"]["landuse"] == "commercial" or allFeatures["properties"]["landuse"] == "retail":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'building'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
   
                ## RESIDENTIAL        
                elif allFeatures["properties"]["landuse"] == "residential":
                    landinfo = allFeatures['properties']
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo) 
        
            if "leisure" in allFeatures['properties']:
                ## OUTLAND(PARK)
                if allFeatures["properties"]["leisure"] == "park" or allFeatures["properties"]["leisure"] == "garden"\
                or allFeatures["properties"]["leisure"] == "pitch" or allFeatures["properties"]["leisure"] == "playground"\
                or allFeatures["properties"]["leisure"] == "recreation_ground":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'outland'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
                
            if "building" in allFeatures["properties"]:
                ## RESIDENTIAL
                if allFeatures["properties"]["building"] == "house" or allFeatures["properties"]["building"] == "apartments" \
            or allFeatures["properties"]["building"] == "residential" or allFeatures["properties"]["building"] == "detached"\
            or allFeatures["properties"]["building"] == "hotel":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'residential'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
            
                ## BUILDING
                elif allFeatures["properties"]["building"] == "yes" or allFeatures["properties"]["building"] == "university" \
            or allFeatures["properties"]["building"] == "church" or allFeatures["properties"]["building"] == "office" \
            or allFeatures["properties"]["building"] == "museum" or allFeatures["properties"]["building"] == "school"\
            or allFeatures["properties"]["building"] == "commercial" or allFeatures["properties"]["building"] == "retail":
                    landinfo = allFeatures['properties']
                    landinfo["landuse"] = 'building'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
                    if locarr.ndim > 2:
                        locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim        
                    if locarr.ndim == 1:
                        locarr = np.array([locarr,locarr])        
                    if locarr.shape[0] > 2:
                        landinfo['location'] = locarr
                        osmLands.append(landinfo)
                    
    osmLands = pd.DataFrame.from_dict(osmLands)
    osmLands = osmLands.set_index('osm_id')
    
    # Filtering and pre-processing 
    # osmLands = osmLands[osmLands['type'] != 'multipolygon']
    #osmLands['utmLocation'] = osmLands.apply(lambda row: utm_convert(row), axis=1) # convert to utm coordinate
    osmLands['locationPolygon'] = osmLands.apply(lambda row: polygon(row), axis=1)
    # construct the line using a list of coordinate-tuples
    osmLands = osmLands.filter(['name','landuse', 'location', 'locationPolygon'])
    # only care about related attributes 
    osmLands.dropna(subset=['locationPolygon'], inplace=True)
    osm_lands = gpd.GeoDataFrame(osmLands)
    osm_lands = osm_lands.rename({'locationPolygon': 'geometry'}, axis = 1)
    osm_lands['geometry'] = gpd.GeoSeries(osm_lands['geometry'])
    return osm_lands

def add_land_feature(osm_lands, data):
    full_data = data
    points_full = gpd.GeoDataFrame(full_data, geometry = gpd.points_from_xy(full_data['gpsLongitude'],full_data['gpsLatitude']))
    offset = 0.00025 # Roughly 50 meters
    bbox_full = points_full.bounds + [-offset, -offset, offset, offset]
    hits_full = bbox_full.apply(lambda row: list(osm_lands.sindex.intersection(row)), axis=1)
    
    dist_df_full = pd.DataFrame({'pt_idx':np.repeat(hits_full.index, hits_full.apply(len)), 'close_land_idx': np.concatenate(hits_full.values)})
    dist_df_full = dist_df_full.join(full_data['geometry'].rename('point'), on='pt_idx')
    dist_df_full = dist_df_full.join(osm_lands[['geometry','location', 'landuse']].reset_index(drop=True), on='close_land_idx')

    dist_gdf_full = gpd.GeoDataFrame(dist_df_full)
    dist_gdf_full['if_contain'] = dist_gdf_full['geometry'].contains(gpd.GeoSeries(dist_gdf_full['point']))
    
    contain = dist_gdf_full[dist_gdf_full['if_contain'] == True]
    df_contain = contain[['pt_idx', 'landuse', 'close_land_idx']].set_index('pt_idx')
    new_full_data = full_data.join(df_contain)
    new_full_data['landuse'].fillna('unknown', inplace=True)
    
    return new_full_data

def load_station (where):
    osmStation = []
    for i in range(1,6):
        path = f'./data/geojson/{where}/map{i}.geojson'
        with open(path, encoding="utf-8") as f:
             osmlines = geojson.load(f)
        for allFeatures in osmlines.features:
            if 'highway' in allFeatures['properties']:
               if allFeatures["properties"]["highway"] == "bus_stop":
                  roadinfo = allFeatures['properties']
                  roadinfo['station'] = 'bus_stop'
                  locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
        
                  if locarr.ndim == 3:
                     locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim
                  if locarr.ndim == 1:
                     locarr = np.array([locarr,locarr]) 
                  roadinfo['location'] = locarr
                  osmStation.append(roadinfo)
            if 'railway' in allFeatures['properties']:
                if allFeatures["properties"]["railway"] == "station" or allFeatures["properties"]["railway"] == 'halt' \
                or allFeatures["properties"]["railway"] == "platform" or allFeatures["properties"]["railway"] == "stop":
                    roadinfo = allFeatures['properties']
                    roadinfo['station'] = 'railway_station'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
        
                    if locarr.ndim == 3:
                       locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim
                    if locarr.ndim == 1:
                       locarr = np.array([locarr,locarr]) 
                    roadinfo['location'] = locarr
                    osmStation.append(roadinfo)
                       
                elif allFeatures["properties"]["railway"] == "subway_entrance":
                    roadinfo = allFeatures['properties']
                    roadinfo['station'] = 'subway_station'
                    locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
        
                    if locarr.ndim == 3:
                       locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim
                    if locarr.ndim == 1:
                       locarr = np.array([locarr,locarr]) 
                    roadinfo['location'] = locarr
                    osmStation.append(roadinfo)
            if 'station' in allFeatures['properties']:
                if allFeatures["properties"]["station"] == 'subway':
                   roadinfo = allFeatures['properties']
                   roadinfo['station'] = 'subway_station'
                   locarr = np.array(allFeatures['geometry']['coordinates'], dtype="object")
        
                   if locarr.ndim == 3:
                      locarr = locarr.reshape(-1, locarr.shape[-1])  # 3-dim -> 2-dim
                   if locarr.ndim == 1:
                      locarr = np.array([locarr,locarr]) 
                   roadinfo['location'] = locarr
                   osmStation.append(roadinfo)
    osmStation = pd.DataFrame.from_dict(osmStation)
    osmStation = osmStation.set_index('osm_id')
    # Filtering and pre-processing 
    # osmRoads = osmRoads[osmRoads['type'] != 'multipolygon']
    osmStation['utmLocation'] = osmStation.apply(lambda row: utm_convert(row), axis=1) # convert to utm coordinate
    osmStation['locationLineString'] = osmStation.apply(lambda row: linestring(row), axis=1)
    # construct the line using a list of coordinate-tuples
    osmStation = osmStation.filter(['name','station', 'location', 'utmLocation', 'locationLineString'])
    # only care about related attributes 
    osm_station = gpd.GeoDataFrame(osmStation)
    osm_station = osm_station.rename({'locationLineString': 'geometry'}, axis = 1)
    osm_station['geometry'] = gpd.GeoSeries(osm_station['geometry'])
    return osm_station



def add_station_feature(osm_station, data):
    full_data = data
    full_data = min_max_norm(full_data, 'temperature')
    full_data = min_max_norm(full_data, 'humidity')
    # Now convert this to a points geoDataFrame
    points_full = gpd.GeoDataFrame(full_data, geometry = gpd.points_from_xy(full_data['gpsLongitude'],full_data['gpsLatitude']))
    offset = 0.00025 # Roughly 50 meters
    bbox_full = points_full.bounds + [-offset, -offset, offset, offset]
    hits_full = bbox_full.apply(lambda row: list(osm_station.sindex.intersection(row)), axis=1)
    dist_df_full = pd.DataFrame({'pt_idx':np.repeat(hits_full.index, hits_full.apply(len)), 'close_station_idx': np.concatenate(hits_full.values)})
    dist_df_full = dist_df_full.join(points_full['geometry'].rename('point'), on='pt_idx')
    dist_df_full = dist_df_full.join(osm_station[['geometry','location', 'station']].reset_index(drop=True), on='close_station_idx')

    dist_gdf_full = gpd.GeoDataFrame(dist_df_full)
    dist_gdf_full['station_distance'] = dist_gdf_full['geometry'].distance(gpd.GeoSeries(dist_gdf_full['point']))
    dist_gdf_full = dist_gdf_full.sort_values(by=['station_distance'])
    dist_gdf_full = dist_gdf_full.groupby('pt_idx').first()
    new_full_data = full_data.join(dist_gdf_full[['station', 'station_distance', 'close_station_idx']])
    new_full_data['station'] = new_full_data['station'].fillna('none')
    new_full_data['station_distance'] = new_full_data['station_distance'].fillna(0.010)
   
    return new_full_data        


def distance_euclidean(data):
    '''
    Calculate the Euclidean distance between two GPS points based on the longitude and latitude
    :param data: DataFrame --> Needs to include gpsLongitude and gpsLatitude features
    :return: DataFrame with gps_distance included as additional feature
    '''

    N = data.shape[0]
    gps_dist = [0] * N
    data['gps_dist'] = gps_dist

    for n in range(1, N):
        data['gps_dist'].iloc[n] = np.sqrt((data['gpsLongitude'].iloc[n-1] - data['gpsLongitude'].iloc[n])**2 \
                                               + (data['gpsLatitude'].iloc[n-1] - data['gpsLatitude'].iloc[n])**2)

    return data

def distance_euclidean_home(data,gpsLongitude,gpsLatitude):
    '''
    Calculate the Euclidean distance between two GPS points based on the longitude and latitude
    :param data: DataFrame --> Needs to include gpsLongitude and gpsLatitude features
    :return: DataFrame with gps_distance included as additional feature
    '''

    N = data.shape[0]
    gps_dist = [0] * N
    data['home_dist'] = gps_dist

    for n in range(0, N):
        data['home_dist'].iloc[n] = np.sqrt((data['gpsLongitude'].iloc[n] - gpsLongitude)**2 \
                                               + (data['gpsLatitude'].iloc[n] - gpsLatitude)**2)

    return data


def distance_euclidean_work(data,gpsLongitude,gpsLatitude):
    '''
    Calculate the Euclidean distance between two GPS points based on the longitude and latitude
    :param data: DataFrame --> Needs to include gpsLongitude and gpsLatitude features
    :return: DataFrame with gps_distance included as additional feature
    '''

    N = data.shape[0]
    gps_dist = [0] * N
    data['work_dist'] = gps_dist

    for n in range(0, N):
        data['work_dist'].iloc[n] = np.sqrt((data['gpsLongitude'].iloc[n] - gpsLongitude)**2 \
                                               + (data['gpsLatitude'].iloc[n] - gpsLatitude)**2)

    return data

def calculate_std(data, column_name, k=10):
    '''
    Calculates the standard deviation of a given columns
    :param data: DataFrame --> data
    :param column_name: String --> Column name for the standard deviation is to be calculated
    :param k: Int --> Window size
    :return: DataFrame --> Data with additional column for the standard deviation (column_name_std)
    '''
    N = data.shape[0]
    var = [0] * N
    data[column_name + '_std'] = var

    n = k

    while n < N:
        data[column_name + '_std'].iloc[n] = np.std(data[column_name].iloc[n-k:n])
        n += 1
    return data

def view_missing_value(df):
    for column in list(df.columns):
        print("{}:  {} % missing values \n".format(column, ((len(df) - df[column].count()) / len(df))*100))
        
def min_max_norm(data, col):
    target_col = data[col]
    max_num = max(target_col.dropna())
    min_num = min(target_col.dropna())
    std = (target_col - min_num) / (max_num - min_num)
    data[col] = std
    
    return data

def plot_confusionMatrix(y_true, y_pred,  normalize=None,
                          title='Confusion Matrix', plot_numbers=False, display_names=None,
                          figsize=(18, 12)):

    cm = confusion_matrix(y_true, y_pred,  normalize=normalize)
    

    df_cm = pd.DataFrame(cm, index=display_names, columns=display_names).round(2)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=plot_numbers, cmap='Blues', fmt='.2%')
    plt.setp(plt.gca().get_xticklabels(), ha="right", rotation=45, fontsize=16)
    plt.setp(plt.gca().get_yticklabels(), fontsize=16)
    plt.ylabel('True Label', fontsize = 18)
    plt.xlabel('Predicted Label', fontsize = 18)
    plt.title(title, fontsize = 18)
    #return fig

# add time-series features
def add_time_feature_train(data, t=10):
    try:
        data.sort_values(by=['phoneTimestamp'], inplace=True)  
    except: 
        data.sort_values(by=['timestamp'], inplace=True) 
    time_data = data.iloc[0:2]
    time_data['avg_dist'] = 0
    time_data['same_land'] = 0
    time_data['same_road'] = 0
    for i in range(len(data) - t):
        if (len(data[i:i+t+1]['file label'].unique()) == 1):
            temp = data.iloc[i]
            if ((len(data[i:i+t+1]['close_road_idx'].unique()) == 1) and data.iloc[i]['close_road_idx'] != 0):
                temp['same_road'] = 1
            else:
                temp['same_road'] = 0
            if ((len(data[i:i+t+1]['close_land_idx'].unique()) == 1) and data.iloc[i]['close_land_idx'] != 0):
                temp['same_land'] = 1
            else:
                temp['same_land'] = 0  
            temp['avg_dist'] = data[i:i+t+1]['gps_dist'].mean()
            time_data = time_data.append(temp)
        
    time_data = time_data.iloc[2:]  
    time_data['highway'] = time_data['highway'].astype('category')
    time_data['highway_encode'] = time_data['highway'].cat.codes
    time_data['landuse'] = time_data['landuse'].astype('category')
    time_data['landuse_encode'] = time_data['landuse'].cat.codes
    time_data['station'] = time_data['station'].astype('category')
    time_data['station_encode'] = time_data['station'].cat.codes

    return time_data

# add time_Series for inhale data
def add_time_feature_station(data, t=10):
    try:
        data.sort_values(by=['phoneTimestamp'], inplace=True)
    except:
        data.sort_values(by=['timestamp'], inplace=True)
    time_data = data.iloc[0:2]
    time_data['avg_dist'] = 0
    time_data['same_land'] = 0
    time_data['same_road'] = 0
    for i in range(len(data) - t):
        temp = data.iloc[i]
        if ((len(data[i:i+t+1]['close_road_idx'].unique()) == 1) and data.iloc[i]['close_road_idx'] != 0):
            temp['same_road'] = 1
        else:
            temp['same_road'] = 0
        if ((len(data[i:i+t+1]['close_land_idx'].unique()) == 1) and data.iloc[i]['close_land_idx'] != 0):
            temp['same_land'] = 1
        else:
            temp['same_land'] = 0
        temp['avg_dist'] = data[i:i+t+1]['gps_dist'].mean()
        time_data = time_data.append(temp)

    time_data = time_data.iloc[2:]
    time_data['highway'] = time_data['highway'].astype('category')
    time_data['highway_encode'] = time_data['highway'].cat.codes
    time_data['landuse'] = time_data['landuse'].astype('category')
    time_data['landuse_encode'] = time_data['landuse'].cat.codes
    time_data['station'] = time_data['station'].astype('category')
    time_data['station_encode'] = time_data['station'].cat.codes

    return time_data

def transport_time_adjustment(data):
    data1 = data
    data1['timestamp'] = pd.to_datetime(data1['timestamp'])
    data1 = data1.sort_values(by='timestamp')
    data1['hour'] = data1['timestamp'].dt.hour
    data1 = data1.loc[(data1['hour'] >= 6)]
    data1 = data1.loc[(data1['hour'] <= 19)]
    return data1
        

