# Code from mapping notebook (could put this in a .py file)
from sqlite3 import Timestamp
from geopy import distance
import folium

def getCoords(df, timestamp=False):
    # return list of lists, each inside list contains the coordinate of one data point
    coords = df.filter(['gpsLatitude', 'gpsLongitude'])
    coords = list(coords.to_records(index=timestamp))
    return [list(c) for c in coords]

def getMap(coords, df, label_col, transport=False, classification=False, line=False):
    # draw the map, if classification=True, draw different class with different colors
    # Maps © www.thunderforest.com, Data © www.osm.org/copyright
    m = folium.Map(location=coords[0], zoom_start=12)
    if transport:
        m = folium.Map(location=coords[0], zoom_start=12, tiles = 'https://tile.thunderforest.com/transport/{z}/{x}/{y}.png?apikey=79284124c2f24955b9cd0d84c306bd0d', attr= "Transport")
    cs = []
    colour = 'red'
    for i,c in enumerate(coords):
        if classification==True:
            io = df.iloc[i][label_col]
            if io == 0: ## indoor
                colour = 'crimson'
            elif io == 'Indoor':
                colour = 'crimson' 
            elif io == 1: ## commuting
                colour = 'darkblue'
            elif io == 2:
                colour = 'darkblue'

            
        coord = c
        cs.append(coord)
        # time = df.index[i].strftime(("%H:%M:%S")) 
        label = "Coords: {0}".format(c)
        if classification== True:
           label += "\n Classification: {0}".format(io)
      
        # label += "\n Timestamp: {0}".format(time)

        
        folium.Circle(
            radius=10,
            location=coord,
            popup=label,
            color=colour,
            fill=True,
        ).add_to(m)
    
    if line==True:
        folium.PolyLine(cs).add_to(m)
    return m
                    