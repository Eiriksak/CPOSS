from flask import render_template, Blueprint, request, jsonify
from datetime import datetime, date
import os, json
import folium
from folium import plugins
from folium.plugins import MarkerCluster, FastMarkerCluster, HeatMapWithTime
import datetime
import pandas as pd
import numpy as np


demo_bp = Blueprint('demo_bp', __name__)

def stringify(x):
    return str(int(x))

with open('src/static/districts.geojson') as f:
    geo_data = json.load(f)


GEOJSON = {}
for feature in geo_data['features']:
    GEOJSON[feature['properties']["dist_num"]] = feature['geometry']['coordinates']

palette = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']

def create_cbar(counts, palette):
    return {i:palette[int(j)] for i,j in enumerate(np.linspace(1,8,max(counts)))}


def generate_html(district, count, hour, date):
    hour = datetime.time(int(hour)).strftime("%I:00 %p")
    cprob = get_crime_probabilities(district, date, hour)
    li = '<ul>'
    for k,v in cprob.items():
        li += '<li>' +  str(k) + ': ' + str(v) + '</li>'
    li += '</ul>'
    html = '<h1>District ' + str(district) + '</h1>' +\
    '<h4>Crime probabilities ' + str(hour) + '</h4>' +\
    '<p> Expected: ' + str(count) + ' crimes </p>' +\
    str(li)
    return html


crime_probs = {
    "009":{
        '2019-06-26':{
            11:{
                "Theft": 0.66,
                "Narcotis": 0.21,
                "Burglary": 0.08
            }
        }
    }
}

def get_crime_probabilities(district, date, hour):
    #Get crime prob dict;
    print("Get probs", district, date, hour)
    prob = crime_probs
    return prob["009"]['2019-06-26'][11]



@demo_bp.route('/')
def main():
    return render_template("index.html")


@demo_bp.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        if request.json:
            print(request.json)
            return "Hello"
        else:
            return "Yello"


def filter_day(df,day):
    return df[df.Day == day]

def get_by_hour(df,hour):
    return df[df.Hour==hour]



@demo_bp.route('/forecast', methods=['GET', 'POST'])
def forecast():
    print("In forecast")
    if request.method == 'POST':
        if request.json:
            date = request.json["date"]
            hour = request.json["hour"]
            print(request.json)
            df = pd.read_csv("src/static/forecast.nosync.csv")
            #df = filter_day(df, datetime.datetime.strptime('2019-06-29', '%Y-%m-%d').date())
            df = filter_day(df, date)
            df = get_by_hour(df,int(hour)).reset_index()[["District","count"]]
            df["District"] = df["District"].apply(lambda x: str(int(x)))
            print(df)
            chicago = [41.85, -87.68]
            m = folium.Map(chicago,
                        zoom_start=10.4)

            plugins.Fullscreen(
                position='topright',
                title='Expand me',
                title_cancel='Exit me',
                force_separate_button=True).add_to(m)

            used_districts = []
            cbar = create_cbar(df["count"].values, palette)
            for idx,r in df.iterrows():
                _dist = r["District"]
                try: 
                    col = cbar[r["count"]]
                except KeyError:
                    col = cbar[(r["count"]-1)]
                    
                gj = folium.GeoJson(
                    data={
                        'type': 'MultiPolygon',
                        'coordinates': GEOJSON[_dist],
                    },
                    style_function = lambda feature, fillColor=col: {
                        'color': 'grey',
                        'fillColor': fillColor,
                        'weight':2,
                        'fillOpacity' : 0.6,
                        'lineOpacity': 0.3,
                        'highlight': True
                        }
                )
                html = generate_html(_dist, r["count"], idx, '2019-06-26')
                test = folium.Html(html, script=True)
                folium.Popup(test, max_width=600).add_to(gj)
                gj.add_to(m)
                used_districts.append(_dist)
                
            for d in GEOJSON:
                if d not in used_districts:
                    gj = folium.GeoJson(
                        data={
                            'type': 'MultiPolygon',
                            'coordinates': GEOJSON[d]
                        },
                        style_function = lambda feature: {
                            'color': 'grey',
                            'fillColor': palette[0],
                            'weight':2,
                            'fillOpacity' : 0.6,
                            'lineOpacity': 0.3,
                            'highlight': True
                            }
                    )
                    gj.add_to(m)
                    used_districts.append(_dist)
                

            #folium.TileLayer('openstreetmap').add_to(m)
            folium.TileLayer('cartodbpositron').add_to(m)
            folium.LayerControl().add_to(m)

            return m._repr_html_()
    else:
        return render_template("index.html")


@demo_bp.route('/true', methods=['GET', 'POST'])
def true():
    print("In true")
    if request.method == 'POST':
        if request.json:
            date = request.json["date"]
            hour = int(request.json["hour"]) + 1
            print(request.json)
            df = pd.read_csv("src/static/forecast.nosync.csv")
            #df = filter_day(df, datetime.datetime.strptime('2019-06-29', '%Y-%m-%d').date())
            df = filter_day(df, date)
            df = get_by_hour(df,int(hour)).reset_index()[["District","count"]]
            df["District"] = df["District"].apply(lambda x: str(int(x)))
            chicago = [41.85, -87.68]
            m = folium.Map(chicago,
                        zoom_start=10.4)

            plugins.Fullscreen(
                position='topright',
                title='Expand me',
                title_cancel='Exit me',
                force_separate_button=True).add_to(m)

            used_districts = []
            cbar = create_cbar(df["count"].values, palette)
            for idx,r in df.iterrows():
                _dist = r["District"]
                try: 
                    col = cbar[r["count"]]
                except KeyError:
                    col = cbar[(r["count"]-1)]
                    
                gj = folium.GeoJson(
                    data={
                        'type': 'MultiPolygon',
                        'coordinates': GEOJSON[_dist],
                    },
                    style_function = lambda feature, fillColor=col: {
                        'color': 'grey',
                        'fillColor': fillColor,
                        'weight':2,
                        'fillOpacity' : 0.6,
                        'lineOpacity': 0.3,
                        'highlight': True
                        }
                )
                html = generate_html(_dist, r["count"], idx, '2019-06-26')
                test = folium.Html(html, script=True)
                folium.Popup(test, max_width=600).add_to(gj)
                gj.add_to(m)
                used_districts.append(_dist)
                
            for d in GEOJSON:
                if d not in used_districts:
                    gj = folium.GeoJson(
                        data={
                            'type': 'MultiPolygon',
                            'coordinates': GEOJSON[d]
                        },
                        style_function = lambda feature: {
                            'color': 'grey',
                            'fillColor': palette[0],
                            'weight':2,
                            'fillOpacity' : 0.6,
                            'lineOpacity': 0.3,
                            'highlight': True
                            }
                    )
                    gj.add_to(m)
                    used_districts.append(_dist)
                

            #folium.TileLayer('openstreetmap').add_to(m)
            folium.TileLayer('cartodbpositron').add_to(m)
            folium.LayerControl().add_to(m)

            return m._repr_html_()
    else:
        return render_template("index.html")