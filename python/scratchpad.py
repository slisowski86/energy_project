import requests
import pandas as pd
import os
from weather import download_weather_data
from api_keys import *
import holidays


def geolocation_per_city(city, state):
    location = city + ' ' + state
    api_key = GOOGLE_API
    payload = {'address': location,
               'key': api_key}

    headers = {'Accept-Encoding': 'gzip'}
    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    r = requests.get(url=url, params=payload, headers=headers)
    result = r.json()
    latitude = round(result['results'][0]['geometry']['location']['lat'], 2)
    longitude = round(result['results'][0]['geometry']['location']['lng'], 2)

    return [latitude, longitude]


def check_if_holiday(date, state):
    state_holidays =  holidays.CountryHoliday('US', state=state)
    holiday = False
    if date in state_holidays:
        holiday = True
    weekend = False
    if date.weekday() > 4:
        weekend = True

    if holiday or weekend:
        return True
    else:
        return False


def to_date(timestamp):
    return timestamp.date()


def get_valid_station_by_geolocation_per_dataid(lat, lng, date):
    geocode = str(lat) + ',' + str(lng)
    product = 'pws'
    frmt = 'json'
    api_key = WEATHER_API
    payload = {'geocode': geocode, 'product': product, 'format': frmt, 'apiKey': api_key}

    headers = {'Accept-Encoding': 'gzip, deflate, br'}

    url = 'https://api.weather.com/v3/location/near'

    r = requests.get(url=url, params=payload, headers=headers)
    response = r.json()
    stations = response['location']['qcStatus']
    for index, value in enumerate(stations):
        if value == 1:
            station_id = response['location']['stationId'][index]
            if download_weather_data(str(date).replace('-', ''), station_id) == 200:
                return station_id

    return 'None'


def create_geolocation_frame(dataid, metadata, id_full_df, lenght=0, exists=False):
    #Odczyt istniejącej bazy jeśli istnieje
    if exists:
        df = pd.read_csv('data/geolocation' + str(dataid) + '.csv', index_col=0)
    else:
        # Utworzenie dataframe zawierającego date, nazwe miasta, stan, dlugość i szerokość gograficzną, id stacji pogodowej
        df = pd.DataFrame(columns=['date', 'city', 'state', 'lat', 'lng', 'station_id'])
        # Utworzenie pustego csv z headerami
        df.to_csv('data/geolocation' + str(dataid) + '.csv')
    # Stworzenie pomocniczego df
    support_df = pd.DataFrame(columns=['date', 'city', 'state', 'lat', 'lng', 'station_id'])
    # Kopia i odzyskanie unikalnych dat z pliku źródłowego
    dates = id_full_df.local_15min.copy()
    dates = dates.apply(to_date)
    support_df.date = dates.unique()
    support_df = support_df[lenght:]
    # Pobranie nazwy miasta i stanu
    support_df['city'] = metadata.loc[metadata.dataid == dataid].city.values[0]
    support_df['state'] = metadata.loc[metadata.dataid == dataid].state.values[0]
    support_df['lat'], support_df['lng'] =  geolocation_per_city(support_df['city'][lenght], support_df['state'][lenght])
    # Uzupełnienie dataframe
    for index, row in support_df.iterrows():
        row['station_id'] = get_valid_station_by_geolocation_per_dataid(row['lat'], row['lng'], row['date'])
        pd.DataFrame(row).T.to_csv('data/geolocation' + str(dataid) + '.csv', mode='a', header=False)


    # Zapisanie danych do pliku
    #df.to_csv('data/geolocation' + str(dataid) + '.csv')
    # final_df = df.append(support_df, ignore_index=True)
    # print('final_df', final_df.shape)
    # print(final_df)
    # if exists is False:
    return pd.read_csv('data/geolocation' + str(dataid) + '.csv', index_col=0)
    # else:
    #     return final_df


def get_location_and_station_per_dataid(metadata, id_full_df):

    dataid = id_full_df.dataid.values[0]

    # sprawdzenie czy istnieje już plik z danymi geolokalizacyjnymi
    if os.path.exists('data/geolocation'+str(dataid)+'.csv'):
        geoloc =  pd.read_csv('data/geolocation'+str(dataid)+'.csv', index_col=0)
        dates = id_full_df.local_15min.copy()
        dates = dates.apply(to_date)
        dates = dates.unique()
        print(len(geoloc), len(dates))
        if len(geoloc) == len(dates):
            return geoloc
        else:
            return create_geolocation_frame(dataid, metadata, id_full_df, lenght=len(geoloc), exists=True)
    else:
        return create_geolocation_frame(dataid, metadata, id_full_df)


def location_and_station_per_dataid(date, cities_frame):
    return cities_frame.loc[cities_frame['date']==date, ['station_id', 'lat', 'lng']].values[0]
