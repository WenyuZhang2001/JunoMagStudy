import datetime
year_doy_pj = {'2016':[[240,1],[346,3]],
              '2017':[[33,4],[86,5],[139,6],[191,7],[244,8],[297,9],[350,10]],
              '2018':[[38,11],[91,12],[144,13],[197,14],[249,15],[302,16],[355,17]],
              '2019':[[43,18],[96,19],[149,20],[201,21],[254,22],[307,23],[360,24]],
               '2020':[[48,25],[101,26],[154,27],[207,28],[259,29],[312,30],[365,31]],
               '2021':[[52,32],[105,33],[159,34],[202,35],[245,36],[289,37],[333,38]],
               '2022':[[12,39],[55,40],[99,41],[142,42],[186,43],[229,44],[272,45],[310,46],[348,47]],
               '2023':[[22,48],[60,49],[98,50]]}

'''
 Notes

 N0 Orbit 0, 2 data
 Orbit 45 has joint observation with Europa, has data missing
 Orbit 47 has data missing
 
 
'''
def find_date_by_orbit(orbit_number):
    for year, data in year_doy_pj.items():
        for day_of_year, orbit in data:
            if orbit == orbit_number:
                # Create a datetime object for the given year and day of year
                date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=day_of_year - 1)
                return date.strftime('%Y-%m-%d')  # Format date as "Year-Month-Day"
    return "NAN Date"

def find_orbit_by_datetime(datetime_str):
    # Parse the datetime string to ignore time part
    date = datetime.datetime.strptime(datetime_str.split()[0], '%Y-%m-%d')
    year = date.year
    day_of_year = date.timetuple().tm_yday

    # Search for the day of year and year in the dictionary
    if str(year) in year_doy_pj:
        for day_orbit in year_doy_pj[str(year)]:
            if day_orbit[0] == day_of_year:
                return day_orbit[1]
    return "NAN Date"

def find_orbit_by_timestamp(timestamp):
    # Convert Timestamp to date and find day of the year
    year = timestamp.year
    day_of_year = timestamp.to_pydatetime().timetuple().tm_yday

    # Search for the day of year and year in the dictionary
    if str(year) in year_doy_pj:
        for day_orbit in year_doy_pj[str(year)]:
            if day_orbit[0] == day_of_year:
                return day_orbit[1]
    return "No orbit found for this date."

if __name__ == '__main__':
    Orbit = 30
    date = find_date_by_orbit(Orbit)
    print(date)

    time = '2021-02-21 03:03'
    orbit = find_orbit_by_datetime(time)
    print(orbit)

    year = 2023
    day_of_year = 136
    date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=day_of_year - 1)
    print(date.strftime('%Y-%m-%d'))
