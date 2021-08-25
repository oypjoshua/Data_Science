from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import datetime

headers = {'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
'sec-ch-ua-mobile': '?0',
'Upgrade-Insecure-Requests': '1',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36'
}
#'Accept-Encoding': 'identity'

url_heading = 'https://www.realtor.com/realestateandhomes-search/California'
data = []
exceptions = 0

for i in range(1,51):
    print('page' + str(i))
    url = url_heading + '/pg-' + str(i)
    response=requests.get(url,headers=headers)
    print(datetime.datetime.now())
    print(response)
    soup=BeautifulSoup(response.content,'lxml')
    for item in soup.select('.component_property-card'):
        price = 'None'
        address = 'None'
        beds = 'None'
        baths = 'None'
        sqft = 'None'
        lot = 'None'
        try:
            link = 'https://www.realtor.com'+ item.select('a[href]')[0]['href']
        except:
            exceptions += 1
        
        try:
            price = item.select('[data-label=pc-price-wrapper]')[0].get_text()
        except:
            exceptions += 1
        
        try:
            address = item.select('[data-label=pc-address]')[0].get_text()
        except:
            exceptions += 1
        
        try:
            beds = item.select('[data-label=pc-meta-beds]')[0].get_text()
        except:
            exceptions += 1

        try:    
            baths = item.select('[data-label=pc-meta-baths]')[0].get_text()
        except:
            exceptions += 1
        
        try:
            sqft = item.select('[data-label=pc-meta-sqft]')[0].get_text()
        except:
            exceptions += 1
        
        try:
            lot = item.select('[data-label=pc-meta-sqftlot]')[0].get_text()
        except:
            exceptions += 1
        individual_data = [link, price, address, beds, baths, sqft, lot]
        data.append(individual_data)

print(exceptions)
csv_heading = ['link', 'price', 'address', 'beds', 'baths', 'sqft', 'lot']

with open('data.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(csv_heading)

    # write multiple rows
    writer.writerows(data)
    f.close()