import csv
features = ['Garage Spaces:', 'Source Property Type:', 'Year Built:', 'Levels or Stories:', 'County:', 'Pool Features:']
header = ['Link', 'Price', 'Address', 'CA', 'Bedroom', 'Bathroom', 'Size', 'Lot',
    'Garage Spaces:', 'Source Property Type:', 'Year Built:', 'Levels or Stories:', 'County:', 'Pool Features:']
data = []
with open('realtor_unstructured.csv', newline='') as unsturctured:
    reader  = csv.reader(unsturctured)
    for row in reader:
        data_row = row[:8]
        for feature in features:
            feature_to_add = 'None'
            for item in row:
                if feature in item:
                    feature_to_add = item
            data_row.append(feature_to_add)
        data.append(data_row)

with open('realtor_preprocessed.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
    f.close()