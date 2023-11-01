import sys
if sys.version_info[0]==3:
    import urllib.request
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {'http': 'http://brd-customer-hl_5e5f6fba-zone-residential:2alnjvvbx31p@brd.superproxy.io:22225',
            'https': 'http://brd-customer-hl_5e5f6fba-zone-residential:2alnjvvbx31p@brd.superproxy.io:22225'}))
    print(opener.open('http://lumtest.com/myip.json').read())


import pandas as pd
import re
import os 


# Read original CSV
df = pd.read_csv('.\Projects/kilter/grouped_instagram_links.csv.csv')

# Initialize dictionary for grouped data
grouped_data = {}

# Regex pattern for Instagram URLs
pattern = r'https://www\.instagram\.com/p/([\w-]+)/'

# Loop through rows to filter and group data
for index, row in df.iloc[:].iterrows():
    input_string = row['link_and_username']
    climb_name = row['name']

    if isinstance(input_string, str):
        matches = re.findall(pattern, input_string)
        if matches:
            for match in matches:
                link = f"https://www.instagram.com/p/{match}/"
                if climb_name not in grouped_data:
                    grouped_data[climb_name] = []
                grouped_data[climb_name].append(link)

# Convert dictionary to DataFrame
grouped_df = pd.DataFrame(list(grouped_data.items()), columns=['name', 'links'])

# Save DataFrame to CSV
grouped_df.to_csv('.\grouped_instagram_links.csv', index=False)
print(grouped_df.head())
for index, row in grouped_df.iloc[:].iterrows():
    print(row['name'])
    row['name'] = row['name'].replace('/', '').replace(' ', '_').replace(':', '').replace('?', '').replace('!', '').replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('\'', '').replace('\"', '').replace(';', '').replace('*', "").replace('>','').replace('<', '').replace('|', '').replace('\\', '')
    os.makedirs(f".\instagram_links\{index}_{row['name']}", exist_ok=True)