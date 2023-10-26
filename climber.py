import json 
import instaloader
import requests
import re
import os
from requests.exceptions import MissingSchema
from instaloader.exceptions import BadResponseException
import shutil
import time
import pandas as pd


# Initialize Instaloader and login
L = instaloader.Instaloader()

import sys
import urllib.request

# Initialize proxy if Python 3
if sys.version_info[0] == 3:
    proxy_support = urllib.request.ProxyHandler({
        'http': 'http://brd-customer-hl_5e5f6fba-zone-residential:2alnjvvbx31p@brd.superproxy.io:22225',
        'https': 'http://brd-customer-hl_5e5f6fba-zone-residential:2alnjvvbx31p@brd.superproxy.io:22225'
    })
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)
proxies = {
        'http': 'http://brd-customer-hl_5e5f6fba-zone-residential:2alnjvvbx31p@brd.superproxy.io:22225',
        'https': 'http://brd-customer-hl_5e5f6fba-zone-residential:2alnjvvbx31p@brd.superproxy.io:22225'
    }

# Initialize Instaloader and login
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/601.7.8 (KHTML, like Gecko)'

L = instaloader.Instaloader(user_agent=user_agent)
# username = "XXX"
# password = "XXX"
# L.context.login(username, password)

# Assuming you have imported pandas for this
df = pd.read_csv('./grouped_instagram_links.csv')
print(df.head())



for index, row in df.iloc[14:].iterrows():
    input_string = row['links']
    climb_name = row['name']
    id = index
    # This pattern is updated to include parentheses around the shortcode part, which will make it a capturing group
    pattern = r'https://www\.instagram\.com/p/([\w-]+)/'

    # Find all matches in the input string and extract the shortcode
    matches = re.findall(pattern, input_string)
    print(matches)

    # Print the shortcode

    for counter, shortcode in enumerate(matches):
        # Create a new directory if it doesn't exist

        output_directory = f'./vids/{id}_{climb_name}/{counter}'
        try:
            post = instaloader.Post.from_shortcode(L.context, shortcode)  # replace 'shortcode_here'
        except BadResponseException:
            continue

        video_url = post.video_url

        # Stream video through proxy
        try:
            r = requests.get(video_url, stream=True, proxies=proxies, verify='/usr/local/share/ca-certificates/ca.crt')
        except MissingSchema:
            print('Invalid URL')
            continue
        if r.status_code == 200:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)  # This will create the directory if it doesn't exist
            with open(f'{output_directory}/{shortcode}.mp4', 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded {shortcode}.mp4 to {output_directory}")
    