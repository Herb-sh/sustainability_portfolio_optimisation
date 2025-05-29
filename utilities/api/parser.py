#%%
import requests
from bs4 import BeautifulSoup
import json
import csv

"""
Makes an API call and returns the HTML content.

Parameters:
url (str): The URL of the API endpoint.

Returns:
str: HTML content returned by the API.
"""
def get_html_from_api(url, form_data):
    response = requests.post(url, data=form_data)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    # Check if the request was successful
    if response.status_code == 200:
        print(f"Request was successful for {form_data['industry']} and {'Negligible ESG Risk' if form_data['rating'] == 0 else 'Low ESG Risk'}!")
        if form_data['rating'] == 1:
            print("----------------------------------------------------------------------")
        return response.content
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)
        return ""


"""
Parses HTML content and converts it into a dictionary.

Parameters:
html (str): HTML content to parse.

Returns:
dict: Parsed data.
"""
def parse_html_to_dict(html, industry: str):
    soup = BeautifulSoup(html, 'html.parser')
    html_list = soup.find_all('div', class_="company-row")
    dict_list = []
    for item in html_list:
        data = {}
        data['company_name'] = item.find('a').text
        data['ticker_symbol'] = item.find('small').text
        data['company_esg_score'] = item.find(class_="col-2").text
        data['company_esg_score_group'] = item.find(class_="col-lg-6 col-md-10").text
        data['industry'] = industry
        dict_list.append(data)

    return dict_list

def create_csv(data, filename = 'output.csv'):
    fileDir = '../data/'
    fileFormat = '.csv'


    if fileFormat not in filename:
        filename += fileFormat

    filePath = fileDir + filename

    # Get the header from the keys of the first dictionary
    header = data[0].keys()
    # Write to a CSV file
    with open(filePath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()  # Write the header
        writer.writerows(data)  # Write the data
    return ''

"""
Converts a dictionary to a JSON string.

Parameters:
data_dict (dict): Dictionary to convert.

Returns:
str: JSON string.
"""
def convert_dict_to_json(data_dict):
    return json.dumps(data_dict, indent=2)


#%%
