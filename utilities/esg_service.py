import importlib
import utilities.variables as variables
import utilities.parser as parser
importlib.reload(parser)

'''
Create only one file containing all industries data_esg_raw.csv
'''
def generate_esg_file():
    data = []
    try:
        for i, industry in enumerate(variables.industries):
            if len(industry) != 0:
                industry_data = get_esg_data_by_industry(industry)
                data = data + industry_data
        if len(data) > 0:
            parser.create_csv(data, filename='data_esg_raw.csv')
    except RuntimeError:
        print('A runtime error occurred while generating csv files')


'''
Create a file for each industry (automobile.csv, banks.csv)
'''
def generate_esg_file_for_each_industry():
    try:
        for i, industry in enumerate(variables.industries):
            if len(industry) != 0:
                data = get_esg_data_by_industry(industry)
                filename = industry.replace(" ", "").replace("&", "").lower() + ".csv"
                #
                parser.create_csv(data, filename)
                print('Filename {} was created for industry "{}" '.format(filename + '.csv', industry))
    except RuntimeError:
        print('A runtime error occurred while generating csv files')

'''
Get ESG(Negligible Risk, Low Risk) rated companies of a given industry.
'''
def get_esg_data_by_industry(industry: str):
    url = 'https://www.sustainalytics.com/sustapi/companyratings/getcompanyratings'  # Replace with your actual URL

    # Negligible ESG risk companies
    payload = get_payload(industry, rating=0)
    html_content = parser.get_html_from_api(url, payload)
    parsed_data = parser.parse_html_to_dict(html_content, industry=industry)

    # Low ESG risk companies
    payload_r1 = get_payload(industry, rating=1)
    html_content_r1 = parser.get_html_from_api(url, payload_r1)
    parsed_data_r1 = parser.parse_html_to_dict(html_content_r1, industry=industry)

    parsed_data_merged = parsed_data + parsed_data_r1

    return parsed_data_merged

'''
Generate payload for Sustainalytics used in get_esg_data_by_industry
'''
def get_payload(industry: str, rating=0, pagesize=100):
    return {
        'resourcePackage': 'Sustainalytics',
        'industry': industry,
        'rating': rating,
        'filter': '',
        'page': 1,
        'pageSize': pagesize
    }