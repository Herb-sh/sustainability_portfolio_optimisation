import importlib
import utilities.variables as variables
import utilities.parser as parser
importlib.reload(parser)

def generate_esg_files():
    try:
        for i, industry in enumerate(variables.industries):
            if len(industry) != 0:
                generate_esg_file_from_industry(industry)
    except RuntimeError:
        print('A runtime error occurred while generating csv files')


def generate_esg_file_from_industry(industry: str):
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
    #
    filename = industry.replace(" ", "").replace("&", "").lower()
    #
    fileDir = '../data/'
    fileFormat = '.csv'
    parser.create_csv(parsed_data_merged, filename=fileDir + filename + fileFormat)
    print('Filename {} was created for industry "{}" '.format(filename + '.csv', industry))

    return parsed_data_merged

def get_payload(industry: str, rating=0, pagesize=100):
    return {
        'resourcePackage': 'Sustainalytics',
        'industry': industry,
        'rating': rating,
        'filter': '',
        'page': 1,
        'pageSize': pagesize
    }