# Content Scrapping

# 42 Industries
industries = [
    'Aerospace & Defense', 'Auto Components', 'Automobiles',
    'Banks', 'Building Products',
    'Chemicals', 'Commercial Services', 'Construction & Engineering', 'Construction Materials', 'Consumer Durables', 'Consumer Services', 'Containers & Packaging',
    'Diversified Financials', 'Diversified Metals',
    'Electrical Equipment', 'Energy Services',
    'Food Products', 'Food Retailers',
    'Healthcare', 'Homebuilders', 'Household Products',
    'Industrial Conglomerates', 'Insurance',
    'Machinery', 'Media',
    'Oil & Gas Producers',
    'Paper & Forestry', 'Pharmaceuticals', 'Precious Metals',
    'Real Estate', 'Refiners & Pipelines', 'Retailing',
    'Semiconductors', 'Software & Services', 'Steel',
    'Technology Hardware', 'Telecommunication Services', 'Textiles & Apparel', 'Traders & Distributors', 'Transportation', 'Transportation Infrastructure',
    'Utilities'
]
selected_stock_exchange = ['NYS', 'NAS', 'TKS', 'LON',  'ETR']

ESG_WEIGHT = 0.25
MARKET_CAPITAL_WEIGHT = 0.75

#
selected_ratings = [0, 1]  # Only Negligible ESG Risk & Low ESG Risk are included
time_span_years = [1, 5, 10, 25]
max_span_years = 25

#
ALL_YEARS_NR = max_span_years
TRAIN_MONTH_NR = 12 * 20
TEST_YEARS_NR = 5 # 3
TEST_MONTHS_NR = TEST_YEARS_NR * 12
PORTF_ALLOC_YEARS_NR = 5
FIVE_YEARS_NR = 5
FIFTEEN_YEARS_NR = 15
MIN_AVG_RETURN = 0.05
