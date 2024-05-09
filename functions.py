import pandas as pd
import plotly.express as px
from io import StringIO
import requests, random
from bs4 import BeautifulSoup

stocks_ticker = pd.read_csv("stocks_ticker.csv")
base_url = "https://ticker.finology.in/company/"
user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    ]

#Creating list of Stocks listed in NSE
def get_stock_names():
    stocks = list(stocks_ticker['Company Name'])
    return stocks

#Creating dictionary of stocks and their tickers for fetching data from yahoo finance
def get_stock_ticker_dict():
    dict_stocks = dict(zip(stocks_ticker['Company Name'],stocks_ticker.Symbol))
    return dict_stocks

#Fundamental Analysis

#Getting Financial Tables
def get_tables(ticker):
    # tables = pd.read_html("https://ticker.finology.in/company/"+ticker)
    tables = pd.read_html(StringIO(requests.get(base_url + ticker).text))
    return tables

#Separating Balance Sheet
def get_balance_sheet(ticker):
    balance_sheet = get_tables(ticker)[3]
    balance_sheet.reset_index(drop=True,inplace=True)
    balance_sheet.fillna("-",inplace=True)
    return balance_sheet

#Separating Profit and Loss Statement
def get_profit_and_loss(ticker):
    p_and_l = get_tables(ticker)[2]
    return p_and_l

#Separating Cash Flow Statement
def get_cashflow(ticker):
    cash_flow = get_tables(ticker)[4]
    cash_flow = cash_flow.iloc[:,0:6]
    return cash_flow

#Separating Quarterly Results
def get_quarterly_results(ticker):
    quarter_results = get_tables(ticker)[1]
    return quarter_results

#Separating Promoter Details
def get_promoter_details(ticker):
    promoters = get_tables(ticker)[5]
    return promoters

#Separating Promoter Details in case of Banking Stocks
def get_promoter_details_bank(ticker):
    promoters = get_tables(ticker)[4]
    return promoters

#Separating Investor Details
def get_investor_details(ticker):
    investors = get_tables(ticker)[6]
    return investors

#Separating Investor Details in case of Banking stocks
def get_investor_details_bank(ticker):
    investors = get_tables(ticker)[5]
    return investors

#Separating Pledging Details
def get_promoter_pledging(ticker):
    pledging = get_tables(ticker)[0]
    return pledging

#Technical Analysis

#Simple Moving Average
def SMA(data, ndays): 
    SMA = pd.Series(data['close'].rolling(ndays).mean(), name = 'SMA') 
    data = data.join(SMA) 
    return data

#Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA = pd.Series(data['close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA) 
    return data

#Calculating Simple Moving Average for given duration and plotting it
def calculate_and_plot__sma(data,ndays,ticker):
    sma = SMA(data,ndays)
    sma = sma.dropna()
    sma = sma['SMA']
    temp_df = pd.DataFrame({"Price":data["close"],"SMA":sma},index=data.index)
    fig = px.line(temp_df,x=temp_df.index,y=temp_df.columns,title=ticker, labels={'value': 'Price', 'index': 'Date'},line_shape="spline")
    return fig

#Calculating Exponentially-weighted Moving Average for given duration and plotting it
def calculate_and_plot__ewma(data,ndays,ticker):
    ewma = EWMA(data,ndays)
    ewma = ewma.dropna()
    ewma = ewma['EWMA_'+str(ndays)]
    temp_df = pd.DataFrame({"Price":data["close"],"EWMA":ewma},index=data.index)
    fig = px.line(temp_df,x=temp_df.index,y=temp_df.columns,title=ticker, labels={'value': 'Price', 'index': 'Date'},line_shape="spline")
    return fig

#Relative-Strength Index
def rsi(close, periods = 14):
    close_delta = close.diff()
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

#Calculating RSI and plotting it
def calculate_and_plot_rsi(data,ticker):
    data['RSI'] = rsi(data['close'])
    temp_df = pd.DataFrame({"Price":data['close'],"RSI":data['RSI']},index=data.index)
    fig = px.line(temp_df, x=temp_df.index, y=temp_df.RSI, title=ticker, labels={'value': 'Price', 'index': 'Date'},line_shape="spline")
    return fig

#Bollinger Bands
def bollinger_bands(data,n):
    MA = data.close.rolling(window=n).mean()
    SD = data.close.rolling(window=n).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data

#Calculating Bollinger Bands and plotting it
def calculate_and_plot_bollinger(data,ndays,ticker):
    data = bollinger_bands(data,ndays)
    temp_df = pd.DataFrame({"Price":data['close'],"Upper Band":data['UpperBand'],"Middle Band":data["MiddleBand"],"Lower Band":data['LowerBand']},index=data.index)
    fig = px.line(temp_df,x=temp_df.index,y=temp_df.columns,title=ticker,labels={'value': 'Price', 'index': 'Date'},color_discrete_map={"Upper Band": "black", "Middle Band": "red", "Lower Band": "black","Price":"blue"})
    return fig
    
#Volume
def plot_volume(data,ticker):
    fig = px.bar(data,x=data.index,y=data['volume']/1000000,title=ticker,labels={'y':'Million','index':'Date'})
    return fig

# def get_quote_table(ticker , dict_result = True, headers = {'User-agent': 'Mozilla/5.0'}): 
    
#     '''Scrapes data elements found on Yahoo Finance's quote page 
#        of input ticker
    
#        @param: ticker
#        @param: dict_result = True
#     '''

#     site = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker
    
#     tables = pd.read_html(StringIO(requests.get(site, headers=headers).text))

#     data = pd.concat([tables[0], tables[1]], axis=0)

#     data.columns = ["attribute" , "value"]
    
#     quote_price = pd.DataFrame(["Quote Price", get_live_price(ticker)]).transpose()
#     quote_price.columns = data.columns.copy()
    
#     data = pd.concat([data, quote_price], axis=0)
    
#     data = data.sort_values("attribute")
    
#     data = data.drop_duplicates().reset_index(drop = True)
    
#     data["value"] = data.value.map(force_float)

#     if dict_result:
        
#         result = {key : val for key,val in zip(data.attribute , data.value)}
#         return result
        
#     return data 

def get_quote_table(ticker): 
    site = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker
    print(site)
    response = requests.get(site, headers={'User-agent': random.choice(user_agents)})
    soup = BeautifulSoup(response.content, 'html.parser')
    values_dict = {}
    values_dict["Quote Price"] = soup.find('fin-streamer', class_='livePrice')['data-value']
    data = soup.find_all("li","svelte-tx3nkj")
    for i in data:
        values_dict[i.find("span","label svelte-tx3nkj").text] = i.find("span","value svelte-tx3nkj").text.strip()
    return values_dict

def get_stats(ticker, headers = {'User-agent': random.choice(user_agents)}):
    
    '''Scrapes information from the statistics tab on Yahoo Finance 
       for an input ticker 
    
       @param: ticker
    '''

    stats_site = "https://finance.yahoo.com/quote/" + ticker + \
                 "/key-statistics?p=" + ticker
    

    tables = pd.read_html(StringIO(requests.get(stats_site, headers=headers).text))
    
    tables = [table for table in tables[1:] if table.shape[1] == 2]
    table = pd.concat(tables, ignore_index=True)
    # table = tables[0]
    # for elt in tables[1:]:
    #     table = table.append(elt)

    table.columns = ["Attribute" , "Value"]
    
    table = table.reset_index(drop = True)
    
    return table

def get_stats_valuation(ticker, headers = {'User-agent': random.choice(user_agents)}):
    
    '''Scrapes Valuation Measures table from the statistics tab on Yahoo Finance 
       for an input ticker 
    
       @param: ticker
    '''

    stats_site = "https://finance.yahoo.com/quote/" + ticker + \
                 "/key-statistics?p=" + ticker
    
    
    tables = pd.read_html(StringIO(requests.get(stats_site, headers=headers).text))
    
    tables = [table for table in tables if "Trailing P/E" in table.iloc[:,0].tolist()]
    
    
    table = tables[0].reset_index(drop = True)
    
    return table