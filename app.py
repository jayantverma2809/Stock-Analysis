import streamlit as st
from yahoo_fin import stock_info as si
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from functions import *
import datetime

# Page Configuration
st.set_page_config(page_title="Jayant Verma", page_icon=Image.open('statics/Me.jpg'),layout="wide",initial_sidebar_state="expanded")

#Sidebar

# Add profile image
profile_image = Image.open("statics/Me.jpg")
st.sidebar.image(profile_image, use_column_width=True)

# Add contact information
st.sidebar.title("Jayant Verma")
st.sidebar.write("Data Scientist & AI Engineer")
st.sidebar.write("You can reach me at:")
st.sidebar.subheader("jayantverma9380@gmail.com")
st.sidebar.subheader("[LinkedIn](https://www.linkedin.com/in/jayantverma28)")
st.sidebar.subheader("[X](https://x.com/__kanhaiya__)")
st.sidebar.subheader("[Instagram](http://instagram.com/_._.kanhaiya)")
st.sidebar.subheader("[GitHub](https://github.com/jayantverma2809)")
st.sidebar.subheader("[Kaggle](https://www.kaggle.com/jayantverma9380)")

#Skills
st.sidebar.header("Skills")
st.sidebar.write("Here are some of my top skills:")
st.sidebar.write("- Python, SQL")
st.sidebar.write("- Databases : PostgreSQL, MongoDB, Pinecone, Qdrant")
st.sidebar.write("- Libraries & Frameworks : FastAPI, Openai, Llama-index, Langchain, Pinecone, LLM-Sherpa, Openpipe, PyMongo, Transformers, Huggingface, Boto3, Pandas, Numpy, Pandasai, Matplotlib, Seaborn, Scikit-Learn, Streamlit, BeautifulSoup")
st.sidebar.write("- Models : GPT-3.5, GPT-4, Text-da-vinci, Dall.E,Instructor Large, Llama-2, LLama-3, Stable Diffusion, Bart-Base, Claude Instant V1, Claude 3, Phi-3")
st.sidebar.write("- Data Science : Data Collection, Data Wrangling, Data Visualization, Exploratory Data Analysis, Feature Engineering, Feature Selection, Machine Learning(Regression, Classification, Clustering), Model Evaluation, Model Deployment")
st.sidebar.write("- AWS Services : Bedrock, Textract, Cognito")
st.sidebar.write("- Version Control & Deployment : Git & Github")

st.sidebar.title("Experience")
st.sidebar.header("- Data Scientist - Softsensor.ai [June, 2023 - Present]")
st.sidebar.write("Worked on Softsensor X, an AI-based application that allows users to perform QA on their data source, compare and contrast QA between multiple docs, image generation, and offers data visualization capabilities.")
st.sidebar.subheader("Key responsibilities and achievements include:")
st.sidebar.write("- Developed and managed the entire backend of the web app using FastAPI. - Created pipelines for data ingestion, RAG (Retrieval Augmented Generation), RAG Fusion, Hybrid Search, Metadata filtering, Data Visualization, Image Generation, Agent for full automation, Query Transformation, Compare and contrast, etc. - Fine-tuned models such as GPT-3.5, Llama-2 for QA use case and Bart-base for query transformation")
st.sidebar.write("- Created pipelines for data ingestion, RAG (Retrieval Augmented Generation), RAG Fusion, Hybrid Search, Metadata filtering, Data Visualization, Image Generation, Agent for full automation, Query Transformation, Compare and contrast, etc.")
st.sidebar.write("- Fine-tuned models such as GPT-3.5, Llama-2 for QA use case and Bart-base for query transformation")

st.sidebar.title("Open Source Contributions")
st.sidebar.subheader("Llama-index")
st.sidebar.write("An open-source framework for developing applications that use language models. It's a data framework that can connect custom data sources to large language models (LLMs).")
#Projects
st.sidebar.title("Projects")
st.sidebar.write("Here are some of my projects:")
st.sidebar.subheader("- Softsensor X")
st.sidebar.write("Developed and managed the entire backend of the web app using FastAPI.")
st.sidebar.write("Created pipelines for data ingestion, RAG (Retrieval Augmented Generation), RAG Fusion, Hybrid Search, Metadata filtering, Data Visualization, Image Generation, Agent for full automation, Query Transformation, Compare and contrast, etc.")
st.sidebar.write("Fine-tuned models such as GPT-3.5, Llama-2 for QA use case and Bart-base for query transformation.")
st.sidebar.subheader("- [Cyberbullying Tweet Recognition App](https://lnkd.in/d_beCUvK)")
st.sidebar.write("This App predicts the nature of the tweet into 6 categories :  Age, Ethnicity, Gender, Religion, other, Not cybebullying")
st.sidebar.subheader("- [Used Phone Price Prediction](https://usedphonepriceprediction.azurewebsites.net/)")
st.sidebar.write("Using unsupervised learning techniques to predict prices of used phones using their various features such as days used, camera, battery,etc.")
st.sidebar.subheader("- [EDA & Feature Engineering - Bike Sharing Data](https://lnkd.in/dzjAsajs)")
st.sidebar.write("Under this data preprocessing project, I have performed time series analysis, exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, etc to convert raw data into model training ready data.")
st.sidebar.subheader("- [EDA & Feature Engineering - Wine Quality Data](https://lnkd.in/dKRMT7Ym)")
st.sidebar.write("Under this data preprocessing project, I have performed exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, standardization to convert raw data into model training ready data.")


# Main Page Starts Here !

stocks = get_stock_names()
stock_dict = get_stock_ticker_dict()

st.title("Stock Analysis App")
image = Image.open('statics/stocks.jpg')
st.image(image,width=700)

user_input = st.selectbox('__Please select the stock for Fundamental and Technical analysis__',(stocks))

ticker = stock_dict[user_input]
ticker_yf = ticker+".NS"
quote_data = get_quote_table(ticker_yf)
for key in quote_data.keys():
    if quote_data[key] is np.nan:
        quote_data[key] = "n/a"
    else:
        pass
today_date = datetime.datetime.today()
stats_value = get_stats_valuation(ticker_yf)
stats_value.fillna("n/a",inplace=True)
attr = list(stats_value.iloc[:,0])
val = list(stats_value.iloc[:,1])
stats_value_dict = dict(zip(attr,val))
stats = get_stats(ticker_yf)
stats.fillna("n/a",inplace=True)
stats_dict = dict(zip(stats["Attribute"],stats["Value"]))

tab1, tab2 = st.tabs(["__Fundamental Analysis__","__Technical Analysis__"])

with tab1:
    with st.container():
        col1s, col2s = st.columns(2)
        with col1s:
            st.header(user_input)
            st.subheader("₹"+str(quote_data['Quote Price']))
        with col2s:
            st.write("")
            st.subheader("NSE : "+ticker)
        st.write("Volume : "+str(quote_data['Volume']))
        st.write('''***''')
    with st.container():
        st.subheader("Price Summary")
        col1p, col2p, col3p = st.columns(3)
        with col1p:
            #High
            st.metric(label="__High__",value="₹"+str(quote_data["Day's Range"].split("-")[1]),help="Highest price at which a stock traded during the course of the trading day")
            #52-Week High
            st.metric(label="__52-Week High__",value="₹"+str(quote_data["52 Week Range"].split("-")[1]),help="Highest price at which a security has traded during the time period that equates to one year")
        with col2p:
            #Low
            st.metric(label="__Low__",value="₹"+str(quote_data["Day's Range"].split("-")[0]),help="Lowest price at which a stock traded during the course of the trading day")
            #52-Week Low
            st.metric(label="__52-Week Low__",value="₹"+str(quote_data["52 Week Range"].split("-")[0]),help="Lowest price at which a security has traded during the time period that equates to one year")
        with col3p:
            #Open
            st.metric(label="__Open__",value="₹"+str(quote_data['Open']),help="Price at which the financial security opens in the market when trading begins.")
        st.write('''***''')
    with st.container():
        st.subheader("Price")
        tab1w, tab1m, tab3m, tab6m, tab1y, tab5y = st.tabs(["1 Week","1 Month","3 Month","6 Month","1 Year","5 Year"])
        end_date = datetime.datetime.today()
        with tab1w:
            start_date = end_date - datetime.timedelta(days=7)
            trade_data_week = si.get_data(ticker_yf, start_date=start_date,end_date=end_date)
            trade_data_week = trade_data_week.iloc[:,0:5]
            fig_week = px.line(trade_data_week['close'],title=user_input,labels={"index":"","value":"Price"},line_shape="spline",markers=True)
            st.plotly_chart(fig_week,use_container_width=True)
        with tab1m:
            start_date = end_date - datetime.timedelta(days=30)
            trade_data_month = si.get_data(ticker_yf, start_date=start_date,end_date=end_date)
            trade_data_month = trade_data_month.iloc[:,0:5]
            fig_mon = px.line(trade_data_month['close'],title=user_input,labels={"index":"","value":"Price"},line_shape="spline",markers=True)
            st.plotly_chart(fig_mon,use_container_width=True)
        with tab3m:
            start_date = end_date - datetime.timedelta(days=90)
            trade_data_3month = si.get_data(ticker_yf, start_date=start_date,end_date=end_date)
            trade_data_3month = trade_data_3month.iloc[:,0:5]
            fig_3mon = px.line(trade_data_3month['close'],title=user_input,labels={"index":"","value":"Price"},line_shape="spline")
            st.plotly_chart(fig_3mon,use_container_width=True)
        with tab6m:
            start_date = end_date - datetime.timedelta(days=180)
            trade_data_6month = si.get_data(ticker_yf, start_date=start_date,end_date=end_date)
            trade_data_6month = trade_data_6month.iloc[:,0:5]
            fig_6mon = px.line(trade_data_6month['close'],title=user_input,labels={"index":"","value":"Price"},line_shape="spline")
            st.plotly_chart(fig_6mon,use_container_width=True)
        with tab1y:
            start_date = end_date - datetime.timedelta(days=365)
            trade_data_year = si.get_data(ticker_yf, start_date=start_date,end_date=end_date)
            trade_data_year = trade_data_year.iloc[:,0:5]
            fig_1y = px.line(trade_data_year['close'],title=user_input,labels={"index":"","value":"Price"},line_shape="spline")
            st.plotly_chart(fig_1y,use_container_width=True)
        with tab5y:
            start_date = end_date - datetime.timedelta(days=1826)
            trade_data_5year = si.get_data(ticker_yf, start_date=start_date,end_date=end_date)
            trade_data_5year = trade_data_5year.iloc[:,0:5]
            fig_5y = px.line(trade_data_5year['close'],title=user_input,labels={"index":"","value":"Price"})
            st.plotly_chart(fig_5y,use_container_width=True)
        st.write('''***''')
    with st.container():
        st.subheader("Company Info")
        col1c, col2c, col3c = st.columns(3)
        with col1c:
            #Market Capital
            # market_cap = quote_data['Market Cap'] 
            market_cap = quote_data['Market Cap (intraday)'] 
            if market_cap == "n/a":
                st.metric(label="__Market Capital__",value=market_cap,help="Market capitalization is the aggregate valuation of the company based on its current share price and the total number of outstanding shares.")
            else:
                st.metric(label="__Market Capital__",value="₹"+str(market_cap),help="Market capitalization is the aggregate valuation of the company based on its current share price and the total number of outstanding shares.")
            #Total Cash
            total_cash = stats_dict['Total Cash (mrq)']
            if total_cash == "n/a":
                st.metric(label="__Total Cash__",value=total_cash,help="The cash balance at the end of year after paying out dividends and expenses.")
            else:
                st.metric(label="__Total Cash__",value="₹"+str(total_cash),help="The cash balance at the end of year after paying out dividends and expenses.")
            #Total Shares
            total_shares = stats_dict['Shares Outstanding 5']
            st.metric(label="__Total Shares__",value=total_shares,help="It shows the number of shares outstanding in the company.")
            
        with col2c:
            #Enterprise Value
            enterprise_value = stats_value_dict["Enterprise Value"]
            if enterprise_value == "n/a":
                st.metric(label="__Enterprise Value__",value=enterprise_value,help="It measures companys total value, which includes market capitalization, debt and excludes cash.")
            else:
                st.metric(label="__Enterprise Value__",value="₹"+str(enterprise_value),help="It measures companys total value, which includes market capitalization, debt and excludes cash.")
            #Total Debt
            total_debt = stats_dict['Total Debt (mrq)']
            if total_debt == "n/a":
                st.metric(label="__Total Debt (mrq)__",value=total_debt,help="It is the sum of all short term and long term debts taken by the company.")
            else:
                st.metric(label="__Total Debt (mrq)__",value="₹"+str(total_debt),help="It is the sum of all short term and long term debts taken by the company.")
        with col3c:
            #Revenue
            total_revenue = stats_dict["Revenue (ttm)"]
            if total_revenue == "n/a":
                st.metric(label="__Revenue (ttm)__",value=total_revenue,help="It is companys core revenue net of discounts and returns.")
            else:
                st.metric(label="__Revenue (ttm)__",value="₹"+str(total_revenue),help="It is companys core revenue net of discounts and returns.")
            #EBITDA
            ebitda = stats_dict['EBITDA']
            if ebitda == "n/a":
                st.metric(label="__EBITDA__",value=ebitda,help="Earnings Before Interest, Taxes, Depreciation, and Amortisation, or EBITDA, is a statistic used to assess a company's operating performance")
            else:
                st.metric(label="__EBITDA__",value="₹"+str(stats_dict['EBITDA']),help="Earnings Before Interest, Taxes, Depreciation, and Amortisation, or EBITDA, is a statistic used to assess a company's operating performance")
        st.write('''***''')
    with st.container():
        st.subheader("Ratio Analysis")
        col1r, col2r,col3r,col4r = st.columns(4)
        with col1r:
            #P.E. Ratio
            st.metric(label = "__P.E. Ratio__",value=quote_data['PE Ratio (TTM)'],help="It is a valuation parameter that measures the company's current share price relative to its per-share earnings. Generally, high P/E is Overvalued & low P/E is Undervalued.")
            #Total Cash Per Share
            st.metric(label="__Total Cash/Share (mrq)__",value=stats_dict['Total Cash Per Share (mrq)'],help="Cash per share is the broadest measure of available cash to a business divided by the number of equity shares outstanding. Cash per share tells us the percentage of a company's share price available to spend on strengthening the business, paying down debt, returning money to shareholders, and other positive campaigns.")
            #Return On Assets
            st.metric(label="__ROA (ttm)__",value=stats_dict['Return on Assets (ttm)'],help="It indicates how profitable a company relative to its total assets & also explains how efficiently company using its assets to generate earnings.")
        with col2r:
            #E.P.S.
            st.metric(label="__E.P.S.(TTM)__",value=quote_data['EPS (TTM)'],help="It is the net profit allocated to each outstanding share of common stock.(latest year)")
            #Revenue Per Share
            st.metric(label="__Revenue/Share (ttm)__",value=stats_dict['Revenue Per Share (ttm)'],help="Total revenue earned per share over a designated period, whether quarterly, semi-annually, annually, or trailing twelve months (TTM).")
            #PEG Ratio(5Yrs Expected)
            st.metric(label="__PEG Ratio (5yr-exp.)__",value=stats_value_dict['PEG Ratio (5yr expected)'],help="P/E ratio divided by the growth rate of its earnings for a specified time period. PEG ratio is used to determine a stock's value while also factoring in the company's expected earnings growth")
        with col3r:
            #P.B.Ratio
            st.metric(label="__P.B. Ratio (mrq)__",value=stats_value_dict['Price/Book'],help="It shows the relationship between the current price and the book value of each share. A lower P/B ratio can mean that the stock is undervalued.")
            #Payout Ratio
            st.metric(label="__Payout Ratio__",value=stats_dict['Payout Ratio 4'],help="The payout ratio is a financial metric showing the proportion of earnings a company pays its shareholders in the form of dividends")
            #Price to Sales Ratio
            st.metric(label="__Price/Sales Ratio__",value=stats_value_dict['Price/Sales'],help="It indicates how much investor paid for a share compared to the sales a company.")
        with col4r:
            #Total Debt to Equity Ratio (mrq)
            st.metric(label="__Total Debt/Equity Ratio (mrq)__",value=stats_dict['Total Debt/Equity (mrq)'],help="Total Debt for the most recent interim period divided by Total Shareholder Equity for the same period. Measure of degree to which a company is financing through debt vs owned funds, ideal ratio should be less than 1.")
            #Return on Equity (ttm)
            st.metric(label="__ROE (ttm)__",value=stats_dict['Return on Equity (ttm)'],help="It measures the ability of a firm to generate profits from its shareholders investments in the company. A higher ROE Indicates better efficiency.")
            #Book Value Per Share
            st.metric(label="__Book Value/Share__",value=stats_dict['Book Value Per Share (mrq)'],help="Book value per share (BVPS) is the ratio of equity available to common shareholders divided by the number of outstanding shares.")
        st.write('''***''')
    with st.container():
        st.subheader("Financials")
        st.write("__Balance Sheet (All figures in crores)__")
        st.dataframe(get_balance_sheet(ticker),use_container_width=True,)

        st.write("__Profit & Loss Statement (All figures in crores)__")
        st.dataframe(get_profit_and_loss(ticker),use_container_width=True)

        if "Bank" not in user_input:
            st.write("__Cash Flow Statement (All figures in crores)__")
            st.dataframe(get_cashflow(ticker),use_container_width=True)

        st.write("__Quarterly Results (All figures in crores)__")
        st.dataframe(get_quarterly_results(ticker),use_container_width=True)

        st.write("__Promoter Details__")
        if "Bank" not in user_input:
            st.dataframe(get_promoter_details(ticker),use_container_width=True)
        else:
            st.dataframe(get_promoter_details_bank(ticker),use_container_width=True)

        st.write("__Investor Details__")
        if "Bank" not in user_input:
            st.dataframe(get_investor_details(ticker),use_container_width=True)
        else:
            st.dataframe(get_investor_details_bank(ticker),use_container_width=True)
        
    st.write('''***''')

with tab2:
    with st.container():
        col1s, col2s = st.columns(2)
        with col1s:
            st.header(user_input)
            st.subheader("₹"+str(quote_data['Quote Price']))
        with col2s:
            st.write("")
            st.subheader("NSE : "+ticker)
        st.write("Volume : "+str(quote_data['Volume']))
        st.write('''***''')
    end_date = datetime.datetime.today()
    start_date_t = end_date - datetime.timedelta(days=365)
    trade_data = si.get_data(ticker_yf, start_date=start_date_t,end_date=end_date)
    trade_data = trade_data.iloc[:,0:6]
    with st.container():
        st.subheader("Volume")
        fig_vol = plot_volume(trade_data,ticker)
        st.plotly_chart(fig_vol,use_container_width=True)
        st.write('''***''')
    with st.container():
        st.subheader("Moving Average")
        tabsma, tabewma = st.tabs(["__Simple Moving Average__","__Exponentially-weighted Moving Average__"])
        with tabsma:
            ndays_sma = st.radio("Select days for SMA",(5,10,20),help="Simple Moving Average",horizontal=True)
            fig_sma = calculate_and_plot__sma(trade_data,ndays_sma,ticker)
            st.plotly_chart(fig_sma,use_container_width=True)
        with tabewma:
            ndays_ewma = st.radio("Select days for EWMA",(50,100,200),help="Exponentially-weighted Moving Average",horizontal=True)
            fig_ewma = calculate_and_plot__ewma(trade_data,ndays_ewma,ticker)
            st.plotly_chart(fig_ewma,use_container_width=True)
        st.write('''***''')
    with st.container():
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = calculate_and_plot_rsi(trade_data,ticker)
        st.plotly_chart(fig_rsi,use_container_width=True)
        st.write('''***''')
    with st.container():
        st.subheader("Bollinger Bands")
        ndays_bb = st.radio("Select days for Moving Average",(20,30,50),help="Moving average to be used to calculate bollinger bands",horizontal=True)
        fig_bollinger = calculate_and_plot_bollinger(trade_data,ndays_bb,ticker)
        st.plotly_chart(fig_bollinger,use_container_width=True)
        st.write('''***''')