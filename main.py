import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import pandas_datareader.data as web
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import yfinance as yf
import datetime
from datetime import date, timedelta

st.set_page_config(layout="wide")

count = st_autorefresh(interval=300000, limit=100, key="tickerRefresh")

col1, col2, col3 = st.columns(3)

@st.cache(allow_output_mutation=True)
def load_data():
    bqclient = bigquery.Client()

    # Download the client data file
    # Client Data File
    table = bigquery.TableReference.from_string(
        "bsaw-project.Client.bsaw-client"
    )
    rows = bqclient.list_rows(table)
    client = rows.to_dataframe(create_bqstorage_client=True)

    # Portfolio Data File
    table = bigquery.TableReference.from_string(
        "bsaw-project.Client.bsaw-portfolio"
    )
    rows = bqclient.list_rows(table)
    pfolio = rows.to_dataframe(create_bqstorage_client=True)

    # Download query results.
    query_string = """
    SELECT *
    FROM `bsaw-project.Client.bsaw-client` AS cd
    JOIN `bsaw-project.Client.bsaw-portfolio` AS pf ON cd.Client_ID = pf.Client_ID;
    """

    data = (
        bqclient.query(query_string)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )

    reco = pd.read_csv('gs://bsaw-stock-bucket/data/reco_output.csv')                    # CHANGE THIS
    
    pred_data = pd.read_csv('gs://bsaw-stock-bucket/data/pred_data.csv')                 # CHANGE THIS

    stock = pd.read_csv('gs://bsaw-stock-bucket/data/stock_data.csv')[1:]                # CHANGE THIS

    sentiment_output = pd.read_csv('gs://bsaw-stock-bucket/data/sentiment_output.csv')    # CHANGE THIS

    final_data = pd.read_csv('gs://bsaw-stock-bucket/data/final_data.csv')                # CHANGE THIS

    return client, pfolio, data, reco, pred_data, stock, sentiment_output, final_data

client, pfolio, data, reco, pred_data, stock, sentiment_output, final_data = load_data() 
customers = sorted(set(client['Client_ID']))
risk_profiles = sorted(set(reco['volatile']))
risk_profiles.append(0)

# CHANGE THE IMAGE HERE
image = Image.open('logo.jpeg')

with st.sidebar:
    st.image(image, width=300)
    with st.form(key = 'input-form'):
        customer = st.selectbox(
            'Customer',
            customers
        )

        risk_index = client.Risk_Profile[client['Client_ID'] == customer].values[0]
        RISK_INDEX = risk_profiles.index(risk_index)
        risk = st.selectbox(
            'Profile',
            risk_profiles,
            index = RISK_INDEX,
            help = "1 = Aggressive | 2 = Moderate | 3 = Conservative"
        )

        amount = st.number_input('Amount', min_value = 0)

        month = st.selectbox(
            'Months',
            (1, 3, 6)
        )

        submitted1 = st.form_submit_button('Submit')

if customer in data['Client_ID'].unique():
    col1.subheader(customer +'(Current Pfolio)')
    # new = data[data['Client_ID'].str.contains(customer)]
    new = data[data['Client_ID'] == customer]
    new['Market_Value'] = new['Market_Value'].round()

    fig = px.bar(new, x="Ticker", y="Market_Value", color="Ticker", text_auto=True, width=400, height=400)
    col1.plotly_chart(fig)
    col1.subheader(f"Asset under Management (AUM) : ${new['Market_Value'].sum()}")

def recommendation(df):
    symbol = st.selectbox(
        'Ticker',
        df['Ticker'].unique()
    )

    stockcols = list(pred_data.columns.values)

    if symbol in stockcols:
        if month == 1:
            month_1_stock = stock[-25:]
            month_1_pred = pred_data[:31]

            fig = go.Figure()
            fig1 = go.Scatter(x=month_1_stock['Date'], y=month_1_stock[symbol], name = "Actual")
            # fig2 = go.Scatter(x=month_1_pred["Date"], y=month_1_pred[symbol], name = "Predicted")
            fig.update_layout(title=f"Stock Price History of {symbol}",
                    xaxis_title='Stocks',
                    yaxis_title='Price',
                    width = 1200,
                    height = 600)
            fig.add_trace(fig1)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            # fig.add_trace(fig2)
            st.plotly_chart(fig)

        elif month == 3:
            month_1_stock = stock[-50:]
            month_1_pred = pred_data[:91]

            fig = go.Figure()
            fig1 = go.Scatter(x=month_1_stock['Date'], y=month_1_stock[symbol], name = "Actual")
            # fig2 = go.Scatter(x=month_1_pred["Date"], y=month_1_pred[symbol], name = "Predicted")
            fig.update_layout(title=f"Stock Price History of {symbol}",
                    xaxis_title='Stocks',
                    yaxis_title='Price')
            fig.add_trace(fig1)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            # fig.add_trace(fig2)
            st.plotly_chart(fig)

        elif month == 6:
            month_6_stock = stock
            month_6_pred = pred_data

            fig = go.Figure()
            fig1 = go.Scatter(x=month_6_stock['Date'], y=month_6_stock[symbol], name = "Actual")
            # fig2 = go.Scatter(x=month_6_pred["Date"], y=month_6_pred[symbol], name = "Predicted")
            fig.update_layout(title=f"Stock Price History of {symbol}",
                    xaxis_title='Stocks',
                    yaxis_title='Price')
            fig.add_trace(fig1)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            # fig.add_trace(fig2)
            st.plotly_chart(fig)
    else:
        st.error(f"{symbol} - Data not found !!!")

    buyers = customer_risk_data[customer_risk_data['Recommendation'] == "Buy"]
    count_buy = len(buyers)
    if count_buy == 0:
        per_stock = 0
    else:
        per_stock = round(amount / count_buy, 2)

    #name = client[client['Client_ID'] == customer]['Client_Name'].values[0]
    pfolio_data = data[data['Client_ID'] == customer]
    # st.write(pfolio_data)
    pfolio_data = pfolio_data[['Ticker', 'Units', 'Unit_Price', 'Market_Value']]

    for idx, stock_name in enumerate(buyers['Ticker']):
        unit_price = final_data[final_data['Ticker'] == stock_name]['Day_0'].values[0]
        units = per_stock / unit_price

        if month == 1:
            price = units * final_data[final_data['Ticker'] == stock_name]['Day_30'].values[0]
        elif month == 3:
            price = units * final_data[final_data['Ticker'] == stock_name]['Day_90'].values[0]
        elif month == 6:
            price = units * final_data[final_data['Ticker'] == stock_name]['Day_180'].values[0]

        #name = client[client['Client_ID'] == customer]['Client_Name'].values[0]
        # pfolio_data = data[data['Client_ID'] == customer]
        # # st.write(pfolio_data)
        # pfolio_data = pfolio_data[['Ticker', 'Units', 'Unit_Price', 'Market_Value']]

        new_data = {
            'Ticker': buyers.values[idx][0],
            'Units': round(units, 2),
            'Unit_Price': unit_price,
            'Market_Value': price
        }
        pfolio_data = pfolio_data.append(new_data, ignore_index=True)

    pfolio_data['Market_Value'] = pfolio_data['Market_Value'].round()
    col3.subheader(customer +'(Target Pfolio)')
    fig = px.bar(pfolio_data, x="Ticker", y="Market_Value", color="Ticker", text_auto=True, width=400, height=400)
    col3.plotly_chart(fig)
    col3.subheader(f"Asset under Management (AUM) : ${pfolio_data['Market_Value'].sum()}")
    
if customer in reco['Client_ID'].unique():
    st.markdown("<h1 style='text-align: center;'>Recommendation</h1>", unsafe_allow_html=True)
    customer_data = reco[reco['Client_ID'] == customer]
    
    if risk in customer_data['volatile'].unique():
        customer_risk_data = customer_data[customer_data['volatile'] == risk]

        if month == 1:
            customer_risk_data = customer_risk_data[['Ticker', 'Trend_180', 'Day_30', 'volatile', 'Sentiment', 'Buy/Not']]
            customer_risk_data.rename(columns = {
            "Trend_180": "Trend",
            "Day_30": "Predicted Price",
            "Buy/Not": "Recommendation"
        }, inplace=True)
        elif month == 3:
            customer_risk_data = customer_risk_data[['Ticker', 'Trend_180', 'Day_90', 'volatile', 'Sentiment', 'Buy/Not']]
            customer_risk_data.rename(columns = {
            "Trend_180": "Trend",
            "Day_90": "Predicted Price",
            "Buy/Not": "Recommendation"
        }, inplace=True)
        elif month == 6:
            customer_risk_data = customer_risk_data[['Ticker', 'Trend_180', 'Day_180', 'volatile', 'Sentiment', 'Buy/Not']]
            customer_risk_data.rename(columns = {
            "Trend_180": "Trend",
            "Day_180": "Predicted Price",
            "Buy/Not": "Recommendation"
        }, inplace=True)

        #customer_risk_data = customer_risk_data[['Ticker', 'Trend_180', 'Volatile', 'Sentiment', 'Buy/Not']]
        # customer_risk_data.rename(columns = {
        #     "Trend_180": "Trend",
        #     "Buy/Not": "Recommendation"
        # }, inplace=True)

        # st.write(customer_risk_data)

        # SELL PART
        new_customer_data = pfolio[pfolio['Client_ID'] == customer]
        # new_customer_data = new_customer_data.merge(sentiment_output, on="Ticker")
        new_customer_data = new_customer_data.merge(sentiment_output, on="Ticker", how='left')
        new_customer_data['Sentiment'].fillna(value = 'Neutral', inplace = True)
        new_customer_data['ss_rank'].fillna(value = 2, inplace = True)
        new_customer_data = new_customer_data.merge(final_data, on="Ticker")
        new_customer_data.loc[new_customer_data['Trend_180'] == "Up", "Recommendation"] = "Hold"
        new_customer_data.loc[(new_customer_data['Trend_180'] == "Down") & (new_customer_data['ss_rank'] == 1), "Recommendation"] = "Hold"
        new_customer_data.loc[(new_customer_data['Trend_180'] == "Down") & (new_customer_data['ss_rank'] == 2) | (new_customer_data['ss_rank'] == 3), "Recommendation"] = "Sell"
        
        if month == 1:
            new_customer_data = new_customer_data[['Ticker', 'Trend_180', 'Day_30', 'Volatile', 'Sentiment', 'Recommendation']]
            new_customer_data.rename(columns = {
            "Trend_180": "Trend",
            "Day_30": "Predicted Price",
            "Volatile": "volatile"
        }, inplace=True)
        elif month == 3:
            new_customer_data = new_customer_data[['Ticker', 'Trend_180', 'Day_90', 'Volatile', 'Sentiment', 'Recommendation']]
            new_customer_data.rename(columns = {
            "Trend_180": "Trend",
            "Day_90": "Predicted Price",
            "Volatile": "volatile"
        }, inplace=True)
        elif month == 6:
            new_customer_data = new_customer_data[['Ticker', 'Trend_180', 'Day_180', 'Volatile', 'Sentiment', 'Recommendation']]
            new_customer_data.rename(columns = {
            "Trend_180": "Trend",
            "Day_180": "Predicted Price",
            "Volatile": "volatile"
        }, inplace=True)

        newDF = pd.concat([customer_risk_data,new_customer_data])
        tickers = newDF.Ticker
        df_stock_data = yf.download(tickers.to_list(), datetime.datetime.now(), datetime.datetime.now(), auto_adjust=True)['Close']
        df_stock_data = df_stock_data.reset_index()
        df_stock_data.drop('Date', axis = 1, inplace = True)
        df_stock_data = df_stock_data.T
        df_stock_data = df_stock_data.reset_index().rename(columns={"index":"Ticker", 0:"Market Value"})
        newDF = pd.merge(newDF, df_stock_data, on='Ticker')
        
        #st.table(newDF[['Ticker', 'Trend', 'volatile', 'Market Value','Sentiment', 'Recommendation']])
        hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        #st.table(newDF)
        st.table(newDF[['Ticker', 'Trend', 'Predicted Price', 'volatile','Sentiment','Recommendation']])

        ################################################################################################
        # REAL TIME STOCK PRICE DATA
        ################################################################################################
        st.markdown("""---""")
        st.markdown("<h1 style='text-align: center;'>Real-time Stock Price Data</h1>", unsafe_allow_html=True)
        ncol = len(list(newDF['Ticker']))
        wcol = len(list(newDF['Ticker']))
        cols = st.columns(ncol)

        for i,ticker in enumerate(newDF['Ticker']):
            col = cols[i%wcol]

            today = date.today()

            d1 = today.strftime("%Y/%m/%d")
            end_date1 = d1
            d2 = date.today() - timedelta(days=1)
            d2 = d2.strftime("%Y/%m/%d")
            start_date1 = d2

            d1 = date.today() - timedelta(days=1)
            end_date2 = d1
            d2 = date.today() - timedelta(days=2)
            d2 = d2.strftime("%Y/%m/%d")
            start_date2 = d2

            a = ticker
            data1 = web.DataReader(name=a, data_source='yahoo', start=start_date1, end=end_date1)
            close_today1 = data1["Close"].tolist()[0]

            data2 = web.DataReader(name=a, data_source='yahoo', start=start_date2, end=end_date2)
            close_today2 = data2["Close"].tolist()[0]

            col.metric(label=a, value = f'$ {round(close_today1, 2)}', delta = f'{round(close_today1-close_today2, 2)} %')
        st.markdown("""---""")
        ##############################################################################################

        recommendation(newDF)
    else:
        st.warning(f"There are no stocks recommended !!!")
# else:
#     st.warning(f"{customer} does not have any recommendations !!!")