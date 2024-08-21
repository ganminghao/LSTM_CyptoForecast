import json

import requests
import pytz
import pandas as pd
from datetime import datetime, timedelta, timezone
import okx.MarketData as MarketData
import time

get_days = 301
interval = '15m'

flag = "0"  # Production trading:0, demo trading:1
marketDataAPI = MarketData.MarketAPI(flag=flag)

def get_ohlcv(symbol, interval, after):
    # Retrieve the candlestick charts of the index
    result = marketDataAPI.get_history_candlesticks(
        instId=symbol,
        after=after,
        bar=interval,
        limit=96  # 一天中有 96 个 15min, 方便后续数据拼接
    )
    return result.get('data', [])

def to_timestamp(dt):
    return int(dt.timestamp() * 1000)

def data_empty_check(data):
    if not data:
        print("未能获取到任何数据，请检查API请求和时间戳是否正确。")
    else:
        print(f"获取到 {len(data)} 条数据")

if __name__ == "__main__":
    symbol = 'BTC-USDT'

    # 结束时间和开始时间
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=get_days)  # 获取n天的 k info

    # 初始化空列表用于存储数据
    all_data = []
    current_time = end_time

    # 循环请求数据直到到达开始时间
    while current_time > start_time:
        # 计算当前批次的开始时间
        print(f'current_time:{current_time}')
        batch_start_time = max(start_time, current_time - timedelta(days=1))
        current_timestamp = to_timestamp(current_time)
        start_timestamp = to_timestamp(batch_start_time)
        print(f'current_timestamp:{current_timestamp}')

        # 获取数据
        data = get_ohlcv(symbol, interval, current_timestamp)
        data_empty_check(data)

        if data:
            all_data.extend(data)

        # 更新当前时间
        current_time = batch_start_time

        # Rate Limit，每 2 秒最多 10 个请求
        time.sleep(0.2)  # 暂停 2 秒

    # 处理数据
    ohlcv = []
    for item in all_data:
        ts, o, h, l, c, con = item[:6]  # 解包操作

        # 转换时间
        utc_time = datetime.fromtimestamp(int(ts) / 1000, timezone.utc)
        beijing_time = utc_time.astimezone()  # 转换为北京时间

        ohlcv.append([beijing_time, float(o), float(h), float(l), float(c), float(con)])

    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    df.set_index("Timestamp", inplace=True)

    #df = df.drop(df.index[96])

    # 确保数据按照时间升序排序
    df.sort_index(inplace=True)

    # 计算 SMA 5、10、20
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    df['SMA10_5'] = df['SMA_10'] - df['SMA_5']
    df['SMA20_5'] = df['SMA_20'] - df['SMA_5']
    df['SMA20_10'] = df['SMA_20'] - df['SMA_10']

    df['High-Low'] = df['High'] - df['Low']
    df['Close-Open'] = df['Close'] - df['Open']
    df = df[df['Close-Open'] != 0]
    df['Uppershadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lowershadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Percentage'] = (df['Close'] - df['Open']) / df['Open']
    df['Range-to-Change'] = df['Close-Open'] / df['High-Low']
    df['Volume_change'] = df['Volume'].pct_change()

    # 计算 lables
    df[f'return_{interval}'] = df['Close'].pct_change(1)  # 15 分钟的 return

    # 删除前二十行

    df = df.iloc[20:]
    df = df[df['Volume'] != 0]
    df = df[df['Volume_change'] != 0]

    # 检查 DataFrame 是否为空
    if df.empty:
        print("数据帧为空，未能处理任何数据。")
    else:
        print("数据处理完成，保存到文件中。")
        print(df)

    df = df.reset_index()

    # 保存到文件
    file_path = f'/Users/mac/Desktop/QuantData/{symbol}/past{get_days}days_{interval}.csv'
    df.to_csv(file_path)
    print(f"数据已保存到 {file_path}")

