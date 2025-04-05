import time

from config import symbols
from execution.trade_manager import ConvertToRFFormat
import tvDatafeed
import ta


def construct_trades():
    trades = []
    for symbol in symbols:
        n = 10000
        print(f"\nConstructing {n} trades for {symbol['symbol']}")
        data = fetch_trades(symbol, bars=n+30)
        for i in range(30, len(data)-6):
            start_price = data[i][-1]
            inputs = []
            for j in range(i-30, i):
                data_point = data[j][:9]
                inputs.append(data_point)
            inputs = ConvertToRFFormat(inputs)
            results = {}
            for j in range(1, 5):
                index = i+j
                duration_price = data[index][-1]
                change = (duration_price-start_price)/start_price
                results[str(j+1)] = change
            trade = {
                'symbol': symbol,
                'input': inputs,
                'results': results
            }
            trades.append(trade)
    return trades


def fetch_trades(symbol_data, bars=1, get_max=True):
    # Buffer for indicator calculation (ensure enough data)
    calc_buffer = 50  # Fetch at least 30 extra candles
    fetch_count = bars + calc_buffer

    tv = tvDatafeed.TvDatafeed()
    symbol = symbol_data['symbol']
    exchange = symbol_data['exchange']

    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=tvDatafeed.Interval.in_1_minute, n_bars=fetch_count)

    if df is None or len(df) < fetch_count:
        print(f"Not enough data: {"None" if df is None else len(df)}")
        time.sleep(1)
        return fetch_trades(symbol_data, bars=len(df) - calc_buffer if get_max and df is not None else bars)

    df = df.rename(columns={
        'volume': 'Volume',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    })

    # Calculate indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd_obj = ta.trend.MACD(df['Close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    df['ema7'] = ta.trend.EMAIndicator(df['Close'], window=7).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    df['sma7'] = ta.trend.SMAIndicator(df['Close'], window=7).sma_indicator()
    df['sma21'] = ta.trend.SMAIndicator(df['Close'], window=21).sma_indicator()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

    # Normalize volume
    df['normalized_volume'] = df['Volume'] / (df['Volume'] + 1e-6)
    # Drop NaNs from indicator calculation
    df = df.dropna()

    # Extract the last `bars` rows only
    df = df.tail(bars)

    if len(df) < bars:
        print("Not enough valid rows after indicators")
        time.sleep(1)
        return fetch_trades(symbol_data, bars=bars)

    # Extract indicators per row
    results = []
    for _, row in df.iterrows():
        inputs = [
            row['rsi'],
            row['macd'],
            row['macd_signal'],
            row['ema7'],
            row['ema21'],
            row['sma7'],
            row['sma21'],
            row['williams_r'],
            row['normalized_volume'],
            row['Close']
        ]
        results.append(inputs)
    return results[0] if bars == 1 else results
