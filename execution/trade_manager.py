import numpy as np
from tradingview_ta import TA_Handler, Interval, get_multiple_analysis
import time
from datetime import datetime, UTC, timedelta, timezone
import math
from PIL import ImageGrab
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from tensorflow.keras.optimizers import Adam
import pyautogui
from config import symbols, settings
from ml_pipeline.save_and_load import load_brains
from data_manager.trade_storage import LoadTrades


def fetch_historical_values(symbol):
    try:
        handler = TA_Handler(
            symbol=symbol['symbol'],
            screener=symbol['screener'],
            exchange=symbol['exchange'],
            interval=Interval.INTERVAL_1_MINUTE
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators

        open_price = analysis.indicators.get('open', 0)
        high_price = analysis.indicators.get('high', 0)
        low_price = analysis.indicators.get('low', 0)
        close_price = analysis.indicators.get('close', 0)
        volume = analysis.indicators.get('volume', 0)

        price_data = [open_price, high_price, low_price, close_price]
        price_min = min(price_data)
        price_max = max(price_data)
        normalized_prices = [(x - price_min) / (price_max - price_min + 1e-6) for x in price_data]

        normalized_volume = volume / (volume + 1e-6)

        sma7 = indicators.get('SMA7') or indicators.get('SMA5')
        sma21 = indicators.get('SMA20') or indicators.get('SMA30')
        ema7 = indicators.get('EMA7') or indicators.get('EMA10')
        ema21 = indicators.get('EMA20') or indicators.get('EMA30')
        macd = indicators.get('MACD.macd', 0)
        macd_signal = indicators.get('MACD.signal', 0)
        williams_r = indicators.get('W.R', 0)
        stoch_k = indicators.get('Stoch.K', 0)
        stoch_d = indicators.get('Stoch.D', 0)
        rsi = indicators.get('RSI', 0)

        params = {
            "OPEN": normalized_prices[0],
            "HIGH": normalized_prices[1],
            "LOW": normalized_prices[2],
            "CLOSE": normalized_prices[3],
            "VOLUME": normalized_volume,
            "SMA7": sma7,
            "SMA21": sma21,
            "EMA7": ema7,
            "EMA21": ema21,
            "MACD": macd,
            "MACD_SIGNAL": macd_signal,
            "WILLIAMS_R": williams_r,
            "STOCH_K": stoch_k,
            "STOCH_D": stoch_d,
            "RSI": rsi,
        }
        return params

    except Exception as e:
        print(f"Error fetching indicators: {e}")
        return {}


def perform_action(action, coords=None, text=None, delay=0.5):
    """
    Generic function for automated UI actions via pyautogui.
    """
    try:
        if action == "move":
            if coords:
                pyautogui.moveTo(coords[0], coords[1], duration=0.2)
            else:
                raise ValueError("Coordinates are required for 'move' action.")
        elif action == "click":
            if coords:
                pyautogui.click(coords[0], coords[1])
            else:
                raise ValueError("Coordinates are required for 'click' action.")
        elif action == "doubleclick":
            if coords:
                pyautogui.click(coords[0], coords[1])
                pyautogui.click(coords[0], coords[1])
            else:
                raise ValueError("Coordinates are required for 'click' action.")
        elif action == "type":
            if text:
                pyautogui.write(text, interval=0.05)
            else:
                raise ValueError("Text is required for 'type' action.")
        else:
            raise ValueError(f"Unknown action: {action}")
        time.sleep(delay)
    except Exception as e:
        print(f"Error performing action: {e}")


def handle_ticker_logic(first_box, second_box, symbol_area):
    """
    Example function that checks the color of squares on screen for some logic.
    """

    def is_square_same_color(top_left, bottom_right):
        try:
            screenshot = ImageGrab.grab()
            region = screenshot.crop((*top_left, *bottom_right))
            pixels = list(region.getdata())
            first_pixel = pixels[0]
            return all(pixel == first_pixel for pixel in pixels)
        except Exception as e:
            print(f"Error checking square color: {e}")
            return False

    first_empty = is_square_same_color(first_box[0], first_box[1])
    first_otc = not is_square_same_color(symbol_area[0], symbol_area[1])
    second_empty = is_square_same_color(second_box[0], second_box[1])

    if first_empty:
        return "Skip Trade"
    elif first_otc:
        if not second_empty:
            pyautogui.click(second_box[0][0] + 5, second_box[0][1] + 5)
            return "Click Second"
        else:
            return "Skip Trade"
    else:
        pyautogui.click(first_box[0][0] + 5, first_box[0][1] + 5)
        return "Click First"


def EnterLiveTrade(trade_list):
    """
    Example function that tries to place trades in some live environment.
    """
    trade = trade_list[0]
    perform_action("move", coords=(756, 337))
    perform_action("click", coords=(756, 337))
    perform_action("move", coords=(1038, 438))
    perform_action("click", coords=(1038, 438))
    index = 0
    while True:
        perform_action('type', text=trade['symbol'])
        first_box = ((986, 521), (1482, 566))
        second_box = ((986, 566), (1482, 611))
        symbol_box = ((1190, 521), (1400, 566))
        action = handle_ticker_logic(first_box, second_box, symbol_box)
        if action == "Skip Trade":
            index += 1
            if index == len(trade_list):
                perform_action("move", coords=(1508, 329))
                perform_action("click", coords=(1508, 329))
                return None
            trade = trade_list[index]
            pyautogui.typewrite(['backspace'] * 10)
        else:
            break

    perform_action("move", coords=(1508, 329))
    perform_action("click", coords=(1508, 329))
    perform_action("move", coords=(1731, 347))
    perform_action("click", coords=(1731, 347))
    perform_action("move", coords=(1491, 412))
    perform_action("doubleclick", coords=(1491, 412))
    perform_action('type', text=trade['duration'])

    if trade['side'] == 'BUY':
        perform_action("move", coords=(1787, 613))
        perform_action("click", coords=(1787, 613))
    else:
        perform_action("move", coords=(1770, 699))
        perform_action("click", coords=(1770, 699))
    return trade


def CheckIfMarketOpen(symbol_info):
    """
    Simple check if the forex market is open.
    """
    current_time = datetime.now(timezone.utc)
    current_day = (current_time.weekday() + 1) % 7
    current_hour = current_time.hour
    current_minute = current_time.minute
    screener = symbol_info.get('screener')

    # Forex: 24/5
    if screener == "forex":
        if current_day == 5 and current_hour >= 21:
            return False
        if current_day == 6:
            return False
        if current_day == 0:
            if current_hour < 22:
                return False
            return True
        return True

    # US stocks example
    if screener == "america":
        if current_day in [5, 6]:
            return False
        if (current_hour < 14) or (current_hour == 14 and current_minute < 30):
            return False
        if current_hour >= 21:
            return False
        return True

    return False


def WaitForCandleClose():
    current_time = datetime.now(UTC)
    seconds_until_next_minute = 60 - current_time.second
    future_time = current_time + timedelta(hours=2)
    formatted_time = future_time.strftime('%d/%m/%y - %H:%M:%S')
    print(f"\n🕒 Waiting for candle close - {formatted_time}")
    for remaining in range(seconds_until_next_minute, 0, -1):
        bar_length = 30
        filled_length = math.ceil((60 - remaining) / 60 * bar_length)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {remaining - 1} sec left", end='', flush=True)
        time.sleep(1)
    print("\n✅ Candle closed. Proceeding...")


def GetCurrentPrice(symbol):
    return float(symbol['last price'])


def predict_trades(brain, trades):

    # 1) Gather GAF images into an array for a single predict call
    X = [t['input'] for t in trades]
    X = np.array(X, dtype=np.float32)

    # 2) Model inference => shape (N, 3)
    preds = brain.predict(X)

    # 3) Build results
    results = []
    for i, prob_vector in enumerate(preds):
        # prob_vector might look like [0.3, 0.2, 0.5]
        side_idx = np.argmax(prob_vector)
        side_prob = prob_vector[side_idx]

        # Determine side label
        if side_idx == 0:
            side_label = 'BUY'
        elif side_idx == 1:
            side_label = 'SELL'
        else:
            side_label = 'NEUTRAL'

        # Skip if NEUTRAL
        if side_label == 'NEUTRAL':
            continue

        # 4) Add to the results
        results.append({
            'score': float(side_prob),
            'trade': trades[i],
            'side': side_label
        })

    # 5) Sort by abs(score) descending
    results.sort(key=lambda x: abs(x['score']), reverse=True)
    return results


def UpdatePrices(interval=Interval.INTERVAL_1_MINUTE):
    """
    Fetch the latest prices for all known symbols.
    """
    try:
        formatted_symbols = [f"{symbol['exchange']}:{symbol['symbol']}" for symbol in symbols]
        screener = symbols[0].get('screener', 'forex')
        analysis_data = get_multiple_analysis(
            screener=screener,
            interval=interval,
            symbols=formatted_symbols
        )
        for symbol in symbols:
            symbol_key = f"{symbol['exchange']}:{symbol['symbol']}"
            analysis = analysis_data.get(symbol_key)
            if analysis and 'close' in analysis.indicators:
                symbol['last price'] = str(analysis.indicators['close'])
            else:
                symbol['last price'] = '0'
    except Exception as e:
        print(f"Error updating prices: {e}")


class AITradeManager:
    def __init__(self):
        self.start_time = time.time()
        self.symbols = symbols
        self.durations = list(range(1, 6))
        self.open_trades_list = []
        self.params_names = list(fetch_historical_values(symbols[0]).keys())
        self.valid_symbols = []
        self.active_trades = []

        # In case no trades are loaded, add a dummy for shape
        trades = LoadTrades()
        if not trades:
            trades.append({'input': np.zeros((15, 30, 30)), 'results': {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}})
        self.retro_trades = trades
        # Build the EntryBrain / DirectionBrain for each duration:
        self.brains = load_brains("./saved_models")

        self.gaf_cache = {}
        self.historical_data = {symbol['symbol']: [] for symbol in symbols}
        self.historical_window = 30
        current_time = datetime.now(UTC)
        current_minute_time = current_time.replace(second=0, microsecond=0)
        self.last_recorded_time = current_minute_time - timedelta(minutes=1)

    def get_trades(self):
        self.validate_time()
        WaitForCandleClose()
        UpdatePrices()
        self.close_trades()
        self.update_valid_symbols()
        self.open_trades()

    def open_trades(self):
        self.update_gaf_cache()
        warmup = True
        trades = []
        for symbol in self.valid_symbols:
            if symbol['symbol'] in self.gaf_cache:
                gaf_image = self.gaf_cache[symbol['symbol']]
                if gaf_image is not None:
                    warmup = False
                    trades.append({
                        'symbol': symbol,
                        'input': gaf_image,
                        'price': GetCurrentPrice(symbol)
                    })
                    if settings['collect trades']:
                        price = GetCurrentPrice(symbol)
                        new_trade = Trade(symbol, self.durations, gaf_image, price)
                        self.open_trades_list.append(new_trade)
        if warmup or len(self.valid_symbols) == 0:
            print("❌ Not Enough Data To Predict Trades")
        else:
            print(f"✅ {len(self.open_trades_list)} Trades Are Open")
            if settings['suggest trades']:
                self.suggest_trades(trades, settings['auto execute'])

    def close_trades(self):
        trades_to_remove = []
        for idx in range(len(self.open_trades_list)):
            trade = self.open_trades_list[idx]
            finished = trade.check_for_close()
            if finished:
                self.retro_trades.append({
                    'symbol': trade.symbol,
                    'input': trade.original_input,
                    'results': trade.results
                })
                trades_to_remove.append(idx)
        for i in reversed(trades_to_remove):
            self.open_trades_list.pop(i)
        if trades_to_remove:
            print(f"\n✅ {len(trades_to_remove)} Trades Closed")
        else:
            print(f"\n✅ No Trades Needed To Close")

    def update_valid_symbols(self):
        self.valid_symbols = [symbol for symbol in symbols if CheckIfMarketOpen(symbol)]
        if not self.valid_symbols:
            print("❌ No valid symbols available for trading.")

    def generate_gaf(self, historical_data, symbol):
        try:
            if len(historical_data) < self.historical_window:
                raise ValueError(f"Not enough data to generate GAF. Need at least {self.historical_window} samples.")
            keys = historical_data[0].keys()
            data_matrix = np.array([[entry[key] for key in keys] for entry in historical_data])
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(data_matrix)
            gasf = GramianAngularField(image_size=self.historical_window)
            gaf_images = [
                gasf.fit_transform(scaled_data[:, i].reshape(1, -1))[0]
                for i in range(scaled_data.shape[1])
            ]
            self.gaf_cache[symbol['symbol']] = np.stack(gaf_images, axis=0)
        except Exception as e:
            return None

    def generate_historical_data(self, symbol):
        params = fetch_historical_values(symbol)
        self.historical_data[symbol['symbol']].append(params)
        if len(self.historical_data[symbol['symbol']]) > self.historical_window:
            self.historical_data[symbol['symbol']].pop(0)

    def update_gaf_cache(self):
        for symbol in self.symbols:
            self.generate_historical_data(symbol)
            historical_data = self.historical_data[symbol['symbol']]
            self.generate_gaf(historical_data, symbol)

    def validate_time(self):
        current_time = datetime.now(UTC)
        current_minute_time = current_time.replace(second=0, microsecond=0)
        if current_minute_time != self.last_recorded_time + timedelta(minutes=1):
            print("⏰ Update took too long, clearing open trades.")
            self.open_trades_list.clear()
            self.gaf_cache.clear()
            self.historical_data = {symbol['symbol']: [] for symbol in symbols}
        self.last_recorded_time = current_minute_time

    def suggest_trades(self, trade_list, auto_click):
        executable_trades = []
        for duration in map(str, self.durations):
            print(f"Predicting with EntryBrain for duration {duration}...")
            actionable_trades = predict_trades(self.brains[duration], trade_list)
            print(f"Number of actionable trades for duration {duration}: {len(actionable_trades)}")
            if actionable_trades:
                print(f"Predicting with DirectionBrain for duration {duration}...")
                for data in actionable_trades:
                    trade = data['trade']
                    predicted_result = data['side']
                    confidence = abs(data['score'])
                    price = GetCurrentPrice(trade['symbol'])
                    executable_trades.append({
                        'symbol': trade['symbol']['symbol'],
                        'side': predicted_result,
                        'duration': duration,
                        'score': confidence,
                        'price': price
                    })
        if executable_trades:
            executable_trades = sorted(executable_trades, key=lambda x: x['score'], reverse=True)
            if auto_click:
                best_trade = EnterLiveTrade(executable_trades)
            else:
                best_trade = executable_trades[0]
            if best_trade:
                print(f"Enter Trade - {int(best_trade['score'] * 100)}%")
                print(f"Symbol: {best_trade['symbol']}")
                print(f"Duration: {float(best_trade['duration'])}")
                print(f"Side: {best_trade['side']}")
                print(f"Price: {best_trade['price']}")
            else:
                print("❌ No Valid Trades Found")
        else:
            print("❌ No Valid Trades Found")


class Trade:
    """
    Represents an open trade in the system.
    """

    def __init__(self, symbol, durations, original_input, price):
        self.symbol = symbol
        self.open_price = price
        self.durations = durations
        self.original_input = original_input
        self.closing_prices = {}
        self.minutes_passed = 0
        self.results = {}

    def check_for_close(self):
        self.minutes_passed += 1
        for int_duration in self.durations:
            if str(int_duration) not in self.results and self.minutes_passed >= int_duration:
                current_price = GetCurrentPrice(self.symbol)
                self.closing_prices[str(int_duration)] = current_price
                price_change = (current_price - self.open_price) / self.open_price
                self.results[str(int_duration)] = price_change
        return len(self.results) == len(self.durations)