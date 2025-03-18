import numpy as np
from tradingview_ta import TA_Handler, Interval, get_multiple_analysis
import time
from datetime import datetime, timedelta, timezone
import math
from PIL import ImageGrab
import pyautogui
from data_manager.trade_storage import LoadTrades, SaveTrades
from config import symbols
from rf_pipline.load_and_save import load_rf_models


def fetch_inputs(symbol):
    try:
        handler = TA_Handler(
            symbol=symbol['symbol'],
            screener=symbol['screener'],
            exchange=symbol['exchange'],
            interval=Interval.INTERVAL_1_MINUTE
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators

        # Core indicators
        rsi = indicators.get('RSI', 0)
        macd = indicators.get('MACD.macd', 0)
        macd_signal = indicators.get('MACD.signal', 0)
        ema7 = indicators.get('EMA7') or indicators.get('EMA10', 0)
        ema21 = indicators.get('EMA20') or indicators.get('EMA30', 0)
        sma7 = indicators.get('SMA7') or indicators.get('SMA5', 0)
        sma21 = indicators.get('SMA20') or indicators.get('SMA30', 0)
        williams_r = indicators.get('W.R', 0)
        volume = indicators.get('volume', 0)

        # Normalize volume (optional scaling)
        normalized_volume = volume / (volume + 1e-6)

        # Assemble relevant input vector
        inputs = [
            rsi,
            macd,
            macd_signal,
            ema7,
            ema21,
            sma7,
            sma21,
            williams_r,
            normalized_volume
        ]

        return inputs

    except Exception as e:
        print(f"Error fetching indicators: {e}")
        return [0] * 9  # Return zeroed inputs if failed


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
        perform_action('type', text=trade['symbol']['symbol'])
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

    if trade['direction'] == 'BUY':
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
    current_time = datetime.now(timezone.utc)
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


def ConvertToRFFormat(data):
    if len(data) != 30 or any(len(row) != 9 for row in data):
        raise ValueError("Invalid input format. Expected 30 lists with 15 elements each.")
    transformed_data = [[data[j][i] for j in range(30)] for i in range(9)]

    return transformed_data


def predict_trades(trades, trade_detector_models, trade_direction_models):
    detected_trades = []

    for duration in trade_direction_models.keys():
        trade_detector = trade_detector_models[duration]
        trade_direction = trade_direction_models[duration]

        print(f"\n🔹 Processing trades for {duration}-minute duration...")

        # Extract inputs
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])

        # **Step 1: Trade Detection**
        trade_predictions = trade_detector.predict(X)
        detected_indices = np.where(trade_predictions == 1)[0]

        if len(detected_indices) == 0:
            continue

        detected_trades_batch = [trades[i] for i in detected_indices]

        # **Step 2: Trade Direction Prediction**
        X_detected = np.array([[x for xs in trade['input'] for x in xs] for trade in detected_trades_batch])
        direction_predictions = trade_direction.predict(X_detected)
        direction_confidences = trade_direction.predict_proba(X_detected)  # Probabilities for each class

        for i, trade in enumerate(detected_trades_batch):
            trade['duration'] = duration  # Store trade duration
            trade['direction'] = "BUY" if direction_predictions[i] == 1 else "SELL"
            trade['confidence'] = max(direction_confidences[i])  # Use max probability as confidence
            if trade['confidence'] > 0.7:
                detected_trades.append(trade)

    # **Step 3: Sort trades by confidence (descending)**
    sorted_trades = sorted(detected_trades, key=lambda x: x['confidence'], reverse=True)

    print(f"✅ {len(sorted_trades)} trades detected and ranked by confidence.")

    return sorted_trades


class AITradeManager:
    def __init__(self):
        self.start_time = time.time()
        self.symbols = symbols
        self.durations = list(range(1, 6))
        self.open_trades_list = []
        self.valid_symbols = []
        self.models = {
            'mag': load_rf_models('mag'),
            'dir': load_rf_models('dir')
        }
        # In case no trades are loaded, add a dummy for shape
        self.added_trades = 0
        trades = LoadTrades()
        self.retro_trades = trades
        self.historical_data = {symbol['symbol']: [] for symbol in symbols}
        self.historical_window = 30
        current_time = datetime.now(timezone.utc)
        current_minute_time = current_time.replace(second=0, microsecond=0)
        self.last_recorded_time = current_minute_time - timedelta(minutes=1)

    def get_trades(self, collect=False):
        self.validate_time()
        WaitForCandleClose()
        UpdatePrices()
        self.close_trades()
        self.update_valid_symbols()
        self.open_trades(collect)

    def open_trades(self, collect):
        self.update_symbols_data()
        warmup = True
        trades = []
        for symbol in self.valid_symbols:
            if len(self.historical_data[symbol['symbol']]) == self.historical_window:
                warmup = False
                inputs = ConvertToRFFormat(self.historical_data[symbol['symbol']])
                trades.append({
                    'symbol': symbol,
                    'input': inputs,
                    'price': GetCurrentPrice(symbol)
                })
                if collect:
                    price = GetCurrentPrice(symbol)
                    new_trade = Trade(symbol, self.durations, inputs, price)
                    self.open_trades_list.append(new_trade)
        if len(self.valid_symbols) == 0:
            print("❌ All Assets Unavailable For Trading")
        elif warmup:
            progress = len(self.historical_data[symbols[0]['symbol']])/self.historical_window
            time_left = self.historical_window - len(self.historical_data[symbols[0]['symbol']])
            print(f"❌ Warmup Stage {int(progress*1000)/10}% | Ready In {time_left} Min")
        else:
            print(f"✅ {len(self.open_trades_list)} Trades Are Open")
            if not collect:
                self.suggest_trades(trades, False)

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
            self.added_trades += 1
            if self.added_trades % 5000 == 0:
                SaveTrades(self.retro_trades)
                print("Saved 5000 trades")
        if trades_to_remove:
            print(f"\n✅ {len(trades_to_remove)} Trades Closed")
        else:
            print(f"\n✅ No Trades Needed To Close")

    def update_valid_symbols(self):
        self.valid_symbols = []
        for symbol in symbols:
            if CheckIfMarketOpen(symbol):
                self.valid_symbols.append(symbol)
            else:
                self.historical_data[symbol['symbol']] = []
        if not self.valid_symbols:
            print("❌ No valid symbols available for trading.")

    def update_symbols_data(self):
        for symbol in self.symbols:
            params = fetch_inputs(symbol)
            self.historical_data[symbol['symbol']].append(params)
            if len(self.historical_data[symbol['symbol']]) > self.historical_window:
                self.historical_data[symbol['symbol']].pop(0)

    def validate_time(self):
        current_time = datetime.now(timezone.utc)
        current_minute_time = current_time.replace(second=0, microsecond=0)
        if current_minute_time != self.last_recorded_time + timedelta(minutes=1):
            print("⏰ Update took too long, clearing open trades.")
            self.open_trades_list.clear()
            self.historical_data = {symbol['symbol']: [] for symbol in symbols}
        self.last_recorded_time = current_minute_time

    def suggest_trades(self, trade_list, auto_click):
        executable_trades = predict_trades(trade_list, self.models['mag'], self.models['dir'])
        if executable_trades:
            if auto_click:
                best_trade = EnterLiveTrade(executable_trades)
            else:
                best_trade = executable_trades[0]
            if best_trade:
                print(f"Enter Trade - {int(best_trade['confidence'] * 100)}%")
                print(f"Symbol: {best_trade['symbol']['symbol']}")
                print(f"Duration: {float(best_trade['duration'])}")
                print(f"Side: {best_trade['direction']}")
                print(f"Price: {GetCurrentPrice(best_trade['symbol'])}")
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
