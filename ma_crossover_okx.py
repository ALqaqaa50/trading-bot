
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import ta # استيراد مكتبة ta

# 1. جلب البيانات
exchange = ccxt.okx() # تغيير المنصة إلى OKX
symbol = 'BTC/USDT'
timeframe = '1h'
required_limit = 5000 # العدد المطلوب من الشموع

ohclcv_data = []
# ابدأ من الآن واذهب للخلف بالقدر المطلوب لضمان الحصول على أحدث البيانات
since_timestamp = exchange.milliseconds() - required_limit * 60 * 60 * 1000 # تقريبي

while len(ohclcv_data) < required_limit:
    # جلب البيانات من الأقدم للأحدث
    fetched_data = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=100) # جلب 100 شمعة في كل مرة
    if not fetched_data:
        break
    ohclcv_data.extend(fetched_data)
    # تحديث since_timestamp للطلب التالي ليكون بعد آخر شمعة تم جلبها
    since_timestamp = fetched_data[-1][0] - 1 # -1 لتجنب تكرار آخر شمعة
    time.sleep(exchange.rateLimit / 1000) # احترام حدود API

# ترتيب البيانات من الأقدم للأحدث إذا تم جلبها بترتيب عكسي
ohclcv_data = sorted(ohclcv_data, key=lambda x: x[0])

df = pd.DataFrame(ohclcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# التأكد من أن لدينا العدد المطلوب من الشموع
if len(df) > required_limit:
    df = df.tail(required_limit) # الاحتفاظ بأحدث الشموع إذا كان هناك المزيد

print(f"Total OHLCV data points fetched: {len(df)}")

# 2. تقسيم البيانات إلى In-sample و Out-of-sample
split_ratio = 0.8 # 80% للتدريب (In-sample)، 20% للاختبار (Out-of-sample)
split_index = int(len(df) * split_ratio)

df_in_sample = df.iloc[:split_index]
df_out_of_sample = df.iloc[split_index:]

print(f"In-sample data points: {len(df_in_sample)}")
print(f"Out-of-sample data points: {len(df_out_of_sample)}")

# 3. محاكاة التداول وإدارة المخاطر (كـ دالة)
def run_backtest(df_data, short_window, long_window, initial_capital=10000,
                 stop_loss_percent=0.02, take_profit_percent=0.04, trading_fee_percent=0.001):

    df_copy = df_data.copy()
    df_copy['SMA_short'] = df_copy['close'].rolling(window=short_window).mean()
    df_copy['SMA_long'] = df_copy['close'].rolling(window=long_window).mean()
    df_sim = df_copy.dropna()

    if len(df_sim) < 2: # نحتاج نقطتين على الأقل للمقارنة
        return 0, 0, 0, 0, 0, 0, []

    capital = initial_capital
    position = 0
    entry_price = 0

    portfolio_history = [capital]
    peak_capital = capital
    max_drawdown = 0

    num_trades = 0
    num_wins = 0
    num_losses = 0
    total_profit_loss = 0

    for i in range(1, len(df_sim)):
        current_price = df_sim['close'].iloc[i]
        previous_short_sma = df_sim['SMA_short'].iloc[i-1]
        current_short_sma = df_sim['SMA_short'].iloc[i]
        previous_long_sma = df_sim['SMA_long'].iloc[i-1]
        current_long_sma = df_sim['SMA_long'].iloc[i]
        
        # تحديث قيمة المحفظة الحالية (لأغراض التتبع فقط)
        if position == 1: # إذا كنا في صفقة شراء
            current_portfolio_value = capital / entry_price * current_price
        elif position == -1: # إذا كنا في صفقة بيع (افتراض بيع على المكشوف بسيط)
            current_portfolio_value = capital * (2 - (current_price / entry_price))
        else:
            current_portfolio_value = capital
        
        portfolio_history.append(current_portfolio_value)

        # حساب الحد الأقصى للانخفاض (Max Drawdown)
        peak_capital = max(peak_capital, current_portfolio_value)
        drawdown = (peak_capital - current_portfolio_value) / peak_capital if peak_capital > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

        # منطق إدارة المخاطر والخروج من الصفقات
        if position == 1: # صفقة شراء مفتوحة
            # وقف الخسارة
            if current_price <= entry_price * (1 - stop_loss_percent):
                profit_loss = (current_price - entry_price) - (entry_price * trading_fee_percent * 2) # رسوم شراء وبيع
                capital += profit_loss
                total_profit_loss += profit_loss
                num_losses += 1
                position = 0
                continue
            # جني الأرباح
            elif current_price >= entry_price * (1 + take_profit_percent):
                profit_loss = (current_price - entry_price) - (entry_price * trading_fee_percent * 2) # رسوم شراء وبيع
                capital += profit_loss
                total_profit_loss += profit_loss
                num_wins += 1
                position = 0
                continue
        elif position == -1: # صفقة بيع مفتوحة (Short Sell)
            # وقف الخسارة
            if current_price >= entry_price * (1 + stop_loss_percent):
                profit_loss = (entry_price - current_price) - (entry_price * trading_fee_percent * 2) # رسوم بيع وشراء لتغطية
                capital += profit_loss
                total_profit_loss += profit_loss
                num_losses += 1
                position = 0
                continue
            # جني الأرباح
            elif current_price <= entry_price * (1 - take_profit_percent):
                profit_loss = (entry_price - current_price) - (entry_price * trading_fee_percent * 2) # رسوم بيع وشراء لتغطية
                capital += profit_loss
                total_profit_loss += profit_loss
                num_wins += 1
                position = 0
                continue

        # منطق فتح الصفقات (فقط إذا لم تكن هناك صفقة مفتوحة)
        if position == 0:
            # إشارة شراء: المتوسط المتحرك القصير يتقاطع فوق المتوسط المتحرك الطويل
            if previous_short_sma <= previous_long_sma and current_short_sma > current_long_sma:
                num_trades += 1
                position = 1 # فتح صفقة شراء
                entry_price = current_price
            # إشارة بيع: المتوسط المتحرك القصير يتقاطع تحت المتوسط المتحرك الطويل
            elif previous_short_sma >= previous_long_sma and current_short_sma < current_long_sma:
                num_trades += 1
                position = -1 # فتح صفقة بيع
                entry_price = current_price

    total_return = ((capital - initial_capital) / initial_capital) * 100
    win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
    return total_return, max_drawdown, num_trades, num_wins, num_losses, capital, portfolio_history

# 4. البحث الشبكي (Grid Search) عن أفضل المعلمات على بيانات In-sample
short_windows = range(10, 30, 5) # نطاق أكثر تركيزًا: من 10 إلى 25 بخطوات 5
long_windows = range(50, 100, 10) # نطاق أكثر تركيزًا: من 50 إلى 90 بخطوات 10

# نطاقات البحث لـ Stop-Loss و Take-Profit
stop_loss_percentages = [0.01, 0.02, 0.03] # 1%, 2%, 3%
take_profit_percentages = [0.02, 0.04, 0.06] # 2%, 4%, 6%

best_return_in_sample = -np.inf
best_params_in_sample = (0, 0, 0, 0)
best_portfolio_history_in_sample = []

print("\nStarting Grid Search for MA Crossover Strategy on In-sample data...")
for sw in short_windows:
    for lw in long_windows:
        if lw <= sw: # يجب أن يكون المتوسط الطويل أكبر من القصير
            continue
        for sl_p in stop_loss_percentages:
            for tp_p in take_profit_percentages:
                current_return, current_drawdown, num_trades, num_wins, num_losses, final_capital, hist = run_backtest(df_in_sample, sw, lw, stop_loss_percent=sl_p, take_profit_percent=tp_p)
                
                if current_return > best_return_in_sample:
                    best_return_in_sample = current_return
                    best_params_in_sample = (sw, lw, sl_p, tp_p)
                    best_portfolio_history_in_sample = hist
                
                print(f"  MA({sw},{lw}) SL={sl_p*100:.0f}% TP={tp_p*100:.0f}%: Return={current_return:.2f}%, Drawdown={current_drawdown:.2f}%, Trades={num_trades}, Wins={num_wins}, Losses={num_losses}, Final Capital={final_capital:.2f}")

print(f"\n--- Grid Search Results (In-sample) ---")
print(f"Best Parameters: Short MA={best_params_in_sample[0]}, Long MA={best_params_in_sample[1]}, SL={best_params_in_sample[2]*100:.0f}%, TP={best_params_in_sample[3]*100:.0f}%")
print(f"Best Return: {best_return_in_sample:.2f}%")

# 5. اختبار الاستراتيجية على بيانات Out-of-sample باستخدام أفضل المعلمات
print("\nTesting Best Strategy on Out-of-sample data...")
out_return, out_drawdown, out_num_trades, out_num_wins, out_num_losses, out_final_capital, out_hist = run_backtest(df_out_of_sample, best_params_in_sample[0], best_params_in_sample[1], stop_loss_percent=best_params_in_sample[2], take_profit_percent=best_params_in_sample[3])

print(f"\n--- Out-of-sample Test Results ---")
print(f"Best Parameters (from In-sample): Short MA={best_params_in_sample[0]}, Long MA={best_params_in_sample[1]}, SL={best_params_in_sample[2]*100:.0f}%, TP={best_params_in_sample[3]*100:.0f}%")
print(f"Return: {out_return:.2f}%")
print(f"Max Drawdown: {out_drawdown:.2f}%")
print(f"Total Trades: {out_num_trades}")
print(f"Winning Trades: {out_num_wins}")
print(f"Losing Trades: {out_num_losses}")
print(f"Final Capital: {out_final_capital:.2f}")

# 6. رسم أداء المحفظة لأفضل المعلمات على بيانات Out-of-sample
if out_hist:
    plt.figure(figsize=(14, 7))
    plt.plot(df_out_of_sample.index[len(df_out_of_sample) - len(out_hist):], out_hist, label="Portfolio Value")
    plt.title(f"Out-of-sample Portfolio Performance (OKX) - MA({best_params_in_sample[0]},{best_params_in_sample[1]}) SL={best_params_in_sample[2]*100:.0f}% TP={best_params_in_sample[3]*100:.0f}%")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USDT)")
    plt.legend()
    plt.grid(True)
    plt.savefig("ma_crossover_okx_out_of_sample_performance.png")
    plt.close()
    print("تم حفظ الرسم البياني لأداء المحفظة خارج العينة في ma_crossover_okx_out_of_sample_performance.png")
else:
    print("لم يتم العثور على صفقات أو بيانات كافية لرسم الأداء خارج العينة.")

# 7. رسم أداء المحفظة لأفضل المعلمات على بيانات In-sample
if best_portfolio_history_in_sample:
    plt.figure(figsize=(14, 7))
    plt.plot(df_in_sample.index[len(df_in_sample) - len(best_portfolio_history_in_sample):], best_portfolio_history_in_sample, label="Portfolio Value")
    plt.title(f"In-sample Portfolio Performance (OKX) - MA({best_params_in_sample[0]},{best_params_in_sample[1]}) SL={best_params_in_sample[2]*100:.0f}% TP={best_params_in_sample[3]*100:.0f}%")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USDT)")
    plt.legend()
    plt.grid(True)
    plt.savefig("ma_crossover_okx_in_sample_performance.png")
    plt.close()
    print("تم حفظ الرسم البياني لأداء المحفظة داخل العينة في ma_crossover_okx_in_sample_performance.png")
else:
    print("لم يتم العثور على صفقات أو بيانات كافية لرسم الأداء داخل العينة.")

