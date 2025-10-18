
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. جلب البيانات
exchange = ccxt.kraken()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 2000 # نفس عدد الشموع المستخدمة لتدريب LSTM

ohclv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohclv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# استخدام أسعار الإغلاق
close_prices = df['close'].values

# 2. حساب المتوسطات المتحركة
# Short-term Moving Average (SMA_short)
# Long-term Moving Average (SMA_long)
short_window = 50
long_window = 200

df['SMA_short'] = df['close'].rolling(window=short_window, min_periods=1).mean()
df['SMA_long'] = df['close'].rolling(window=long_window, min_periods=1).mean()

# إزالة القيم NaN الناتجة عن rolling window
df.dropna(inplace=True)

# 3. محاكاة التداول وإدارة المخاطر
initial_capital = 10000 # رأس المال الأولي
capital = initial_capital
position = 0    # 0: لا توجد صفقة، 1: صفقة شراء، -1: صفقة بيع
entry_price = 0
stop_loss_percent = 0.02 # 2% وقف خسارة
take_profit_percent = 0.04 # 4% جني أرباح
trading_fee_percent = 0.001 # 0.1% رسوم تداول لكل صفقة (شراء وبيع)

portfolio_history = [capital]
peak_capital = capital
max_drawdown = 0

num_trades = 0
num_wins = 0
num_losses = 0
total_profit_loss = 0

# نبدأ المحاكاة من حيث تتوفر كلتا المتوسطات المتحركة
start_index = max(short_window, long_window)

for i in range(start_index, len(df) - 1):
    current_price = df['close'].iloc[i]
    previous_short_sma = df['SMA_short'].iloc[i-1]
    current_short_sma = df['SMA_short'].iloc[i]
    previous_long_sma = df['SMA_long'].iloc[i-1]
    current_long_sma = df['SMA_long'].iloc[i]

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

# حساب العائد الكلي
total_return = ((capital - initial_capital) / initial_capital) * 100
win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0

# 4. رسم أداء المحفظة
plt.figure(figsize=(14, 7))
plt.plot(df.index[start_index : start_index + len(portfolio_history)], portfolio_history, label="Portfolio Value")
plt.title("Portfolio Performance with MA Crossover Strategy (Baseline)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USDT)")
plt.legend()
plt.grid(True)
plt.savefig('ma_crossover_performance.png')
plt.close()

print(f"\n--- تقييم استراتيجية تقاطع المتوسطات المتحركة (Baseline) ---")
print(f"رأس المال الأولي: {initial_capital:.2f} USDT")
print(f"رأس المال النهائي: {capital:.2f} USDT")
print(f"العائد الكلي: {total_return:.2f}%")
print(f"الحد الأقصى للانخفاض (Max Drawdown): {max_drawdown:.2%}")
print(f"\nإحصائيات التداول:")
print(f"إجمالي عدد الصفقات: {num_trades}")
print(f"عدد الصفقات الرابحة: {num_wins}")
print(f"عدد الصفقات الخاسرة: {num_losses}")
print(f"نسبة الصفقات الرابحة: {win_rate:.2f}%")
print("تم حفظ الرسم البياني لأداء المحفظة في ma_crossover_performance.png")

