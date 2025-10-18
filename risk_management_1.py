
import ccxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# 1. جلب البيانات (نفس البيانات المستخدمة لتدريب LSTM)
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

# 2. تطبيق تحويل فورييه السريع (FFT) لاستخراج الميزات
N_fft = len(close_prices)
yf_transform = fft(close_prices)

K = 20 # عدد الترددات المهيمنة التي سنحتفظ بها (نفس العدد المستخدم عند تدريب النموذج)
yf_transform_filtered = np.zeros_like(yf_transform, dtype=complex)

positive_freq_transform = yf_transform[1:N_fft//2]
sorted_positive_indices = np.argsort(np.abs(positive_freq_transform))[::-1]
num_to_keep = min(K // 2, len(sorted_positive_indices))
top_positive_indices = sorted_positive_indices[:num_to_keep]

yf_transform_filtered[0] = yf_transform[0]

for idx in top_positive_indices:
    yf_transform_filtered[idx + 1] = yf_transform[idx + 1]
    yf_transform_filtered[N_fft - (idx + 1)] = yf_transform[N_fft - (idx + 1)]

fft_reconstructed_signal = ifft(yf_transform_filtered)
fft_features = np.real(fft_reconstructed_signal).reshape(-1, 1)

# 3. حساب المؤشرات الفنية (RSI, MACD)
def calculate_rsi(data, window=14):
    diff = data.diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['close'])

exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 4. دمج جميع الميزات
df_features = df[['close', 'RSI', 'MACD', 'Signal_Line']].copy()
df_features['FFT_Feature'] = fft_features[:len(df_features)]
df_features.dropna(inplace=True)

features = df_features.values

# 5. تطبيع البيانات
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

look_back = 60

X_full, y_full = [], []
for i in range(len(scaled_features) - look_back):
    X_full.append(scaled_features[i:(i + look_back), :])
    y_full.append(scaled_features[i + look_back, 0])
X_full, y_full = np.array(X_full), np.array(y_full)

X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], features.shape[1]))

# تقسيم البيانات إلى مجموعات تدريب واختبار (نفس طريقة LSTM)
train_size = int(len(X_full) * 0.8)
X_test = X_full[train_size:len(X_full)]
y_test = y_full[train_size:len(y_full)]

# 6. تحميل نموذج LSTM المحسن
model = load_model("lstm_enhanced_model.h5")

# 7. التنبؤ باستخدام النموذج
test_predict_scaled = model.predict(X_test)

# عكس التطبيع للحصول على الأسعار الحقيقية
test_predict_full_dim = np.zeros((test_predict_scaled.shape[0], features.shape[1]))
test_predict_full_dim[:, 0] = test_predict_scaled[:, 0]
test_predict_prices = scaler.inverse_transform(test_predict_full_dim)[:, 0]

y_test_actual_full_dim = np.zeros((y_test.shape[0], features.shape[1]))
y_test_actual_full_dim[:, 0] = y_test
actual_test_prices = scaler.inverse_transform(y_test_actual_full_dim)[:, 0]

# حساب مقاييس RMSE و MAE
rmse = np.sqrt(mean_squared_error(actual_test_prices, test_predict_prices))
mae = mean_absolute_error(actual_test_prices, test_predict_prices)

# 8. محاكاة التداول وإدارة المخاطر
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

# مؤشر بسيط لاتخاذ القرار: إذا كان السعر المتوقع أعلى من السعر الحالي بنسبة معينة، نشتري
# وإذا كان أقل، نبيع. (هذا مجرد مثال توضيحي، يحتاج لتعقيد أكبر)
prediction_threshold = 0.005 # 0.5% فرق للتداول

# يجب أن تبدأ المحاكاة من حيث تبدأ بيانات الاختبار الفعلية
# وتأخذ في الاعتبار الـ look_back
start_simulation_index = len(df_features) - len(actual_test_prices)

for i in range(len(actual_test_prices) - 1):
    # current_price هو السعر الفعلي في الوقت الحالي (i)
    current_price = actual_test_prices[i]
    # predicted_next_price هو السعر المتوقع للساعة التالية (i+1)
    predicted_next_price = test_predict_prices[i + 1]
    
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
        num_trades += 1 # زيادة عدد الصفقات عند محاولة الفتح
        if predicted_next_price > current_price * (1 + prediction_threshold): # توقع ارتفاع
            position = 1 # فتح صفقة شراء
            entry_price = current_price
        elif predicted_next_price < current_price * (1 - prediction_threshold): # توقع انخفاض
            position = -1 # فتح صفقة بيع
            entry_price = current_price

# حساب العائد الكلي
total_return = ((capital - initial_capital) / initial_capital) * 100
win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0

# 9. رسم أداء المحفظة
plt.figure(figsize=(14, 7))
# يجب أن تتوافق أطوال السلاسل الزمنية مع أطوال البيانات الفعلية والمتوقعة
plt.plot(df_features.index[start_simulation_index : start_simulation_index + len(portfolio_history)], portfolio_history, label="Portfolio Value")
plt.title("Portfolio Performance with LSTM Enhanced Predictions and Risk Management")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USDT)")
plt.legend()
plt.grid(True)
plt.savefig('portfolio_performance_enhanced_lstm.png')
plt.close()

print(f"\n--- تقييم نموذج LSTM المحسن ومحاكاة التداول (مع ميزات FFT, RSI, MACD) ---")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"\nرأس المال الأولي: {initial_capital:.2f} USDT")
print(f"رأس المال النهائي: {capital:.2f} USDT")
print(f"العائد الكلي: {total_return:.2f}%")
print(f"الحد الأقصى للانخفاض (Max Drawdown): {max_drawdown:.2%}")
print(f"\nإحصائيات التداول:")
print(f"إجمالي عدد الصفقات: {num_trades}")
print(f"عدد الصفقات الرابحة: {num_wins}")
print(f"عدد الصفقات الخاسرة: {num_losses}")
print(f"نسبة الصفقات الرابحة: {win_rate:.2f}%")
print("تم حفظ الرسم البياني لأداء المحفظة في portfolio_performance_enhanced_lstm.png")

