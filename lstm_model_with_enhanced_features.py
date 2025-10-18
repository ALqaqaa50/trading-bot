
import ccxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# 1. جلب البيانات
exchange = ccxt.kraken()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 2000

ohclv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohclv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# استخدام أسعار الإغلاق
close_prices = df['close'].values

# 2. تطبيق تحويل فورييه السريع (FFT) لاستخراج الميزات
N_fft = len(close_prices)
yf_transform = fft(close_prices)

K = 20 # عدد الترددات المهيمنة التي سنحتفظ بها
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
# RSI
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

# MACD
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 4. دمج جميع الميزات
# يجب التأكد من أن جميع الميزات لها نفس الطول
# نقوم بإزالة الصفوف التي تحتوي على قيم NaN بعد حساب المؤشرات
df_features = df[['close', 'RSI', 'MACD', 'Signal_Line']].copy()
# يجب أن يكون طول fft_features هو نفس طول df_features بعد dropna
# لذلك، سنقوم بدمجها أولاً ثم dropna
df_features['FFT_Feature'] = fft_features # دمج مبدئي
df_features.dropna(inplace=True)

# الآن، الميزات هي: close, RSI, MACD, Signal_Line, FFT_Feature
# يجب أن تكون جميع الميزات رقمية
features = df_features.values

# 5. تطبيع البيانات
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

look_back = 60

X, y = [], []
for i in range(len(scaled_features) - look_back):
    X.append(scaled_features[i:(i + look_back), :])
    y.append(scaled_features[i + look_back, 0]) # ما زلنا نتنبأ بالسعر فقط (العمود الأول)
X, y = np.array(X), np.array(y)

# إعادة تشكيل البيانات لتناسب مدخلات LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], features.shape[1])) # عدد الميزات الجديدة

# تقسيم البيانات إلى مجموعات تدريب واختبار
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# 6. بناء نموذج LSTM المحسن
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(look_back, features.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. تدريب النموذج
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1, verbose=0) # زيادة epochs و batch_size

# 8. التنبؤ وتقييم النموذج
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# عكس التطبيع للحصول على الأسعار الحقيقية
# نحتاج إلى إعادة تشكيل التنبؤات لتناسب أبعاد scaler (عدد الميزات الأصلية)
# ثم نأخذ العمود الأول (السعر)

train_predict_full_dim = np.zeros((train_predict.shape[0], features.shape[1]))
train_predict_full_dim[:, 0] = train_predict[:, 0]
train_predict_prices = scaler.inverse_transform(train_predict_full_dim)[:, 0]

test_predict_full_dim = np.zeros((test_predict.shape[0], features.shape[1]))
test_predict_full_dim[:, 0] = test_predict[:, 0]
test_predict_prices = scaler.inverse_transform(test_predict_full_dim)[:, 0]

y_train_actual_full_dim = np.zeros((y_train.shape[0], features.shape[1]))
y_train_actual_full_dim[:, 0] = y_train
y_train_actual = scaler.inverse_transform(y_train_actual_full_dim)[:, 0]

y_test_actual_full_dim = np.zeros((y_test.shape[0], features.shape[1]))
y_test_actual_full_dim[:, 0] = y_test
y_test_actual = scaler.inverse_transform(y_test_actual_full_dim)[:, 0]

# 9. رسم النتائج
plt.figure(figsize=(14, 7))
# تصحيح مؤشرات الوقت للرسم البياني
# يجب أن تتوافق أطوال السلاسل الزمنية مع أطوال البيانات الفعلية والمتوقعة

# مؤشرات الوقت لبيانات التدريب
train_indices = df_features.index[look_back : look_back + len(y_train_actual)]
plt.plot(train_indices, y_train_actual, label='Actual Train Prices')
plt.plot(train_indices, train_predict_prices, label='Predicted Train Prices')

# مؤشرات الوقت لبيانات الاختبار
test_indices = df_features.index[train_size + look_back : train_size + look_back + len(y_test_actual)]
plt.plot(test_indices, y_test_actual, label='Actual Test Prices')
plt.plot(test_indices, test_predict_prices, label='Predicted Test Prices')

plt.title(f'BTC/USDT Price Prediction using LSTM with Enhanced Features (1h timeframe)')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.grid(True)
plt.savefig('lstm_enhanced_prediction.png')
plt.close()

model.save("lstm_enhanced_model.h5")
print("تم تدريب نموذج LSTM مع الميزات المحسنة وحفظ الرسم البياني للتنبؤات في lstm_enhanced_prediction.png")
print("تم حفظ النموذج المدرب في lstm_enhanced_model.h5")

