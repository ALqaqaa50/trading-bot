
import ccxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# 1. جلب البيانات
exchange = ccxt.kraken()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 2000 # جلب المزيد من البيانات للتدريب

ohclv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohclv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# استخدام أسعار الإغلاق
close_prices = df['close'].values

# 2. تطبيق تحويل فورييه السريع (FFT) لاستخراج الميزات
N_fft = len(close_prices)

yf_transform = fft(close_prices)

# إعادة بناء الإشارة باستخدام عدد محدود من الترددات المهيمنة
K = 20 # عدد الترددات المهيمنة التي سنحتفظ بها

yf_transform_filtered = np.zeros_like(yf_transform, dtype=complex)

# نأخذ أعلى K/2 ترددات من النصف الموجب (بعد استبعاد التردد الصفري إن وجد)
positive_freq_transform = yf_transform[1:N_fft//2]
sorted_positive_indices = np.argsort(np.abs(positive_freq_transform))[::-1]

num_to_keep = min(K // 2, len(sorted_positive_indices))
top_positive_indices = sorted_positive_indices[:num_to_keep]

# إضافة التردد الصفري (المتوسط) إذا كان مهماً
yf_transform_filtered[0] = yf_transform[0]

# إضافة الترددات الموجبة والسالبة المقابلة
for idx in top_positive_indices:
    yf_transform_filtered[idx + 1] = yf_transform[idx + 1] # +1 لأننا تجاهلنا العنصر 0
    yf_transform_filtered[N_fft - (idx + 1)] = yf_transform[N_fft - (idx + 1)]

# تطبيق Inverse FFT لإعادة بناء الإشارة
fft_reconstructed_signal = ifft(yf_transform_filtered)
fft_features = np.real(fft_reconstructed_signal).reshape(-1, 1)

# دمج أسعار الإغلاق مع ميزات FFT
combined_data = np.hstack((close_prices.reshape(-1, 1), fft_features))

# 3. تطبيع البيانات
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(combined_data)

look_back = 60 # استخدام 60 ساعة سابقة للتنبؤ بالساعة التالية

X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:(i + look_back), :]) # الآن X سيحتوي على ميزتين (السعر و FFT)
    y.append(scaled_data[i + look_back, 0]) # ما زلنا نتنبأ بالسعر فقط (العمود الأول)
X, y = np.array(X), np.array(y)

# إعادة تشكيل البيانات لتناسب مدخلات LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], combined_data.shape[1])) # عدد الميزات هو 2 الآن

# تقسيم البيانات إلى مجموعات تدريب واختبار
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# 4. بناء نموذج LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, combined_data.shape[1])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. تدريب النموذج
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

# 6. التنبؤ وتقييم النموذج
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# عكس التطبيع للحصول على الأسعار الحقيقية
# عند عكس التطبيع، نحتاج إلى إعادة تشكيل التنبؤات لتناسب أبعاد scaler (عدد الميزات الأصلية)
# ثم نأخذ العمود الأول (السعر)

train_predict_full_dim = np.zeros((train_predict.shape[0], combined_data.shape[1]))
train_predict_full_dim[:, 0] = train_predict[:, 0]
train_predict_prices = scaler.inverse_transform(train_predict_full_dim)[:, 0]

test_predict_full_dim = np.zeros((test_predict.shape[0], combined_data.shape[1]))
test_predict_full_dim[:, 0] = test_predict[:, 0]
test_predict_prices = scaler.inverse_transform(test_predict_full_dim)[:, 0]

y_train_actual_full_dim = np.zeros((y_train.shape[0], combined_data.shape[1]))
y_train_actual_full_dim[:, 0] = y_train
y_train_actual = scaler.inverse_transform(y_train_actual_full_dim)[:, 0]

y_test_actual_full_dim = np.zeros((y_test.shape[0], combined_data.shape[1]))
y_test_actual_full_dim[:, 0] = y_test
y_test_actual = scaler.inverse_transform(y_test_actual_full_dim)[:, 0]

# 7. رسم النتائج
plt.figure(figsize=(14, 7))
plt.plot(df.index[look_back:train_size+look_back], y_train_actual, label='Actual Train Prices')
plt.plot(df.index[look_back:train_size+look_back], train_predict_prices, label='Predicted Train Prices')
plt.plot(df.index[train_size+look_back:train_size+look_back+len(y_test_actual)], y_test_actual, label='Actual Test Prices')
plt.plot(df.index[train_size+look_back:train_size+look_back+len(test_predict_prices)], test_predict_prices, label='Predicted Test Prices')
plt.title(f'BTC/USDT Price Prediction using LSTM with FFT Features (1h timeframe)')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.grid(True)
plt.savefig('lstm_fft_prediction.png')
plt.close()

model.save("lstm_fft_model.h5")
print("تم تدريب نموذج LSTM مع ميزات FFT وحفظ الرسم البياني للتنبؤات في lstm_fft_prediction.png")
print("تم حفظ النموذج المدرب في lstm_fft_model.h5")

