
import ccxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

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
data = df['close'].values.reshape(-1, 1)

# 2. تطبيع البيانات
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. إنشاء تسلسلات بيانات للـ LSTM
# عدد الخطوات الزمنية الماضية التي سيستخدمها النموذج للتنبؤ بالخطوة التالية
look_back = 60 # استخدام 60 ساعة سابقة للتنبؤ بالساعة التالية

X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:(i + look_back), 0])
    y.append(scaled_data[i + look_back, 0])
X, y = np.array(X), np.array(y)

# إعادة تشكيل البيانات لتناسب مدخلات LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# تقسيم البيانات إلى مجموعات تدريب واختبار
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# 4. بناء نموذج LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. تدريب النموذج
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 6. التنبؤ وتقييم النموذج
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# عكس التطبيع للحصول على الأسعار الحقيقية
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. رسم النتائج
plt.figure(figsize=(14, 7))
plt.plot(df.index[look_back:train_size+look_back], y_train_actual, label='Actual Train Prices')
plt.plot(df.index[look_back:train_size+look_back], train_predict, label='Predicted Train Prices')
plt.plot(df.index[train_size+look_back:], y_test_actual, label='Actual Test Prices')
plt.plot(df.index[train_size+look_back:], test_predict, label='Predicted Test Prices')
plt.title(f'BTC/USDT Price Prediction using LSTM (1h timeframe)')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.grid(True)
plt.savefig('lstm_prediction.png')
plt.close()

print("تم تدريب نموذج LSTM وحفظ الرسم البياني للتنبؤات في lstm_prediction.png")

