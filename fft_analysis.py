
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
import time

# 1. اختيار الأصل المالي وجلب البيانات باستخدام CCXT
exchange = ccxt.kraken() # استخدام بينانس كمثال، يمكن تغييرها لبورصة أخرى
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 1500 # عدد الشموع (تقريباً 60 يوم * 24 ساعة = 1440 شمعة)

ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# استخدام أسعار الإغلاق للتحليل
close_prices = df['close'].values

# 2. تطبيق تحويل فورييه السريع (FFT)
N = len(close_prices) # عدد نقاط البيانات
T = 1/24 # فترة أخذ العينات (ساعة واحدة = 1/24 من اليوم)

yf_transform = fft(close_prices)
x = fftfreq(N, T)[:N//2]
y = 2.0/N * np.abs(yf_transform[0:N//2])

# 3. حفظ نتائج FFT في ملف نصي (للاستخدام المستقبلي إذا لزم الأمر)
np.savetxt('fft_results.txt', np.column_stack([x, y]), header='Frequency Amplitude', comments='')

# 4. رسم بياني لنتائج FFT
plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.title(f'FFT Analysis of {symbol} Close Prices ({timeframe} timeframe)')
plt.xlabel('Frequency (Cycles per Day)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('fft_analysis.png')
plt.close()

print(f"تم تحليل FFT لـ {symbol} وحفظ الرسم البياني في fft_analysis.png")
print("تم حفظ نتائج التردد والسعة في fft_results.txt")

