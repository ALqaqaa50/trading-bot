import ccxt
import config  # استيراد المفاتيح من config.py

# إعداد اتصال Binance API
exchange = ccxt.binance({
    'apiKey': config.API_KEY,
    'secret': config.SECRET_KEY,
    'options': {'defaultType': 'spot'}  # للتداول الفوري
})

# اختبار الاتصال بـ Binance
try:
    balance = exchange.fetch_balance()
    print("✅ تم الاتصال بـ Binance بنجاح!")
    print("💰 الرصيد الحالي:", balance)
except Exception as e:
    print("❌ خطأ في الاتصال بـ Binance:", str(e))
