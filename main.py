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
import requests

# استرداد عنوان الـ IP المستخدم في إرسال الطلبات
response = requests.get("https://api64.ipify.org?format=json")
ip_address = response.json()["ip"]

print(f"🔍 عنوان IP المستخدم في البرنامج: {ip_address}")
