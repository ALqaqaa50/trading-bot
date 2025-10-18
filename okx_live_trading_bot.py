#!/usr/bin/env python3.11
"""
OKX Live Trading Bot - استراتيجية تقاطع المتوسطات المتحركة (MA Crossover)

هذا السكريبت يربط استراتيجية MA Crossover المحسنة بمنصة OKX للتداول المباشر.

تحذير: هذا السكريبت للأغراض التعليمية والتوضيحية فقط.
للتداول الحقيقي، يجب:
1. إنشاء حساب OKX وتفعيل API
2. الحصول على مفاتيح API (API Key, Secret, Passphrase)
3. البدء بحساب تجريبي أو بأموال صغيرة جدًا
4. إضافة المزيد من آليات الأمان والمراقبة
"""

import ccxt
import pandas as pd
import time
import logging
from datetime import datetime

# إعداد نظام السجلات (Logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class OKXTradingBot:
    def __init__(self, api_key=None, api_secret=None, api_passphrase=None, 
                 symbol='BTC/USDT', timeframe='1h',
                 short_window=25, long_window=80,
                 stop_loss_percent=0.02, take_profit_percent=0.04,
                 trading_fee_percent=0.001, position_size_usdt=100):
        """
        تهيئة بوت التداول
        
        المعاملات:
        - api_key: مفتاح API من OKX
        - api_secret: السر الخاص بـ API
        - api_passphrase: عبارة المرور لـ API
        - symbol: زوج التداول (افتراضي: BTC/USDT)
        - timeframe: الإطار الزمني (افتراضي: 1h)
        - short_window: نافذة المتوسط المتحرك القصير
        - long_window: نافذة المتوسط المتحرك الطويل
        - stop_loss_percent: نسبة وقف الخسارة
        - take_profit_percent: نسبة جني الأرباح
        - trading_fee_percent: نسبة رسوم التداول
        - position_size_usdt: حجم الصفقة بالدولار
        """
        
        # إعداد الاتصال بـ OKX
        if api_key and api_secret and api_passphrase:
            self.exchange = ccxt.okx({
                'apiKey': api_key,
                'secret': api_secret,
                'password': api_passphrase,
                'enableRateLimit': True,
            })
            logging.info("تم الاتصال بـ OKX API بنجاح")
        else:
            # إذا لم يتم توفير مفاتيح API، استخدم الوضع العام (بدون تداول حقيقي)
            self.exchange = ccxt.okx({'enableRateLimit': True})
            logging.warning("لم يتم توفير مفاتيح API. البوت يعمل في وضع المراقبة فقط (بدون تداول حقيقي)")
        
        # معلمات التداول
        self.symbol = symbol
        self.timeframe = timeframe
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.trading_fee_percent = trading_fee_percent
        self.position_size_usdt = position_size_usdt
        
        # حالة الصفقة الحالية
        self.position = 0  # 0: لا توجد صفقة، 1: صفقة شراء، -1: صفقة بيع
        self.entry_price = 0
        self.order_id = None
        
        logging.info(f"تم تهيئة البوت: {symbol}, MA({short_window},{long_window}), SL={stop_loss_percent*100}%, TP={take_profit_percent*100}%")
    
    def fetch_historical_data(self, limit=200):
        """
        جلب البيانات التاريخية
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logging.error(f"خطأ في جلب البيانات التاريخية: {e}")
            return None
    
    def calculate_moving_averages(self, df):
        """
        حساب المتوسطات المتحركة
        """
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        return df
    
    def check_crossover_signal(self, df):
        """
        التحقق من إشارات تقاطع المتوسطات المتحركة
        
        العودة:
        - 'buy': إشارة شراء
        - 'sell': إشارة بيع
        - None: لا توجد إشارة
        """
        if len(df) < 2:
            return None
        
        # الحصول على آخر قيمتين للمتوسطات المتحركة
        previous_short_sma = df['SMA_short'].iloc[-2]
        current_short_sma = df['SMA_short'].iloc[-1]
        previous_long_sma = df['SMA_long'].iloc[-2]
        current_long_sma = df['SMA_long'].iloc[-1]
        
        # التحقق من التقاطع
        if previous_short_sma <= previous_long_sma and current_short_sma > current_long_sma:
            return 'buy'
        elif previous_short_sma >= previous_long_sma and current_short_sma < current_long_sma:
            return 'sell'
        
        return None
    
    def place_order(self, side, amount):
        """
        تنفيذ أمر تداول
        
        المعاملات:
        - side: 'buy' أو 'sell'
        - amount: كمية العملة
        """
        try:
            order = self.exchange.create_market_order(self.symbol, side, amount)
            logging.info(f"تم تنفيذ أمر {side}: {order}")
            return order
        except Exception as e:
            logging.error(f"خطأ في تنفيذ الأمر: {e}")
            return None
    
    def check_risk_management(self, current_price):
        """
        التحقق من إدارة المخاطر (وقف الخسارة وجني الأرباح)
        
        العودة:
        - 'close': يجب إغلاق الصفقة
        - None: الاستمرار في الصفقة
        """
        if self.position == 0:
            return None
        
        if self.position == 1:  # صفقة شراء
            # وقف الخسارة
            if current_price <= self.entry_price * (1 - self.stop_loss_percent):
                logging.info(f"تم تفعيل وقف الخسارة: السعر الحالي {current_price}, سعر الدخول {self.entry_price}")
                return 'close'
            # جني الأرباح
            elif current_price >= self.entry_price * (1 + self.take_profit_percent):
                logging.info(f"تم تفعيل جني الأرباح: السعر الحالي {current_price}, سعر الدخول {self.entry_price}")
                return 'close'
        
        elif self.position == -1:  # صفقة بيع
            # وقف الخسارة
            if current_price >= self.entry_price * (1 + self.stop_loss_percent):
                logging.info(f"تم تفعيل وقف الخسارة: السعر الحالي {current_price}, سعر الدخول {self.entry_price}")
                return 'close'
            # جني الأرباح
            elif current_price <= self.entry_price * (1 - self.take_profit_percent):
                logging.info(f"تم تفعيل جني الأرباح: السعر الحالي {current_price}, سعر الدخول {self.entry_price}")
                return 'close'
        
        return None
    
    def run(self, check_interval=60):
        """
        تشغيل البوت
        
        المعاملات:
        - check_interval: الفترة الزمنية بين كل فحص (بالثواني)
        """
        logging.info("بدء تشغيل البوت...")
        
        while True:
            try:
                # جلب البيانات التاريخية
                df = self.fetch_historical_data(limit=max(self.long_window + 50, 200))
                
                if df is None or len(df) < self.long_window:
                    logging.warning("بيانات غير كافية. سيتم المحاولة مرة أخرى...")
                    time.sleep(check_interval)
                    continue
                
                # حساب المتوسطات المتحركة
                df = self.calculate_moving_averages(df)
                df = df.dropna()
                
                if len(df) < 2:
                    logging.warning("بيانات غير كافية بعد حساب المتوسطات المتحركة. سيتم المحاولة مرة أخرى...")
                    time.sleep(check_interval)
                    continue
                
                # الحصول على السعر الحالي
                current_price = df['close'].iloc[-1]
                
                # التحقق من إدارة المخاطر
                risk_action = self.check_risk_management(current_price)
                if risk_action == 'close':
                    # إغلاق الصفقة
                    if self.position == 1:
                        logging.info(f"إغلاق صفقة الشراء عند السعر {current_price}")
                        # في الوضع الحقيقي، سيتم تنفيذ أمر بيع هنا
                        # self.place_order('sell', amount)
                    elif self.position == -1:
                        logging.info(f"إغلاق صفقة البيع عند السعر {current_price}")
                        # في الوضع الحقيقي، سيتم تنفيذ أمر شراء لتغطية البيع على المكشوف
                        # self.place_order('buy', amount)
                    
                    self.position = 0
                    self.entry_price = 0
                    continue
                
                # التحقق من إشارات التقاطع (فقط إذا لم تكن هناك صفقة مفتوحة)
                if self.position == 0:
                    signal = self.check_crossover_signal(df)
                    
                    if signal == 'buy':
                        logging.info(f"إشارة شراء عند السعر {current_price}")
                        # في الوضع الحقيقي، سيتم تنفيذ أمر شراء هنا
                        # amount = self.position_size_usdt / current_price
                        # order = self.place_order('buy', amount)
                        # if order:
                        #     self.position = 1
                        #     self.entry_price = current_price
                        
                        # للأغراض التوضيحية فقط
                        self.position = 1
                        self.entry_price = current_price
                        logging.info(f"تم فتح صفقة شراء عند السعر {self.entry_price}")
                    
                    elif signal == 'sell':
                        logging.info(f"إشارة بيع عند السعر {current_price}")
                        # في الوضع الحقيقي، سيتم تنفيذ أمر بيع هنا
                        # amount = self.position_size_usdt / current_price
                        # order = self.place_order('sell', amount)
                        # if order:
                        #     self.position = -1
                        #     self.entry_price = current_price
                        
                        # للأغراض التوضيحية فقط
                        self.position = -1
                        self.entry_price = current_price
                        logging.info(f"تم فتح صفقة بيع عند السعر {self.entry_price}")
                
                # طباعة الحالة الحالية
                logging.info(f"الحالة: السعر={current_price:.2f}, SMA_short={df['SMA_short'].iloc[-1]:.2f}, SMA_long={df['SMA_long'].iloc[-1]:.2f}, الصفقة={'شراء' if self.position == 1 else 'بيع' if self.position == -1 else 'لا توجد'}")
                
                # الانتظار قبل الفحص التالي
                time.sleep(check_interval)
            
            except KeyboardInterrupt:
                logging.info("تم إيقاف البوت بواسطة المستخدم")
                break
            except Exception as e:
                logging.error(f"خطأ غير متوقع: {e}")
                time.sleep(check_interval)

if __name__ == "__main__":
    # مثال على الاستخدام
    # للتداول الحقيقي، استبدل None بمفاتيح API الخاصة بك
    bot = OKXTradingBot(
        api_key=None,  # ضع مفتاح API هنا
        api_secret=None,  # ضع السر الخاص بـ API هنا
        api_passphrase=None,  # ضع عبارة المرور هنا
        symbol='BTC/USDT',
        timeframe='1h',
        short_window=25,
        long_window=80,
        stop_loss_percent=0.02,
        take_profit_percent=0.04,
        position_size_usdt=100
    )
    
    # تشغيل البوت (سيتم الفحص كل 60 ثانية)
    bot.run(check_interval=60)

