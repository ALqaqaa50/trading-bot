import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    price = float(data['p'])  # استخراج سعر BTC/USDT
    print(f"السعر الحي: {price} USDT")

def on_error(ws, error):
    print(f"خطأ في WebSocket: {error}")

def on_close(ws, close_status_code, close_msg):
    print("تم إغلاق WebSocket")

def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@trade"],
        "id": 1
    }
    ws.send(json.dumps(payload))

# فتح اتصال WebSocket مع Binance
socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
ws = websocket.WebSocketApp(socket, on_message=on_message, on_error=on_error, on_close=on_close)
ws.on_open = on_open
ws.run_forever()
