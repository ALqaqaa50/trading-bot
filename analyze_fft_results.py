# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# قراءة نتائج FFT من الملف
results = np.loadtxt('fft_results.txt', skiprows=1)
x = results[:, 0]
y = results[:, 1]

# استبعاد تردد الصفر (الذي يمثل متوسط السعر)
non_zero_freq_indices = np.where(x > 0)
x = x[non_zero_freq_indices]
y = y[non_zero_freq_indices]

# البحث عن أعلى 5 ترددات (أقوى الدورات)
# يتم فرز النتائج تنازليًا بناءً على السعة
sorted_indices = np.argsort(y)[::-1]

print("أقوى 5 دورات زمنية مهيمنة في بيانات BTC/USDT:")
print("="*40)
print(f"{ 'التردد (دورات/يوم)':<25} | {'الدورة (أيام)':<15}")
print("-"*40)

for i in range(5):
    idx = sorted_indices[i]
    freq = x[idx]
    period_days = 1 / freq
    print(f"{freq:<25.5f} | {period_days:<15.2f}")

# رسم بياني لتوضيح الدورات المهيمنة
plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.title(f'Dominant Cycles in BTC/USDT FFT Analysis')
plt.xlabel('Frequency (Cycles per Day)')
plt.ylabel('Amplitude')
plt.grid(True)

# إضافة علامات على الرسم البياني للدورات المهيمنة
for i in range(5):
    idx = sorted_indices[i]
    freq = x[idx]
    amp = y[idx]
    period_days = 1 / freq
    plt.plot(freq, amp, 'ro') # 'ro' for red circle
    plt.text(freq, amp, f' {period_days:.1f} days', verticalalignment='bottom')

plt.savefig('dominant_cycles_analysis.png')
plt.close()

print("\nتم إنشاء رسم بياني يوضح الدورات المهيمنة وحفظه في dominant_cycles_analysis.png")

