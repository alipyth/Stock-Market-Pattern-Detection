# شناسایی الگوهای نمودار سهام با YOLOv8

این پروژه به شما کمک می‌کنه که الگوهای مختلف نمودار سهام رو شناسایی کنید و تحلیل بهتری از بازار داشته باشید. ما از مدل YOLOv8 استفاده کردیم که قدرت و سرعت بالایی داره.

## الگوهای پشتیبانی‌شده
مدل ما می‌تونه الگوهای زیر رو تشخیص بده:
- **Head and shoulders bottom**: سر و شانه کف
- **Head and shoulders top**: سر و شانه سقف
- **M_Head**: الگوی M
- **StockLine**: خطوط روند سهام
- **Triangle**: مثلث
- **W_Bottom**: الگوی W کف

## چطور کار می‌کنه؟
1. **ورودی:** شما می‌تونید یک تصویر یا ویدیو از نمودار سهام رو آپلود کنید.
2. **تحلیل:** مدل الگوها رو شناسایی می‌کنه و روی نمودار نشون می‌ده.
3. **خروجی:** تصویر یا ویدیوی حاشیه‌نویسی‌شده به شما نمایش داده می‌شه و می‌تونید ذخیره‌ش کنید.

## ابزارهای استفاده‌شده
- **YOLOv8**: برای شناسایی الگوها
- **OpenCV**: برای پردازش تصاویر و ویدیوها
- **Streamlit**: برای رابط کاربری ساده و کاربرپسند

## چطور ازش استفاده کنم؟
1. کد رو دانلود کن.
2. مدل و کتابخونه‌های لازم رو نصب کن.
3. با اجرای فایل `app.py` در استریم‌لیت، شروع کن:
   ```bash
   streamlit run app.py
