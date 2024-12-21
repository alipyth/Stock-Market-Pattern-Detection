import streamlit as st
from ultralyticsplus import YOLO
import cv2
import numpy as np
from PIL import Image
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,500&family=Noto+Sans+Arabic:wght@500&display=swap');
html{direction: rtl}
.st-emotion-cache-1fttcpj , .st-emotion-cache-nwtri{display:none;}
.st-emotion-cache-5rimss p{text-align:right;font-family: 'DM Sans', sans-serif;
font-family: 'Noto Sans Arabic', sans-serif;
}
h1,h2,h3,h4,h5,h6{font-family: 'Noto Sans Arabic', sans-serif;}
span,p,a,button,ol,li {text-align:right;font-family: 'DM Sans', sans-serif;
font-family: 'Noto Sans Arabic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# عنوان برنامه
st.title("تشخیص الگو روی چارت")
st.subheader('این پروژه به شما امکان میدهد که الگو های تشکیل شده روی چارت را هم در ویدیو و هم در تصویر تشخیص دهید')
st.info('برای بار اول مدل باید دانلود شود ، پس ممکن است کمی طول بکشه ! اما برای دفعات بعد بلافاصله پاسخ رو میگیرید')
# بارگذاری مدل
@st.cache_resource  # برای کش کردن مدل
def load_model():
    model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # Maximum number of detections per image
    return model


# آپلود تصویر یا ویدیو
option = st.sidebar.selectbox("نوع چارت را انتخاب کنید", ("Image", "Video"))

if option == "Image":
    model = load_model()

    # آپلود تصویر
    uploaded_file = st.file_uploader("تصویر چارت رو آپلود کنید", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # خواندن تصویر
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # اجرای YOLOv8
        results = model(image)

        # نمایش تصویر حاشیه‌نویسی‌شده
        annotated_image = results[0].plot()
        st.image(annotated_image, channels="BGR", caption="Annotated Image")

        # ذخیره تصویر
        if st.button("دانلود تصویر"):
            output_path = "annotated_image.jpg"
            cv2.imwrite(output_path, annotated_image)
            st.success(f"تصویر ذخیره شد {output_path}")

elif option == "Video":
    model = load_model()

    # آپلود ویدیو
    uploaded_video = st.file_uploader("یک ویدیو آپلود کنید", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # نمایش ویدیو اصلی
        st.video(uploaded_video)

        # پردازش ویدیو
        cap = cv2.VideoCapture(uploaded_video.name)
        output_frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # اجرای YOLOv8 روی فریم
            results = model(frame)
            annotated_frame = results[0].plot()
            output_frames.append(annotated_frame)

        cap.release()

        # تبدیل فریم‌های پردازش‌شده به ویدیو
        output_path = "annotated_video.mp4"
        height, width, _ = output_frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in output_frames:
            out.write(frame)
        out.release()

        # نمایش لینک دانلود
        with open(output_path, "rb") as file:
            st.download_button(label="دانلود ویدیو", data=file, file_name="annotated_video.mp4")
