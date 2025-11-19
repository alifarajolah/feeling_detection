import torch
import gradio as gr
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

# ایمپورت مدل خودت
from models.feeling_detection_net import FeelingDetectionNet
from config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = FeelingDetectionNet(num_classes=Config.NUM_CLASSES)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ["عصبانی 😡", "انزجار 🤢", "ترس 😱", "خوشحال 😄", "غمگین 😢", "متعجب 😲", "عادی 😐"]

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_with_my_model(img):
    if img is None:
        return "عکس را وارد کنید", None, None

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "صورتی پیدا نشد", None, img


    (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
    face = gray[y:y+h, x:x+w]
    face_pil = Image.fromarray(face)

    input_tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf = torch.max(probs) * 100
        pred = torch.argmax(probs).item()


    prob_dict = {emotions[i]: float(probs[i]) for i in range(7)}


    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 4)
    marked = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    return prob_dict, marked

with gr.Blocks() as demo:

    with gr.Row():
        inp = gr.Image(label="عکس بده", type="numpy")
        out_img = gr.Image(label="صورت تشخیص‌داده‌شده")

    with gr.Row():
        text = gr.Textbox(label="نتیجه")
        chart = gr.Label(num_top_classes=7)

    gr.Button("تشخیص بده").click(predict_with_my_model, inp, [text, chart, out_img])

demo.launch(share=False, inbrowser=True)