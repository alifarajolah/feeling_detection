# test.py
import torch
from torchvision import transforms
from PIL import Image
from models.feeling_detection_net import PersianFERNet
from config import Config

idx_to_class = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

model = PersianFERNet(num_classes=Config.NUM_CLASSES)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.MedianBlur(3),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_emotion(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return idx_to_class[pred]
