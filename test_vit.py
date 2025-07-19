import os
import torch
import timm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# -------------------------------
# Configuración del dispositivo
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# -------------------------------
# Definición de clases manualmente (como en entrenamiento)
# -------------------------------
class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

# -------------------------------
# Carga del modelo Vision Transformer
# -------------------------------
model_vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
model_path = 'vit_grape_disease_model_early_stopping.pth'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

model_vit.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model_vit = model_vit.to(device)
model_vit.eval()
print("Modelo ViT cargado exitosamente.")

# -------------------------------
# Transformaciones como en entrenamiento
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Función para predecir y mostrar imagen
# -------------------------------
def predict_and_show(model, image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    # Mostrar imagen con título
    plt.imshow(np.asarray(image))
    plt.title(f"{class_names[pred_idx]} ({confidence:.2f}%)")
    plt.axis("off")

# -------------------------------
# Clasificar y mostrar 4 imágenes
# -------------------------------
image_dir = r'C:\Users\mburg\OneDrive - Doctorado en Inteligencia Artificial\Proyectos\Arquitecturas de DL\modelos_VID\Dataset de prueba'

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"No se encontró el directorio: {image_dir}")

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) < 4:
    raise ValueError("Se requieren al menos 4 imágenes para la visualización.")

# Mostrar las 4 primeras imágenes con predicción
plt.figure(figsize=(12, 6))
for i, img_file in enumerate(image_files[:4]):
    plt.subplot(1, 4, i + 1)
    predict_and_show(model_vit, img_file, class_names)

plt.tight_layout()
plt.show()
