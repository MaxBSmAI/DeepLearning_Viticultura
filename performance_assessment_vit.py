import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from timm.models.vision_transformer import Attention


# -------------------------------------------
# 1.Tamaño del modelo Teacher (ViT)
# -------------------------------------------
def get_model_size_teacher(filepath="vit_grape_disease_model_early_stopping.pth"):
    """
    Muestra el tamaño del archivo del modelo Teacher (Vision Transformer).
    """
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Tamaño del modelo Teacher: {size_mb:.2f} MB")
        return size_mb
    else:
        print(f"No se encontró el archivo del modelo en: {filepath}")
        return None

# -------------------------------------------
# 2.Medición de latencia y throughput
# -------------------------------------------
def measure_latency_throughput_teacher(model_teacher, loader, device, batch_sizes=[1, 2, 8, 32]):
    """
    Mide la latencia promedio y máxima, y el throughput (inferencias por segundo)
    para distintos tamaños de batch.
    """
    model_teacher.eval()
    model_teacher.to(device)
    print("\nEvaluando latencia y throughput del modelo Teacher")

    # Tomamos una imagen real del loader
    one_input = next(iter(loader))[0][0].unsqueeze(0).to(device)

    # Warm-up para estabilizar el rendimiento
    for _ in range(10):
        _ = model_teacher(one_input)

    # Latencia
    latencias = []
    for _ in range(100):
        start = time.time()
        _ = model_teacher(one_input)
        end = time.time()
        latencias.append((end - start) * 1000)  # ms

    avg_latency = sum(latencias) / len(latencias)
    max_latency = max(latencias)

    print(f"Latencia promedio (batch=1): {avg_latency:.2f} ms")
    print(f"Latencia máxima  (batch=1): {max_latency:.2f} ms")

    # Throughput
    print("\n Throughput (inferencias/segundo):")
    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 224, 224).to(device)
        for _ in range(5):
            _ = model_teacher(dummy_input)

        start = time.time()
        _ = model_teacher(dummy_input)
        end = time.time()

        throughput = bs / (end - start)
        print(f"- Batch size {bs:<2}: {throughput:.2f} inf/seg")

    return avg_latency, max_latency

# -------------------------------------------
# 3.Parchar Attention para capturar attn_probs
# -------------------------------------------
def patch_attention_to_capture_attn():
    """
    Modifica el forward de los bloques Attention para capturar attn_probs
    durante la inferencia del Vision Transformer.
    """
    def new_forward(self, x, attn_mask=None): ## correcion: attn_mask=None
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_probs = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    Attention.forward = new_forward

# -------------------------------------------
# 4.Visualización de mapas de atención
# -------------------------------------------
def plot_attention_grid_teacher(model_teacher, dataset, device, indices=range(8), cols=4):
    """
    Visualiza un grid con imágenes y sus mapas de atención superpuestos.
    """
    model_teacher.eval()
    rows = 2
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    # Parámetros de desnormalización
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Obtener nombres de las clases
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
        class_names = dataset.dataset.classes
    else:
        class_names = dataset.classes

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model_teacher(input_tensor)

        attn = model_teacher.blocks[-1].attn.attn_probs
        if attn is None:
            print("attn_probs no encontrado. Verifica monkey patch.")
            return

        attn = attn[0].mean(0)  # Media sobre los heads
        cls_attn = attn[0, 1:]  # Atención del token CLS sobre los patches
        grid_size = int(cls_attn.shape[0] ** 0.5)
        cls_attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
        cls_attn_map = cv2.resize(cls_attn_map, (224, 224))
        cls_attn_map = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min())

        # Imagen desnormalizada
        img_np = image.numpy()
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np * 255).astype(np.uint8)

        # Generar heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        axs[i].imshow(superimposed)
        axs[i].set_title(f"Clase: {class_names[label]}", fontsize=12)
        axs[i].axis("off")

    # Quitar ejes sobrantes
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()