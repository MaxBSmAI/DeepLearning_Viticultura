# -------------------------------
# performance_assessment_cnn.py
# Evaluación del rendimiento del modelo StudentCNN
# -------------------------------

# Importación de librerías necesarias
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------
# Función: evaluación de tamaño, latencia y throughput
# -------------------------------
def performance_assessment_student(model_student, loader_student, model_path_student="student_model_last.pth", batch_sizes=[1, 2, 8, 32]):
    """
    Evalúa el tamaño del modelo StudentCNN, su latencia y throughput para distintos tamaños de batch.
    """

    model_student.eval()  # Poner el modelo en modo evaluación
    device = next(model_student.parameters()).device  # Detectar el dispositivo en que está el modelo (CPU o GPU)
    model_student.to(device)  # Enviar modelo al dispositivo adecuado

    print("\nEvaluación de Performance del modelo StudentCNN")

    # Tamaño del archivo del modelo (en MB)
    if os.path.exists(model_path_student):
        size_bytes = os.path.getsize(model_path_student)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Tamaño del modelo Student: {size_mb:.2f} MB")
    else:
        print(f"El archivo {model_path_student} no fue encontrado.")

    # Preparación para la medición de latencia (usando el primer batch del loader)
    one_input = next(iter(loader_student))[0][0].unsqueeze(0).to(device)  # Extrae 1 imagen, le agrega batch dim
    latencias = []

    # Warm-up para evitar medir tiempos con compilación inicial o caching
    for _ in range(10):
        _ = model_student(one_input)

    # Medición real de latencia (100 ejecuciones)
    for _ in range(100):
        start = time.time()
        _ = model_student(one_input)
        end = time.time()
        latencias.append((end - start) * 1000)  # Convertido a milisegundos

    print(f"Latencia promedio Student (batch=1): {np.mean(latencias):.2f} ms")
    print(f"Latencia máxima Student  (batch=1): {np.max(latencias):.2f} ms")

    # Throughput: cantidad de inferencias por segundo para distintos tamaños de batch
    print("\nThroughput del modelo Student (inferencias por segundo):")
    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 224, 224).to(device)  # Entrada ficticia simulando batch real

        # Warm-up
        for _ in range(5):
            _ = model_student(dummy_input)

        start = time.time()
        _ = model_student(dummy_input)
        end = time.time()

        duration = end - start
        throughput = bs / duration
        print(f"- Batch size {bs:<2}: {throughput:.2f} inf/seg")

# -------------------------------
# Función: visualización Grad-CAM
# -------------------------------
def plot_attention_grid_student(model_student, dataset_student, num_per_class=2, cols=4):
    """
    Visualiza un grid de mapas de atención Grad-CAM para distintas clases.
    Se muestran `num_per_class` imágenes por clase.
    """

    model_student.eval()
    device = next(model_student.parameters()).device

    # Identificar índices de imágenes para cada clase (hasta `num_per_class` por clase)
    class_to_indices = {i: [] for i in range(len(dataset_student.dataset.classes))}
    for idx in range(len(dataset_student)):
        _, label = dataset_student[idx]
        if len(class_to_indices[label]) < num_per_class:
            class_to_indices[label].append(idx)

    # Reunir todos los índices seleccionados en una lista plana
    selected_indices = [idx for indices in class_to_indices.values() for idx in indices]
    rows = (len(selected_indices) + cols - 1) // cols  # Calcular cantidad de filas para grid

    # Crear figura con subplots
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()

    # -------------------------------
    # DESNORMALIZACIÓN automática
    # -------------------------------
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    # ------------------------------------------
    # Selección DINÁMICA de capa convolucional
    # ------------------------------------------
    conv_layers = [module for module in model_student.modules() if isinstance(module, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No se encontraron capas convolucionales en el modelo.")
    target_layers = [conv_layers[-1]]  # Última capa convolucional encontrada automáticamente

    # Inicializar GradCAM con la capa seleccionada
    cam = GradCAM(model=model_student, target_layers=target_layers)

    # Iterar sobre los índices seleccionados y generar visualizaciones
    for i, idx in enumerate(selected_indices):
        image, label = dataset_student[idx]
        input_tensor = image.unsqueeze(0).to(device)

        # Ejecutar GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(label)])[0]

        # Desnormalizar imagen para visualizar
        image_vis = (image.unsqueeze(0).to(device) * std + mean).clamp(0, 1).squeeze().cpu().numpy()
        image_vis = np.transpose(image_vis, (1, 2, 0))  # Convertir de [C, H, W] a [H, W, C] para OpenCV

        # Superponer mapa de calor sobre la imagen
        visualization = show_cam_on_image(image_vis, grayscale_cam, use_rgb=True)

        # Renderizar subplot
        axs[i].imshow(visualization)
        axs[i].set_title(f"Clase: {dataset_student.dataset.classes[label]}")
        axs[i].axis('off')

    # Ocultar subplots vacíos si hay
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
