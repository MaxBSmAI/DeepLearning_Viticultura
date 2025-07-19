# VISION TRANSFORMER
# Implementación de Vision Transformer con Early Stopping y mejoras

import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np

from dataset_vinegrape import train_loader, val_loader, test_loader

# -------------------------------
# 0. Reproducibilidad y Dispositivo
# -------------------------------
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo en uso: {device}")

# Verificación de cargadores
assert train_loader is not None and val_loader is not None and test_loader is not None, \
    "Uno de los cargadores de datos es None. Revisa 'dataset_vinegrape.py'"

# -------------------------------
# 1. Configuración del Modelo
# -------------------------------

# Crear modelo ViT preentrenado y adaptado a 4 clases
model_vit = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=4,
    drop_rate=0.2
).to(device)

# Verifica dimensiones de entrada/salida
with torch.no_grad():
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    sample_output = model_vit(sample_input)
    print(f"Salida del modelo (dim): {sample_output.shape}")  # Debe ser [1, 4]

# Configuración de pérdida, optimizador y scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_vit.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# epoch y patience
EPOCHS = 15
PATIENCE = 3
DELTA = 0.001

# -------------------------------
# 2. Early Stopping
# -------------------------------

class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=DELTA):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping activado.")

# -------------------------------
# 3. Entrenamiento
# -------------------------------

def train_vit_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS, patience=PATIENCE):
    early_stopping = EarlyStopping(patience=patience)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # --- Validación ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)

        scheduler.step()

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train     - Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]*100:.2f}%")
        print(f"Val       - Loss: {val_losses[-1]:.4f}, Acc: {val_accuracies[-1]*100:.2f}%")

        early_stopping(val_losses[-1])
        if early_stopping.early_stop:
            break

    total_time = time.time() - start_time
    print(f"\nTiempo total de entrenamiento: {total_time:.2f} segundos")

    return train_losses, val_losses, train_accuracies, val_accuracies

# -------------------------------
# 4. Evaluación
# -------------------------------

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# -------------------------------
# 5. Entrenamiento y Evaluación
# -------------------------------

train_losses, val_losses, train_accs, val_accs = train_vit_with_early_stopping(
    model_vit, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS, patience=PATIENCE
)

# Test
test_acc = evaluate_model(model_vit, test_loader)
print(f"\nPrecisión en test: {test_acc*100:.2f}%")

# Guardar modelo
torch.save(model_vit.state_dict(), "vit_grape_disease_model_early_stopping.pth")
print("Modelo guardado: vit_grape_disease_model_early_stopping.pth")

# -------------------------------
# 6. Métricas y Performance
# -------------------------------

from plot_metrics_vit import plot_training_metrics, evaluate_metrics_by_set
# -------------------------------
# Imports para métricas y visualización
# -------------------------------
from performance_assessment_vit import (
    get_model_size_teacher,
    measure_latency_throughput_teacher,
    patch_attention_to_capture_attn,
    plot_attention_grid_teacher
)

from performance_assessment_vit import (
    get_model_size_teacher,
    measure_latency_throughput_teacher,
    patch_attention_to_capture_attn,
    plot_attention_grid_teacher
)

dataset_classes = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

plot_training_metrics(train_losses, val_losses, train_accs, val_accs)

evaluate_metrics_by_set(model_vit, train_loader, dataset_classes, device, set_name="Entrenamiento")
evaluate_metrics_by_set(model_vit, val_loader, dataset_classes, device, set_name="Validación")
evaluate_metrics_by_set(model_vit, test_loader, dataset_classes, device, set_name="Prueba")

get_model_size_teacher("vit_grape_disease_model_early_stopping.pth")
measure_latency_throughput_teacher(model_vit, test_loader, device)
patch_attention_to_capture_attn()
plot_attention_grid_teacher(model_vit, test_loader.dataset, device, indices=range(8), cols=4)
