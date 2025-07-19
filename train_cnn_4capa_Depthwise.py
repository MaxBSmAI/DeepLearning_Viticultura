# -------------------------------
# Entrenamiento del modelo StudentCNN con destilación de conocimiento
# -------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import timm  # Para cargar el modelo teacher ViT
from dataset_vinegrape import train_loader, val_loader, test_loader

# epoch y patience
EPOCHS = 15
PATIENCE = 3
DELTA = 0.001

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# -------------------------------
# 1. Definición del modelo StudentCNN (con Depthwise Separable Convolutions)
# -------------------------------

class StudentCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(StudentCNN, self).__init__()

        # Se definen las capas convolucionales usando Depthwise + Pointwise
        self.features_student = nn.Sequential(

            # # Capa convolucional 1
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),        # Depthwise
            nn.Conv2d(3, 32, kernel_size=1),                            # Pointwise
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ 224 → 112

            # # Capa convolucional 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),     # Depthwise
            nn.Conv2d(32, 64, kernel_size=1),                           # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ 112 → 56

            # # Capa convolucional 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),     # Depthwise
            nn.Conv2d(64, 128, kernel_size=1),                          # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ 56 → 28

            # # Capa convolucional 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.Conv2d(128, 256, kernel_size=1),                         # Pointwise
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # ↓ 28 → 14
        )

        # Cálculo automático del número de características para la capa lineal
        dummy_input = torch.zeros(1, 3, 224, 224)  # Simula una imagen
        dummy_output = self.features_student(dummy_input)
        num_features = dummy_output.view(1, -1).size(1)

        # Clasificador completamente conectado
        self.classifier_student = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features_student(x)
        x = self.classifier_student(x)
        return x
    
# -------------------------------
# 2. Early stopping
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
# 3. Distillation loss
# -------------------------------

def distillation_loss_student(student_logits, teacher_logits, true_labels, T=2.0, alpha=0.5):
    loss_ce = F.cross_entropy(student_logits, true_labels)
    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    loss_kd = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
    return alpha * loss_kd + (1 - alpha) * loss_ce

# -------------------------------
# 4. Entrenamiento con destilación
# -------------------------------

def train_student_with_early_stopping(student_model, teacher_model, train_loader, val_loader,
                                      T=2.0, alpha=0.5, lr=1e-4, epochs=EPOCHS, patience=PATIENCE):
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopping = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            student_logits = student_model(inputs)
            loss = distillation_loss_student(student_logits, teacher_logits, labels, T, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(student_logits, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct / total)

        # Validación
        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student_model(inputs)
                teacher_outputs = teacher_model(inputs)
                loss_val = distillation_loss_student(outputs, teacher_outputs, labels, T, alpha)

                val_loss += loss_val.item()
                _, val_preds = torch.max(outputs, 1)
                val_correct += torch.sum(val_preds == labels).item()
                val_total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_correct / val_total)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}, Acc: {val_accs[-1]:.4f}")

        torch.save(student_model.state_dict(), "student_model_4capa_depthwise.pth")

        early_stopping(val_losses[-1])
        if early_stopping.early_stop:
            break

    total_time = time.time() - start_time
    print(f"Tiempo total de entrenamiento: {total_time:.2f} segundos")

    return train_losses, val_losses, train_accs, val_accs

# -------------------------------
# 5. Evaluación
# -------------------------------

def evaluate_model_student(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    return correct / total

# -------------------------------
# 6. Main
# -------------------------------

if __name__ == "__main__":
    # Cargar modelo teacher (ViT) y mover al dispositivo
    model_teacher = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
    model_teacher.load_state_dict(torch.load("vit_grape_disease_model_early_stopping.pth", map_location=device))
    model_teacher = model_teacher.to(device)
    model_teacher.eval()

    # Instanciar modelo student
    student_model = StudentCNN(num_classes=4).to(device)

    # Entrenar
    train_losses, val_losses, train_accs, val_accs = train_student_with_early_stopping(
        student_model, model_teacher, train_loader, val_loader,
        T=2.0, alpha=0.5, lr=1e-4, epochs=EPOCHS, patience=PATIENCE
    )

    # Evaluar en test
    test_acc = evaluate_model_student(student_model, test_loader)
    print(f"Precisión del modelo Student en test: {test_acc:.4f}")

    # Mostrar tamaño del modelo
    if os.path.exists("student_model_4capa_depthwise.pth"):
        size_mb = os.path.getsize("student_model_4capa_depthwise.pth") / (1024 * 1024)
        print(f"Tamaño del modelo Student: {size_mb:.2f} MB")

    # Métricas y performance
    from plot_metrics_cnn import plot_student_training_metrics, evaluate_student_metrics_by_set
    from performance_assessment_cnn import performance_assessment_student, plot_attention_grid_student

    plot_student_training_metrics(train_losses, val_losses, train_accs, val_accs)

    class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']
    evaluate_student_metrics_by_set(student_model, train_loader, class_names, device, set_name="Student - Entrenamiento")
    evaluate_student_metrics_by_set(student_model, val_loader, class_names, device, set_name="Student - Validación")
    evaluate_student_metrics_by_set(student_model, test_loader, class_names, device, set_name="Student - Prueba")

    performance_assessment_student(student_model, val_loader, model_path_student="student_model_4capa_depthwise.pth")
    plot_attention_grid_student(student_model, test_loader.dataset, num_per_class=2, cols=4)
