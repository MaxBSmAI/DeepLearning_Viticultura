# -------------------------------
# plot_metrics_cnn.py
# Gráficos y métricas para el modelo StudentCNN
# -------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import torch

# -------------------------------
# Función: graficar pérdida y precisión por época
# -------------------------------
def plot_student_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Genera gráficos de pérdida y precisión para el modelo StudentCNN
    durante el entrenamiento y validación.
    """
    # Gráfico de pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Pérdida Entrenamiento', marker='o')
    plt.plot(val_losses, label='Pérdida Validación', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida del Modelo StudentCNN')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico de precisión
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Precisión Entrenamiento', marker='o')
    plt.plot(val_accuracies, label='Precisión Validación', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión del Modelo StudentCNN')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Función: evaluar y mostrar métricas por conjunto
# -------------------------------
def evaluate_student_metrics_by_set(model, loader, dataset_classes, device, set_name="Conjunto"):
    """
    Calcula y muestra matriz de confusión, reporte de clasificación y exactitud general
    para el conjunto especificado.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n Matriz de Confusión para {set_name}:")
    print(cm)

    # Visualizar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset_classes, yticklabels=dataset_classes)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {set_name}")
    plt.show()

    # Reporte de clasificación
    print(f"\n Reporte de Clasificación para {set_name}:")
    print(classification_report(all_labels, all_preds, target_names=dataset_classes))

    # Exactitud general
    acc = accuracy_score(all_labels, all_preds)
    print(f"Exactitud General para {set_name}: {acc:.4f}")
