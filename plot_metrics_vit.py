import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import torch


# Genera gráficos de pérdida y precisión para analizar Overfitting y Underfitting. 

def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Genera gráficos de pérdida y precisión para analizar Overfitting y Underfitting.
    """
    # Gráfico de Pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Pérdida de Entrenamiento', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Pérdida de Validación', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico de Precisión
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Precisión de Entrenamiento', marker='o')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Precisión de Validación', marker='o')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.show()


# Calcula y muestra matriz de confusión, reporte de clasificación y exactitud para un conjunto.

def evaluate_metrics_by_set(model, loader, dataset_classes, device, set_name="Conjunto"):
    """
    Calcula y muestra matriz de confusión, reporte de clasificación y exactitud para un conjunto.
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

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nMatriz de Confusión para {set_name}:")
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset_classes, yticklabels=dataset_classes)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {set_name}")
    plt.show()

    print(f"\nReporte de Clasificación para {set_name}:")
    print(classification_report(all_labels, all_preds, target_names=dataset_classes))

    acc = accuracy_score(all_labels, all_preds)
    print(f"Exactitud General para {set_name}: {acc:.4f}")
