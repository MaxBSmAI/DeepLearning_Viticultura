##############################################################################################
#                                      CARGAR DATASET
##############################################################################################

# Importación de módulos estándar y especializados
import os  # Para manejo de rutas y verificación de carpetas
import torch  # Framework principal de deep learning
from torchvision import datasets, transforms  # Utilidades para cargar imágenes y aplicar transformaciones
from torch.utils.data import DataLoader, Subset  # Utilidades para batching y creación de subconjuntos
import matplotlib.pyplot as plt  # Visualización de imágenes
from collections import Counter  # Conteo eficiente de clases
from sklearn.model_selection import StratifiedShuffleSplit  # División estratificada del dataset

# Ruta al dataset de imágenes organizado en subcarpetas (una por clase)
DATA_DIR = './Final_Training_Data'

# Verificación de existencia de la carpeta raíz
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"La ruta '{DATA_DIR}' no existe. Asegúrate de tener el dataset en la carpeta correcta.")

# Verificación de que existan subcarpetas que representen clases
subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
if len(subdirs) == 0:
    raise ValueError(f"No se encontraron carpetas de clases dentro de '{DATA_DIR}'.")

# Definición de transformaciones que se aplicarán a todas las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona todas las imágenes a 224x224 (requerido por modelos preentrenados)
    transforms.ToTensor(),          # Convierte imágenes PIL a tensores PyTorch (forma: C x H x W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normaliza usando medias de ImageNet
                         std=[0.229, 0.224, 0.225])   # y desviaciones estándar de ImageNet
])

# Carga el dataset usando ImageFolder (cada subcarpeta se trata como una clase)
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Confirmación de carga correcta
print("Dataset cargado correctamente.")
print(f"Total de imágenes: {len(dataset)}")
print(f"Clases detectadas: {dataset.classes}")

##############################################################################################
#                   Mostrar distribución de clases del dataset completo
##############################################################################################

def print_class_distribution(dataset, class_names, name="dataset completo"):
    """
    Imprime la cantidad de imágenes por clase en el dataset entregado.
    """
    labels = [sample[1] for sample in dataset]  # Extrae todas las etiquetas (clase = índice)
    label_counts = Counter(labels)  # Cuenta ocurrencias por clase
    print(f"\nDistribución de clases en {name}:")
    for i in range(len(class_names)):
        print(f"Clase '{class_names[i]}': {label_counts[i]} imágenes")

# Muestra la distribución del dataset original completo (antes de dividirlo)
print_class_distribution(dataset, dataset.classes)

##############################################################################################
#                       DIVISIÓN ESTRATIFICADA: TRAIN, VAL, TEST
##############################################################################################

# Extrae todas las etiquetas (índices numéricos de clase) del dataset completo
all_labels = [sample[1] for sample in dataset]
all_indices = list(range(len(dataset)))  # Lista de índices de cada imagen

# PRIMERA DIVISIÓN: 70% entrenamiento y 30% temporal (val + test)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in split1.split(all_indices, all_labels):
    pass  # Se obtienen los índices de entrenamiento y temporal

# SEGUNDA DIVISIÓN: divide 30% restante en 50% validación y 50% prueba (es decir, 15% cada uno)
temp_labels = [all_labels[i] for i in temp_idx]
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx_rel, test_idx_rel in split2.split(temp_idx, temp_labels):
    val_idx = [temp_idx[i] for i in val_idx_rel]   # Índices absolutos para validación
    test_idx = [temp_idx[i] for i in test_idx_rel] # Índices absolutos para prueba

# Crea subconjuntos del dataset original usando los índices obtenidos
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Parámetro global para tamaño de batch
BATCH_SIZE = 32

# Creación de DataLoaders con sus respectivos subconjuntos
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Imprime la cantidad de imágenes por subconjunto
print(f"\nDivisión de datos:")
print(f"Entrenamiento: {len(train_dataset)} imágenes")
print(f"Validación:    {len(val_dataset)} imágenes")
print(f"Prueba:        {len(test_dataset)} imágenes")

##############################################################################################
#                       Mostrar distribución de clases por subconjunto
##############################################################################################

def print_subset_distribution(subset, class_names, name):
    """
    Imprime la cantidad de imágenes por clase dentro de un subconjunto (train, val, test).
    """
    labels = [subset[i][1] for i in range(len(subset))]  # Extrae etiquetas del subset
    label_counts = Counter(labels)  # Cuenta cuántas veces aparece cada clase
    print(f"\nDistribución de clases en {name}:")
    for i in range(len(class_names)):
        print(f"Clase '{class_names[i]}': {label_counts[i]} imágenes")

# Mostrar distribución por clase en cada partición
print_subset_distribution(train_dataset, dataset.classes, "entrenamiento")
print_subset_distribution(val_dataset, dataset.classes, "validación")
print_subset_distribution(test_dataset, dataset.classes, "prueba")

##############################################################################################
#                       DIVISIÓN ESTRATIFICADA: TRAIN, VAL, TEST
##############################################################################################

# Extrae todas las etiquetas (índices numéricos de clase) del dataset completo
all_labels = [sample[1] for sample in dataset]
all_indices = list(range(len(dataset)))  # Lista de índices de cada imagen

# PRIMERA DIVISIÓN: 70% entrenamiento y 30% temporal (val + test)
split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in split1.split(all_indices, all_labels):
    pass  # Se obtienen los índices de entrenamiento y temporal

# SEGUNDA DIVISIÓN: divide 30% restante en 50% validación y 50% prueba (es decir, 15% cada uno)
temp_labels = [all_labels[i] for i in temp_idx]
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx_rel, test_idx_rel in split2.split(temp_idx, temp_labels):
    val_idx = [temp_idx[i] for i in val_idx_rel]   # Índices absolutos para validación
    test_idx = [temp_idx[i] for i in test_idx_rel] # Índices absolutos para prueba

# Crea subconjuntos del dataset original usando los índices obtenidos
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Parámetro global para tamaño de batch
BATCH_SIZE = 8

# Creación de DataLoaders con sus respectivos subconjuntos
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Imprime la cantidad de imágenes por subconjunto
print(f"\nDivisión de datos:")
print(f"Entrenamiento: {len(train_dataset)} imágenes")
print(f"Validación:    {len(val_dataset)} imágenes")
print(f"Prueba:        {len(test_dataset)} imágenes")

##############################################################################################
#                       Mostrar distribución de clases por subconjunto
##############################################################################################

def print_subset_distribution(subset, class_names, name):
    """
    Imprime la cantidad de imágenes por clase dentro de un subconjunto (train, val, test).
    """
    labels = [subset[i][1] for i in range(len(subset))]  # Extrae etiquetas del subset
    label_counts = Counter(labels)  # Cuenta cuántas veces aparece cada clase
    print(f"\nDistribución de clases en {name}:")
    for i in range(len(class_names)):
        print(f"Clase '{class_names[i]}': {label_counts[i]} imágenes")

# Mostrar distribución por clase en cada partición
print_subset_distribution(train_dataset, dataset.classes, "entrenamiento")
print_subset_distribution(val_dataset, dataset.classes, "validación")
print_subset_distribution(test_dataset, dataset.classes, "prueba")

##############################################################################################
#                               VISUALIZACIÓN DE IMÁGENES
##############################################################################################

def show_images(loader, class_names):
    """
    Visualiza un batch de imágenes con sus respectivas etiquetas desnormalizadas.
    """
    images, labels = next(iter(loader))  # Extrae el primer batch del loader
    images = images[:BATCH_SIZE]
    labels = labels[:BATCH_SIZE]

    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0)  # De (C, H, W) a (H, W, C) para matplotlib
        # Desnormaliza la imagen (revirtiendo la transformación)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)  # Asegura que los valores estén en [0, 1] para visualizar correctamente

        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]])  # Muestra la clase correspondiente
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Llamado a la función para visualizar imágenes del loader de entrenamiento
show_images(train_loader, dataset.classes)
