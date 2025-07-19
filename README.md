# Clasificación de Enfermedades en Hojas de Vid con Deep Learning y Distillation

Este proyecto implementa una solución basada en **Deep Learning** para la **detección y clasificación de enfermedades en hojas de vid**, utilizando una arquitectura **Vision Transformer (ViT)** como modelo Teacher y múltiples versiones de **Redes Convolucionales Livianas (StudentCNN)** con y sin **Depthwise Separable Convolutions**, entrenadas mediante **Knowledge Distillation**.

---

## Motivación

La agricultura de precisión requiere sistemas automáticos, eficientes y en tiempo real para identificar enfermedades en las plantas. Este proyecto busca **reducir la complejidad computacional** sin comprometer la precisión mediante la **destilación de conocimiento** desde un modelo poderoso (ViT) hacia redes convolucionales más pequeñas, aptas para su despliegue en dispositivos móviles o drones.

---

## Modelos Utilizados

Se implementaron dos tipos principales de modelos:

- **Teacher Model — Vision Transformer (ViT)**  
  Utiliza una arquitectura ViT (`vit_base_patch16_224`) entrenada con Early Stopping. Este modelo actúa como una red experta que transfiere conocimiento a modelos más pequeños.

- **Student Models — Redes Convolucionales (CNN)**  
  Se desarrollaron múltiples arquitecturas CNN de 1 a 4 capas, incluyendo versiones con **convoluciones Depthwise Separable**, para reducir parámetros y acelerar la inferencia. Estos modelos fueron entrenados utilizando **Knowledge Distillation**, aprendiendo de las predicciones del modelo ViT.


---

## Técnicas Aplicadas

- **Transferencia de conocimiento (Knowledge Distillation)**:  
  Técnica donde un modelo grande (Teacher) transfiere su conocimiento a un modelo más pequeño (Student), usando logits suavizados como guía.

- **Early Stopping**:  
  Se utilizó para evitar sobreajuste durante el entrenamiento del modelo Teacher.

- **Depthwise Separable Convolutions**:  
  Implementadas en los modelos Student para reducir el número de parámetros y acelerar la inferencia, manteniendo la capacidad de aprendizaje.

- **Evaluación con Grad-CAM**:  
  Se aplicó visualización de activaciones para interpretar qué regiones de la hoja fueron relevantes para la clasificación.

---

## Impacto en la Industria Vitivinícola

La automatización del diagnóstico de enfermedades en hojas de vid tiene el potencial de:

- **Reducir pérdidas agrícolas** mediante la **detección temprana** de enfermedades como Black Rot, Esca y Leaf Blight.
- **Optimizar el uso de recursos fitosanitarios**, aplicando tratamientos solo en áreas afectadas.
- **Apoyar a pequeños productores** mediante herramientas móviles o integradas en drones que permitan monitoreo constante.
- **Mejorar la trazabilidad y gestión de cultivos** mediante sistemas de clasificación en tiempo real.

Este sistema puede integrarse en dispositivos de bajo consumo energético, generando un impacto directo en el manejo eficiente y sostenible de viñedos.

---

## Estructura del Proyecto

```bash
modelos_VID/
│
├── dataset_vinegrape.py               # Dataset y transformaciones
├── train_teacher_vit.py              # Entrenamiento del modelo ViT (Teacher)
│
├── train_cnn_1capa.py                # StudentCNN con 1 capa
├── train_cnn_2capa.py                # StudentCNN con 2 capas
├── train_cnn_3capa.py                # StudentCNN con 3 capas
├── train_cnn_4capa.py                # StudentCNN con 4 capas
│
├── train_cnn_1capa_depthwise.py      # StudentCNN con 1 capa + Depthwise
├── train_cnn_2capa_depthwise.py      # StudentCNN con 2 capa + Depthwise
├── train_cnn_3capa_depthwise.py      # StudentCNN con 3 capa + Depthwise
├── train_cnn_4capa_depthwise.py      # StudentCNN con 4 capa + Depthwise
│
├── test_cnn.py                       # Clasificación visual con modelo CNN
├── test_vit.py                       # Clasificación visual con modelo ViT
│
├── plot_metrics_vit.py               # Métricas y visualizaciones
├── plot_metrics_cnn.py               # Métricas y visualizaciones
├── performance_assessment_vit.py     # Tamaño, latencia y activaciones (Grad-CAM)
├── performance_assessment_cnn.py     # Tamaño, latencia y activaciones (Grad-CAM)
├── *.pth                             # Archivos de modelos entrenados
└── Dataset de prueba/                # Carpeta con imágenes para test manual
