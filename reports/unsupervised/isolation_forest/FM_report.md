# Reporte de Evaluación (isolation_forest): FM

## Métricas Clave
- **Precisión (Precision)**: 0.9364
- **Recall**: 1.0000
- **F1-Score**: 0.9671
- **Exactitud (Accuracy)**: 0.9660

## Resumen de la Clasificación
| Clase | Precisión | Recall | F1-Score | Soporte |
|:---|:---:|:---:|:---:|:---:|
| **Normal** | 1.00 | 0.93 | 0.96 | 515 |
| **Anomalía** | 0.94 | 1.00 | 0.97 | 515 |
| **Macro Promedio** | 0.97 | 0.97 | 0.97 | 1030 |
| **Promedio Ponderado** | 0.97 | 0.97 | 0.97 | 1030 |

## Matriz de Confusión
| | Predicción: Normal (0) | Predicción: Anomalía (1) |
|---|:---:|:---:|
| **Real: Normal (0)** | 480 | 35 |
| **Real: Anomalía (1)** | 0 | 515 |

## Detalles
- **Número de Anomalías Detectadas**: 550
- **Número de Anomalías Reales**: 515.0

