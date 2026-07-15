# Observaciones técnicas no bloqueantes

## Ruteo

- WARNING: la calidad de snapping legacy es cualitativa, no una distancia persistida.
- NOTE: añadir contadores agregados de nodos inalcanzables y rollback.

## Clasificación modal

- WARNING: serialización pickle exige scikit-learn 1.5.2.
- NOTE: monitorear Carro→Bus y drift sin autoentrenamiento.

## Emisiones

- WARNING: `g/km` es un supuesto provisional aprobado, no metadata original.
- NOTE: versionar el proceso que genera la lookup y su diccionario de unidades.

## Repositorio

- Los archivos >50 MB de datos, grafos y outputs no deben entrar a Git normal. Si deben versionarse, usar Git LFS o almacenamiento de artefactos.
- No se recomienda LFS para los modelos oficiales actuales (≈1.2 MB y ≈0.5 MB) ni la lookup (≈0.6 MB).

## Estado del módulo

- Estado: PASS WITH WARNINGS
- Bloquea producción: No
- Acción recomendada: resolver warnings en una release posterior sin cambiar pipeline v4.

