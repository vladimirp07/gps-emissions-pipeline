# Auditoría de clasificación modal

## Pruebas ejecutadas

- Carga y selección de `hybrid`, `random_forest` y `bayes`.
- Contratos 16/52/25, umbral Bus 0.50 y fallo por variable/modelo faltante.
- Probabilidades válidas, guardrail, repetibilidad y Raw/L1/L2/L3 agrupados.

## Estado

PASS.

## CRITICAL

Ninguno.

## WARNING

- Los PKL de scikit-learn no son portables entre versiones incompatibles.

## NOTE

- Mantener el RF anterior como rollback hasta acumular monitoreo del híbrido.

## Cambios aplicados

- Salida contractual con backend, versión, calidad y causa de rechazo.
- Fábrica única controlada por configuración/entorno.

## Pendientes

- Automatizar monitoreo de drift sin reentrenar en inferencia.

## Recomendación para producción

Usar `hybrid`; conservar RF y Bayes disponibles explícitamente.

## Estado del módulo

- Estado: PASS
- Bloquea producción: No
- Acción recomendada: fijar scikit-learn 1.5.2.

