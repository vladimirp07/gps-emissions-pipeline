# Auditoría de ruteo

## Pruebas ejecutadas

- Contrato de columnas, IDs y timestamps.
- Distancias no negativas, velocidades finitas y límite físico.
- Suma de distancias, duplicados de subsegmentos y continuidad nodo final/inicial.
- Caso difícil con rollback/fallo explícito.

## Estado

PASS WITH WARNINGS.

## CRITICAL

Ninguno abierto.

## WARNING

- La distancia exacta de snapping no se persistía en el router legacy; se declara como no disponible.
- iGraph puede reportar nodos inalcanzables; el estado queda expuesto mediante `routing_status` y `flag_auditoria`.

## NOTE

- Medir snapping seleccionado por subsegmento en una versión futura.

## Cambios aplicados

- Contrato común y normalización para routers activos.
- `physical_trip_id`, `network_hypothesis`, `duration_s`, `routing_status` y estado de snapping.

## Pendientes

- Añadir una métrica numérica de continuidad topológica a observabilidad.

## Recomendación para producción

Usar con logs de fallos y no interpretar `snap_distance_m=NaN` como cero.

## Estado del módulo

- Estado: PASS WITH WARNINGS
- Bloquea producción: No
- Acción recomendada: monitorear fallbacks e inalcanzables.

