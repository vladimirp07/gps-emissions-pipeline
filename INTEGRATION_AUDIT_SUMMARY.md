# Auditoría de integración

## Pruebas ejecutadas

- Smoke end-to-end para Carro, Bus, Metro y Caminar.
- Caso degradado, guardrail y fallo de ruteo explícito.
- Conservación de ID, orden temporal, modo, distancia ruteada y ausencia de duplicados.

## Estado

PASS.

## CRITICAL

Ninguno.

## WARNING

- Los smoke tests usan una lookup mínima determinista; el orquestador completo ya se validó separadamente con la lookup real.

## NOTE

- Conservar outputs mínimos como evidencia de release.

## Cambios aplicados

- Suite rápida independiente de notebooks en `tests/integration/`.
- Fallos y resultados escritos en `outputs/production_smoke_tests/`.

## Pendientes

- Incorporar estos smoke tests al CI cuando exista runner con datos privados.

## Recomendación para producción

Listo para operación controlada.

## Estado del módulo

- Estado: PASS
- Bloquea producción: No
- Acción recomendada: ejecutar pytest antes de cada despliegue.

