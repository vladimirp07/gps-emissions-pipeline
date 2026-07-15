# Auditoría de emisiones

## Pruebas ejecutadas

- Lookup exacta, speed bins, road type conocido/desconocido e imputación faltante.
- Tasas no negativas, suma de subsegmentos y ecuación `g/km × km = g`.
- CO2, CO2e, CO, HC, NOx, PM10 y PM2.5.

## Estado

PASS WITH WARNINGS.

## CRITICAL

Ninguno bajo el supuesto operativo aprobado.

## WARNING

- La lookup no incluye metadata de unidad. Producción asume provisionalmente `g/km`; debe confirmarse contra MOVES original.

## NOTE

- Road types desconocidos usan Road=5 y quedan marcados.

## Cambios aplicados

- Unidad explícita, aliases CO2e/PM2.5 y estados `exact`, imputado o faltante.
- Entradas inválidas fallan claramente.

## Pendientes

- Confirmar unidad fuente y documentar generación del Parquet.

## Recomendación para producción

Operar controladamente y auditar `emission_lookup_status`.

## Estado del módulo

- Estado: PASS WITH WARNINGS
- Bloquea producción: No
- Acción recomendada: confirmar g/km antes de publicación científica definitiva.

