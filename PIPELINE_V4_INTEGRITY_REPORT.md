# Integridad de pipeline_v4_production

Estado: **READY FOR CONTROLLED PRODUCTION**.

## Resultado ejecutivo

- Pytest: 26 aprobadas, 0 fallidas.
- Notebook oficial desde kernel limpio: PASS; matriz OOF visible y PNG generado.
- End-to-end: PASS para Carro, Bus, Metro y Caminar.
- Configuración de hybrid/RF/Bayes: PASS.
- Ruteo: [PASS WITH WARNINGS](ROUTING_AUDIT_SUMMARY.md).
- Clasificación: [PASS](MODAL_CLASSIFICATION_AUDIT_SUMMARY.md).
- Emisiones: [PASS WITH WARNINGS](EMISSIONS_AUDIT_SUMMARY.md).
- Integración: [PASS](INTEGRATION_AUDIT_SUMMARY.md).

## Cambios aplicados

- Contratos centrales, estados explícitos de fallo/lookup y unidad provisional g/km.
- Pruebas reorganizadas; prueba ampliada rechazada archivada con justificación histórica.
- `.gitignore` dejó de excluir extensiones de datos globalmente.

## Riesgos pendientes

- Confirmar unidades de la lookup MOVES.
- Persistir distancia numérica de snapping en una futura versión del router.

## Estado del módulo

- Estado: READY FOR CONTROLLED PRODUCTION
- Bloquea producción: No
- Acción recomendada: etiquetar localmente `pipeline_v4_production` y no hacer push aún.

