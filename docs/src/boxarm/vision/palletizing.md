# src/boxarm/vision/palletizing.py

`GridCounter` es la ?nica autoridad de conteo. Mantiene ocupaci?n persistente
por `(celda,nivel)`, gate de brazo/movimiento, una caja por ciclo y promoci?n
por soporte trabado de al menos dos cajas inferiores.

No usa IDs temporales. Una caja confirmada conserva posici?n, nivel y footprint.

Ver el contrato completo en
[`docs/palletizing_counting.md`](../../../palletizing_counting.md).
