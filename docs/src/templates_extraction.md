# Extraccion general de cajas y templates

Documento unico para las 12 clases. Registra las cajas extraidas por el modelo
y las reglas usadas para construir templates normalizados.

## Clases previstas

| Clase | Estado | Regla |
|---|---|---|
| `bag_10` | Activa | A extraido; B rotado 180 grados |
| `bag_100` | Preparada | Pendiente de referencia |
| `bag_20` | Preparada | Pendiente de referencia |
| `bag_200` | Preparada | Pendiente de referencia |
| `bag_50` | Preparada | Pendiente de referencia |
| `bag_500` | Preparada | Pendiente de referencia |
| `coin_roll_10` | Activa | A extraido; B rotado 180 grados |
| `coin_roll_100` | Activa | A/B definidos |
| `coin_roll_20` | Preparada | Pendiente de referencia |
| `coin_roll_200` | Activa | A extraido; B escalado segun coin_roll_100 |
| `coin_roll_50` | Activa | A extraido; B rotado 90 grados |
| `coin_roll_500` | Preparada | Pendiente de referencia |

## Metodo comun

- Modelo: `models/coin-box/coin-box.pt`.
- Las detecciones se convierten a centros, dimensiones y orientacion `H`/`V`.
- Las coordenadas se normalizan entre 0 y 1 sobre el footprint de la paleta.
- Las variantes rotadas usan `rotate_pattern`.
- Las variantes de distinto tamano usan `scale_pattern`.

## Datos disponibles

### `bag_10`

Fuente: `img_ref/bag_10.jpg`. Se extrajeron 7 cajas unicas.

- `BAG_10_PATTERN_A`: cajas extraidas.
- `BAG_10_PATTERN_B`: rotacion 180 grados de A.

### `coin_roll_10`

Fuente: `img_ref/coin_roll_10.jpg`. Se extrajeron 25 cajas unicas.

- `COIN_ROLL_10_PATTERN_A`: cajas extraidas.
- `COIN_ROLL_10_PATTERN_B`: rotacion 180 grados de A.

### `coin_roll_50`

Fuentes: `img_ref/coin_roll_50.jpg` y `img_ref/coin_roll_50_1.jpg`.
Se extrajeron 18 cajas por referencia; la segunda es la misma topologia rotada.

- `COIN_ROLL_50_PATTERN_A`: cajas extraidas.
- `COIN_ROLL_50_PATTERN_B`: rotacion 90 grados de A.

### `coin_roll_100`

Contiene 15 cajas por nivel y sus dos patrones historicos:
`COIN_ROLL_100_PATTERN_A` y `COIN_ROLL_100_PATTERN_B`.

### `coin_roll_200`

Fuente: `img_ref/coin_roll_200.jpg`. Se extrajeron 15 cajas.
La topologia coincide con `coin_roll_100`, pero las cajas son mayores.

- `COIN_ROLL_200_PATTERN_A`: cajas extraidas de la referencia.
- `COIN_ROLL_200_PATTERN_B`: segundo nivel escalado segun `coin_roll_100`.

## Pendientes

Las siete clases preparadas tienen su modulo `.py`, pero quedan pendientes de
una referencia validada antes de activar sus templates.
