from __future__ import annotations

from dataclasses import dataclass

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Funciones puras de geometria/numeros del paquete de paletizado.

Sin estado, sin dependencia de `.types` ni de `counter.py`: reciben todos
sus datos por parametro. Es la unica pieza del paquete que puede usarse sin
instanciar GridCounter.

Cada funcion lleva su formula explicita en el docstring (notacion de
docs/palletizing_math.md) ademas de la explicacion en prosa: si hay que
optimizar una cuenta, la formula a optimizar esta aca, no hay que rederivarla
leyendo el codigo."""

from math import log

import cv2
import numpy as np

_INTERVAL_SAMPLES = 5  # posiciones probadas dentro del intervalo factible de un recorte


@dataclass(frozen=True)
class _SupportPolygonAssessment:
    """Resultado geométrico puro del criterio de soporte consolidado.

    ``center_inside`` expresa estabilidad estática local. ``interlocked``
    añade la condición de negocio que la mecánica NO puede deducir sola:
    deben existir al menos dos contactos independientes para declarar una
    caja trabada y no una simple redetección sobre una única caja.
    """

    contact_count: int
    hull_area_ratio: float
    center_distance: float
    center_inside: bool
    degenerate: bool
    interlocked: bool
    shares: tuple[float, ...]


def _interval_samples(center: float, slack: float) -> tuple[float, ...]:
    """Muestrea [center-slack, center+slack]; un solo punto si no hay holgura.

    Formula: para n = _INTERVAL_SAMPLES y paso h = 2*slack / (n-1),
        x_i = center - slack + h*i,   i = 0..n-1
    Si slack <= 0, devuelve (center,) sin evaluar la formula."""
    if slack <= 0.0:
        return (center,)
    step = 2.0 * slack / (_INTERVAL_SAMPLES - 1)
    return tuple(center - slack + step * i for i in range(_INTERVAL_SAMPLES))


def _observed_median(values) -> float:
    """Mediana superior: siempre devuelve una observación, nunca un promedio.

    Formula (docs/palletizing_math.md Definicion 2.1), con x ordenado:
        med(x) = x_((n+1)/2)   si n impar
        med(x) = x_(n/2 + 1)   si n par   (mediana SUPERIOR, no promedio de
                                            las dos centrales)
    Implementado como `sorted(values)[n // 2]`, que es exactamente esa
    definicion para ambas paridades de n."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("la mediana requiere al menos una observación")
    return ordered[len(ordered) // 2]


def build_homography(roi_pts: np.ndarray) -> np.ndarray:
    """3x3 que lleva los 4 vertices de la paleta (orden: arriba-izq,
    arriba-der, abajo-der, abajo-izq -- mismo orden que ROI_PTS_DEFAULT)
    al cuadrado unidad [0,1]^2 (seccion 4).

    Formula (docs/palletizing_math.md Definicion 1.2): H tal que
        H * [roi_pts; 1] = [(0,0), (1,0), (1,1), (0,1); 1]
    resuelto por `cv2.getPerspectiveTransform` (DLT de 4 correspondencias)."""
    src = roi_pts.astype(np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _project(homography: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Formula (docs/palletizing_math.md Definicion 1.2):
        (u, v) = pi(H [x, y, 1]^T),   pi([a, b, c]^T) = (a/c, b/c)"""
    pt = np.array([[[x, y]]], dtype=np.float32)
    u, v = cv2.perspectiveTransform(pt, homography)[0, 0]
    return float(u), float(v)


def _rect_intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Area de interseccion de dos rectangulos (u,v,du,dv) centrados en (u,v).

    Formula: con A=[au-adu/2, au+adu/2]x[av-adv/2, av+adv/2] y B analogo,
        area(A cap B) = max(0, min(au+adu/2, bu+bdu/2) - max(au-adu/2, bu-bdu/2))
                      * max(0, min(av+adv/2, bv+bdv/2) - max(av-adv/2, bv-bdv/2))"""
    au, av, adu, adv = a
    bu, bv, bdu, bdv = b
    iu = max(0.0, min(au + adu / 2.0, bu + bdu / 2.0)
             - max(au - adu / 2.0, bu - bdu / 2.0))
    iv = max(0.0, min(av + adv / 2.0, bv + bdv / 2.0)
             - max(av - adv / 2.0, bv - bdv / 2.0))
    return iu * iv


def _rect_overlap_over_min(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Formula (docs/palletizing_math.md Definicion 3.1, R_min):
        R_min(A, B) = area(A cap B) / min(area(A), area(B))
    con area(A cap B) de `_rect_intersection_area` y denominador con piso
    1e-9 para evitar division por cero en un rectangulo degenerado."""
    return _rect_intersection_area(a, b) / max(
        min(a[2] * a[3], b[2] * b[3]), 1e-9,
    )


def _rect_union_coverage(
    target: tuple[float, float, float, float],
    rects: list[tuple[float, float, float, float]],
    occupancy_grid: int,
) -> float:
    """Fraccion de `target` cubierta por la union de `rects`.

    Formula (docs/palletizing_math.md Teorema 7.2 / Corolario 7.3, version
    rasterizada): con U = union(rects),
        coverage = area(target cap U) / area(target)
    aproximado sobre una grilla n x n (`occupancy_grid`) en vez de resuelto
    con geometria exacta de poligonos, por eso el resultado depende de `n`.

    `occupancy_grid` es `PalletizingConfig.occupancy_grid` -- la resolucion
    del raster. Se recibe por parametro, no por `self._cfg`, porque esta
    funcion es pura y no conoce la config."""
    if not rects:
        return 0.0
    n = occupancy_grid
    mask = np.zeros((n, n), dtype=bool)
    for u, v, du, dv in rects:
        u0, u1 = max(0, int((u - du / 2) * n)), min(n, int(np.ceil((u + du / 2) * n)))
        v0, v1 = max(0, int((v - dv / 2) * n)), min(n, int(np.ceil((v + dv / 2) * n)))
        if u1 > u0 and v1 > v0:
            mask[v0:v1, u0:u1] = True
    u, v, du, dv = target
    u0, u1 = max(0, int((u - du / 2) * n)), min(n, int(np.ceil((u + du / 2) * n)))
    v0, v1 = max(0, int((v - dv / 2) * n)), min(n, int(np.ceil((v + dv / 2) * n)))
    if u1 <= u0 or v1 <= v0:
        return 0.0
    window = mask[v0:v1, u0:u1]
    return float(window.sum()) / float(window.size)


def _intersection_over_min_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """R_min en coordenadas de pixel xyxy (a diferencia de
    `_rect_overlap_over_min`, que trabaja en (u,v,du,dv) normalizado).

    Formula (docs/palletizing_math.md Definicion 3.1):
        R_min(A, B) = area(A cap B) / min(area(A), area(B))
    con area(A), area(B) acotadas por abajo en 1.0 (nunca 0) para que un
    bbox degenerado no produzca division por cero."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
    area_b = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return inter / min(area_a, area_b)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU clasico, usado solo para confirmar candidatas entre frames.

    Formula (docs/palletizing_math.md Definicion 3.1):
        IoU(A, B) = area(A cap B) / area(A cup B)
                  = area(A cap B) / (area(A) + area(B) - area(A cap B))
    Notar IoU <= R_min siempre (Proposicion 3.2 del mismo documento): por
    eso esta funcion NO sirve para matching de fragmentos, solo para enlazar
    dos observaciones consecutivas del mismo tamano."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _footprint_overlap_over_min(
    pos_a: tuple[float, float], size_a: tuple[float, float],
    pos_b: tuple[float, float], size_b: tuple[float, float],
) -> float:
    """Solape de dos footprints en [0,1]^2 (rectangulos centrados en
    `pos`, de sides `size`) dividido por el area del MENOR.

    Formula (docs/palletizing_math.md Definicion 3.1, R_min -- misma cuenta
    que `_rect_overlap_over_min` pero con la firma (pos, size) en vez de
    (u,v,du,dv) empaquetado en una sola tupla):
        R_min(A, B) = area(A cap B) / min(area(A), area(B))

    Se normaliza por el smaller y no por la union (IoU) a proposito: el
    duplicado tipico nace de un bbox recortado por oclusion, o sea mas
    chico que la caja real, y la IoU lo castigaria por esa diferencia
    de tamano justo cuando mas se necesita detectarlo."""
    (au, av), (aw, ah) = pos_a, size_a
    (bu, bv), (bw, bh) = pos_b, size_b
    inter_u = max(0.0, min(au + aw / 2.0, bu + bw / 2.0) - max(au - aw / 2.0, bu - bw / 2.0))
    inter_v = max(0.0, min(av + ah / 2.0, bv + bh / 2.0) - max(av - ah / 2.0, bv - bh / 2.0))
    min_area = min(aw * ah, bw * bh)
    if min_area <= 0.0:
        return 0.0
    return (inter_u * inter_v) / min_area


def _footprint_containment(
    detection_pos: tuple[float, float],
    detection_size: tuple[float, float],
    confirmed_pos: tuple[float, float],
    confirmed_size: tuple[float, float],
) -> float:
    """Fraccion del footprint de la DETECCION dentro del confirmado.

    La direccion importa: un fragmento ocluido puede ser menor y quedar
    contenido; dos rectangulos cruzados no representan la misma identidad
    aunque overlap/min sea alto.
    """
    return _rect_intersection_area(
        (*detection_pos, *detection_size),
        (*confirmed_pos, *confirmed_size),
    ) / max(detection_size[0] * detection_size[1], 1e-9)


def _split_detection(det) -> tuple[tuple[int, int, int, int], float | None, str]:
    """Acepta bbox legacy `(x1,y1,x2,y2)` o `(x1,y1,x2,y2,conf)` o
    `(x1,y1,x2,y2,conf,cls_name)`; desempaca la tupla cruda del detector.

    Formula: identidad, sin calculo -- separa bbox de confianza y clase opcional:
        (x1,y1,x2,y2,conf,cls) -> ((int(x1..y2)), float(conf), cls)
        (x1,y1,x2,y2,conf)     -> ((int(x1..y2)), float(conf), "")
        (x1,y1,x2,y2)          -> ((int(x1..y2)), None, "")"""
    if len(det) == 6:
        x1, y1, x2, y2 = map(int, det[:4])
        return (x1, y1, x2, y2), float(det[4]), str(det[5])
    if len(det) == 5:
        x1, y1, x2, y2 = map(int, det[:4])
        return (x1, y1, x2, y2), float(det[4]), ""
    x1, y1, x2, y2 = map(int, det[:4])
    return (x1, y1, x2, y2), None, ""


def _bootstrap_bbox_text(rect: tuple[float, float, float, float]) -> str:
    """BBox normalizado como centro, tamano y esquinas -- solo para logs.

    Formula: de (u,v,du,dv) centrado en (u,v) a esquinas xyxy:
        x1 = u - du/2,  y1 = v - dv/2,  x2 = u + du/2,  y2 = v + dv/2"""
    u, v, du, dv = rect
    return (
        f"centro=({u:.3f},{v:.3f}) tam={du:.3f}x{dv:.3f} "
        f"xyxy=({u - du / 2.0:.3f},{v - dv / 2.0:.3f},"
        f"{u + du / 2.0:.3f},{v + dv / 2.0:.3f})"
    )


def _center_distance_over_min_side(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Distancia entre centros normalizada por el lado mas chico de los dos
    bboxes -- respaldo del matching principal por `R_min`, no criterio
    primario.

    Formula: con centros c_a, c_b y min_side = min(lados de A y de B, piso 1),
        d(A, B) = ||c_a - c_b||_2 / min_side"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    min_side = max(1.0, min(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1))
    return float(np.hypot(acx - bcx, acy - bcy) / min_side)


# -- Geometria basica de bounding-box XYXY ----------------------------------

def _bbox_center_and_size(
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centroide y dimensiones de un bbox XYXY.

    Formula:
        cx = (x1 + x2) / 2,   cy = (y1 + y2) / 2
        w  = x2 - x1,         h  = y2 - y1
    Devuelve ((cx, cy), (w, h))."""
    x1, y1, x2, y2 = bbox
    return (
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (float(x2 - x1), float(y2 - y1)),
    )


def _bbox_max_side(bbox: tuple[int, int, int, int]) -> float:
    """Lado mayor (escala aparente) de un bbox XYXY.

    Formula:
        scale = max(x2 - x1, y2 - y1)"""
    return float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def _rect_overflows_unit_square(
    cu: float, cv: float, width: float, height: float,
) -> bool:
    """True si el rectangulo centrado en (cu, cv) de lados (width, height)
    se sale del cuadrado unidad [0, 1]^2.

    Formula:
        overflow = cu - w/2 < 0  o  cu + w/2 > 1
                 o cv - h/2 < 0  o  cv + h/2 > 1"""
    return (cu - width / 2.0 < 0.0 or cu + width / 2.0 > 1.0
            or cv - height / 2.0 < 0.0 or cv + height / 2.0 > 1.0)


def _quantize_position(
    cu: float, cv: float, quantum: float, horizontal: bool,
) -> tuple[int, int, bool]:
    """Clave discreta para agrupar posiciones cercanas en la paleta.

    Formula:
        key = (round(cu / quantum), round(cv / quantum), horizontal)"""
    return (round(cu / quantum), round(cv / quantum), horizontal)


# -- Holgura de encaje parcial -----------------------------------------------

def _partial_fit_slack(
    canonical: float, measured: float, tolerance: float,
) -> float:
    """Semiancho del intervalo factible para reconstruir el centro de un
    fragmento parcialmente ocluido.

    Formula (docs/palletizing_math.md seccion 6.B):
        slack = (canonical - min(measured, canonical)) / 2 + tolerance
    Si `measured >= canonical` la holgura se reduce a `tolerance`."""
    return (canonical - min(measured, canonical)) / 2.0 + tolerance


# -- Escalera de escalas teoricas -------------------------------------------

def _build_scale_ladder(
    reference_scale_px: float, c_z: float, box_height: float, levels: int,
) -> list[float]:
    """Escala aparente teorica de una caja en cada nivel.

    Formula (docs/palletizing_math.md seccion 5.A):
        s(z) = reference_scale_px * (c_z - box_height)
               / (c_z - (z + 1) * box_height)
    para z = 0 .. levels-1.  La calibracion usa UNA medicion directa
    (reference_scale_px) en vez de fL y C_z por separado."""
    return [
        reference_scale_px * (c_z - box_height)
        / (c_z - (z + 1) * box_height)
        for z in range(levels)
    ]


def _ladder_step_gap(s_low: float, s_high: float) -> float:
    """Semiancho relativo del espacio entre dos peldanos consecutivos.

    Formula (docs/palletizing_math.md Proposicion 5.1):
        half_gap = (s_high - s_low) / (2 * s_low)
    Si `tau_rung >= half_gap`, las bandas de aceptacion se solapan y los
    niveles son indistinguibles por escala."""
    return (s_high - s_low) / (2.0 * s_low)


# -- Medicion de footprint por proyeccion homografica -----------------------

def _measure_footprint(
    homography: np.ndarray,
    cx: float, cy: float,
    w_px: float, h_px: float,
) -> tuple[float, float]:
    """Ancho y alto de un bbox en [0,1]^2, midiendo la distancia entre las
    proyecciones de sus extremos por la homografia.

    Formula: con H la homografia al cuadrado unidad,
        du = |pi(H [cx + w/2, cy, 1]^T)_u - pi(H [cx - w/2, cy, 1]^T)_u|
        dv = |pi(H [cx, cy + h/2, 1]^T)_v - pi(H [cx, cy - h/2, 1]^T)_v|
    donde pi es la division perspectiva."""
    points = np.array(
        [
            [cx - w_px / 2.0, cy],
            [cx + w_px / 2.0, cy],
            [cx, cy - h_px / 2.0],
            [cx, cy + h_px / 2.0],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
    return (
        abs(float(projected[1, 0]) - float(projected[0, 0])),
        abs(float(projected[3, 1]) - float(projected[2, 1])),
    )


# -- Errores relativos contra la escalera -----------------------------------

def _scale_relative_errors(
    scale: float, ladder: list[float],
) -> list[float]:
    """Error relativo de la escala observada contra cada peldano.

    Formula:
        e(z) = |scale - s(z)| / s(z)"""
    return [abs(scale - s) / s for s in ladder]


# -- Fraccion de soporte entre dos cajas (rectangulos centrados) ------------

def _rect_support_share(
    u: float, v: float, du: float, dv: float,
    cu: float, cv: float, fu: float, fv: float,
) -> float:
    """Fraccion de la caja SUPERIOR (u, v, du, dv) que esta sostenida por
    la inferior (cu, cv, fu, fv).

    Formula: con I_u, I_v la longitud de interseccion 1D en cada eje,
        share = (I_u * I_v) / area(superior)
    donde area(superior) = du * dv, acotada por 1e-9.  A diferencia de
    R_min, el denominador es SIEMPRE el area de la caja de arriba: lo que
    se mide es cuanto la sostiene cada inferior, no cuanto se solapan."""
    inter_u = max(0.0, min(u + du / 2.0, cu + fu / 2.0)
                  - max(u - du / 2.0, cu - fu / 2.0))
    inter_v = max(0.0, min(v + dv / 2.0, cv + fv / 2.0)
                  - max(v - dv / 2.0, cv - fv / 2.0))
    return (inter_u * inter_v) / max(du * dv, 1e-9)


def _support_polygon_assessment(
    target: tuple[float, float, float, float],
    supports: list[tuple[float, float, float, float]],
    min_contact_ratio: float,
    min_hull_area_ratio: float,
    center_margin_ratio: float,
) -> _SupportPolygonAssessment:
    """Evalúa §12–13 de ``docs/palletizing_math.md`` con geometría exacta.

    Los contactos son las intersecciones rectangulares ``target ∩ support``.
    Se descartan slivers menores que ``min_contact_ratio`` porque un borde de
    raster no representa una fuerza normal fiable. El hull usa TODOS los
    contactos restantes: aquí no existe top-2 ni ``K_max``.

    IMPORTANTE: centroide dentro del hull es una condición necesaria de
    estabilidad, no prueba entrelazado. Por eso ``interlocked`` exige además
    dos soportes físicos distintos. Esta validación corrige el contraejemplo
    de una caja estable pero alineada sobre una única caja inferior.
    """
    u, v, du, dv = target
    target_area = du * dv
    if target_area <= 0.0:
        return _SupportPolygonAssessment(0, 0.0, float("-inf"), False, True, False, ())

    vertices: list[tuple[float, float]] = []
    shares: list[float] = []
    for cu, cv, fu, fv in supports:
        u0 = max(u - du / 2.0, cu - fu / 2.0)
        u1 = min(u + du / 2.0, cu + fu / 2.0)
        v0 = max(v - dv / 2.0, cv - fv / 2.0)
        v1 = min(v + dv / 2.0, cv + fv / 2.0)
        area = max(0.0, u1 - u0) * max(0.0, v1 - v0)
        share = area / target_area
        if share < max(min_contact_ratio, 0.0) or area <= 0.0:
            continue
        shares.append(share)
        vertices.extend(((u0, v0), (u1, v0), (u1, v1), (u0, v1)))

    shares.sort(reverse=True)
    if not vertices:
        return _SupportPolygonAssessment(0, 0.0, float("-inf"), False, True, False, ())

    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(points)
    hull_area_ratio = float(cv2.contourArea(hull)) / target_area
    degenerate = hull_area_ratio < max(min_hull_area_ratio, 0.0)
    center_distance = float(cv2.pointPolygonTest(hull, (float(u), float(v)), True))

    # El margen se expresa contra el lado corto para mantener la misma unidad
    # física al cambiar tamaño de caja. Positivo = centro realmente interior;
    # cero permite frontera exacta, negativo nunca se genera aquí.
    required_margin = max(center_margin_ratio, 0.0) * min(du, dv)
    center_inside = center_distance >= required_margin
    # Soportes del mismo nivel son esencialmente disjuntos (§8.1), por lo que
    # sus participaciones no pueden sumar mucho más de 1. Si lo hacen, dos
    # identidades inferiores se interpenetran o fueron canonicalizadas sobre
    # la misma región; usar ese hull fabricaría estabilidad duplicando área.
    contacts_consistent = sum(shares) <= 1.0 + max(min_contact_ratio, 0.0)
    interlocked = (
        len(shares) >= 2 and contacts_consistent
        and not degenerate and center_inside
    )
    return _SupportPolygonAssessment(
        len(shares), hull_area_ratio, center_distance, center_inside,
        degenerate, interlocked, tuple(shares),
    )


# -- Rasterizacion de caja centrada a indices de grilla ---------------------

def _rasterize_rect(
    u: float, v: float, du: float, dv: float, n: int,
) -> tuple[int, int, int, int]:
    """Indices (u0, u1, v0, v1) de la caja centrada (u, v, du, dv) en una
    grilla n x n, acotados a [0, n].

    Formula:
        u0 = max(0, floor((u - du/2) * n))
        u1 = min(n, ceil((u + du/2) * n))
        v0 = max(0, floor((v - dv/2) * n))
        v1 = min(n, ceil((v + dv/2) * n))"""
    u0 = max(0, int((u - du / 2.0) * n))
    u1 = min(n, int(np.ceil((u + du / 2.0) * n)))
    v0 = max(0, int((v - dv / 2.0) * n))
    v1 = min(n, int(np.ceil((v + dv / 2.0) * n)))
    return u0, u1, v0, v1


# -- Imagen integral 2D y consulta de ventana -------------------------------

def _build_integral_image(grid: np.ndarray) -> np.ndarray:
    """Imagen integral (summed-area table) de una grilla 2D.

    Formula:
        I(y, x) = sum_{j<=y, i<=x} grid(j, i)
    implementada como doble suma acumulativa."""
    n = grid.shape[0]
    integral = np.zeros((n + 1, n + 1), dtype=np.int64)
    integral[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
    return integral


def _integral_window_sums(
    integral: np.ndarray, w: int, h: int,
) -> np.ndarray:
    """Sumas de ventanas h x w sobre la imagen integral.

    Formula (consulta rectangular en SAT):
        S(y, x) = I(y+h, x+w) - I(y, x+w) - I(y+h, x) + I(y, x)"""
    return (integral[h:, w:] - integral[:-h, w:]
            - integral[h:, :-w] + integral[:-h, :-w])


def _integral_rect_sum(
    integral: np.ndarray, u0: int, u1: int, v0: int, v1: int,
) -> int:
    """Suma de una unica ventana [u0,u1) x [v0,v1) en la imagen integral.

    Formula (consulta rectangular en SAT, caso de una sola ventana en vez de
    todas las posiciones como en `_integral_window_sums`):
        S = I(v1, u1) - I(v0, u1) - I(v1, u0) + I(v0, u0)"""
    return (
        int(integral[v1, u1]) - int(integral[v0, u1])
        - int(integral[v1, u0]) + int(integral[v0, u0])
    )


# -- Contraccion por perspectiva entre niveles ------------------------------

def _perspective_shrink(
    du: float, dv: float, ladder: list[float], from_level: int, to_level: int,
) -> tuple[float, float]:
    """Contrae (du, dv) de `from_level` a la escala de `to_level`.

    Formula:
        factor = ladder[to_level] / ladder[from_level]
        du' = du * factor,  dv' = dv * factor"""
    shrink = ladder[to_level] / ladder[from_level]
    return du * shrink, dv * shrink


# -- Esquinas de interseccion de dos cajas centradas (SceneBox) -------------

def _scene_overlap_corners(
    u_a: float, v_a: float, side_a_a: float, side_b_a: float,
    u_b: float, v_b: float, side_a_b: float, side_b_b: float,
) -> tuple[float, float, float, float] | None:
    """Esquinas (u0, v0, u1, v1) de la interseccion de dos cajas de la
    escena isometrica, centradas en (u, v) con lados (side_a, side_b), o
    None si la interseccion es vacia.

    Formula: misma interseccion de intervalos que _rect_intersection_area,
    pero con la firma de atributos de SceneBox y devolviendo las esquinas
    en vez del area."""
    u0 = max(u_a - side_a_a / 2.0, u_b - side_a_b / 2.0)
    u1 = min(u_a + side_a_a / 2.0, u_b + side_a_b / 2.0)
    v0 = max(v_a - side_b_a / 2.0, v_b - side_b_b / 2.0)
    v1 = min(v_a + side_b_a / 2.0, v_b + side_b_b / 2.0)
    if u1 <= u0 or v1 <= v0:
        return None
    return u0, v0, u1, v1


# -- Fraccion de area de interseccion sobre el target -----------------------

def _template_min_evidence(capacity: int) -> int:
    """Primera cantidad estrictamente mayor que media capa.

    Formula:
        min_evidence = capacity // 2 + 1"""
    if capacity < 1:
        raise ValueError("capacity debe ser positiva")
    return capacity // 2 + 1


def _is_duplicate_observation(
    item_center: tuple[float, float], item_size: tuple[float, float],
    kept_center: tuple[float, float], kept_size: tuple[float, float],
) -> bool:
    """True si dos observaciones representan el mismo bbox re-detectado.

    Dos niveles reales pueden compartir centro pero cambiar de escala por
    perspectiva. Formula: con distancia entre centros d y area_ratio entre
    tamanos,
        center_limit = 0.25 * min(lados de item y de kept)
        area_ratio = area(item) / area(kept)
        duplicado = d <= center_limit  y  0.75 <= area_ratio <= 1/0.75"""
    iu, iv = item_center
    ku, kv = kept_center
    iw, ih = item_size
    kw, kh = kept_size
    center_distance = float(np.hypot(iu - ku, iv - kv))
    center_limit = 0.25 * min(iw, ih, kw, kh)
    area_ratio = (iw * ih) / max(kw * kh, 1e-9)
    return center_distance <= center_limit and 0.75 <= area_ratio <= 1.0 / 0.75


def _affine_point(
    u: float, v: float, sx: float, sy: float, tx: float, ty: float,
) -> tuple[float, float]:
    """Punto (u, v) transformado por escala XY independiente + traslacion.

    Formula:
        u' = u * sx + tx
        v' = v * sy + ty"""
    return u * sx + tx, v * sy + ty


def _affine_size(width: float, height: float, sx: float, sy: float) -> tuple[float, float]:
    """Tamano (width, height) escalado por sx, sy (sin traslacion).

    Formula:
        width'  = width * sx
        height' = height * sy"""
    return width * sx, height * sy


def _greedy_unique_match(
    edges: list[tuple[float, int, int]],
) -> tuple[int, float]:
    """Empareja aristas (distancia, id_a, id_b) por distancia creciente,
    exigiendo que cada id_a e id_b se use a lo sumo una vez.

    Formula (asignacion voraz, no optima global pero suficiente para votar
    una traslacion candidata): con E ordenado por distancia ascendente,
        matched = {(a, b) in E : a, b no usados aun}, tomados en orden
        total_error = suma de las distancias de `matched`
    Devuelve (|matched|, total_error)."""
    used_a: set[int] = set()
    used_b: set[int] = set()
    total_error = 0.0
    matched = 0
    for distance, id_a, id_b in sorted(edges):
        if id_a in used_a or id_b in used_b:
            continue
        used_a.add(id_a)
        used_b.add(id_b)
        total_error += distance
        matched += 1
    return matched, total_error


def _log_ratio_size_error(width_ratio: float, height_ratio: float) -> float:
    """Penalizacion de tamano simetrica en escala logaritmica.

    Formula:
        size_error = |log(width_ratio)| + |log(height_ratio)|
    Simetrica porque un slot al doble o a la mitad de tamano castiga igual."""
    return abs(log(width_ratio)) + abs(log(height_ratio))


def _linear_ratio_size_error(
    width: float, height: float, slot_width: float, slot_height: float,
) -> float:
    """Penalizacion de tamano por diferencia relativa absoluta (no log).

    Formula:
        size_error = |width - slot_width| / slot_width
                   + |height - slot_height| / slot_height
    con denominadores acotados por abajo en 1e-9."""
    return (
        abs(width - slot_width) / max(slot_width, 1e-9)
        + abs(height - slot_height) / max(slot_height, 1e-9)
    )


def _rect_overlap_over_target(
    target: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> float:
    """Area de interseccion dividida por el area del TARGET (no del menor).

    Formula:
        R_target(A, B) = area(A cap B) / area(A)
    con area(A) = target[2] * target[3], acotada por 1e-9."""
    return _rect_intersection_area(target, other) / max(
        target[2] * target[3], 1e-9,
    )
