"""
Genera una figura simetrica (eje horizontal) formada por tres bloques
tipo L, con todas las lineas paralelas a los ejes (horizontal/vertical).

Los 3 bloques se definen primero SIN separacion (tocandose exactamente),
y luego cada uno se "encoge" (erosion) la misma distancia GAP/2 hacia
adentro. Como los tres se encogen por igual, la separacion que queda
entre cualquier par de bloques que se tocaban es siempre GAP: uniforme,
sin superposicion y sin intersecciones.
"""

import matplotlib.pyplot as plt

GAP = 0.1  # separacion final visible entre bloques adyacentes
STEP_H = 0.6  # alto del escalon central: igual para naranja, rojo (medio) y azul


def signed_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def erode_orthogonal_polygon(points, d):
    """Encoge (hacia adentro) un poligono ortogonal (solo lados H/V)
    una distancia d, moviendo cada lado d hacia el interior."""
    pts = points[:]
    if signed_area(pts) < 0:  # forzar orientacion CCW
        pts = pts[::-1]

    n = len(pts)
    # Para cada lado, calcular la linea desplazada (mismo eje, coord corrida).
    shifted_edges = []  # ("h", y_new) o ("v", x_new) por cada lado i -> i+1
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:  # lado horizontal
            y_new = y1 + d if x2 > x1 else y1 - d
            shifted_edges.append(("h", y_new))
        else:  # lado vertical
            x_new = x1 - d if y2 > y1 else x1 + d
            shifted_edges.append(("v", x_new))

    # Cada vertice nuevo = interseccion del lado anterior y el siguiente.
    new_pts = []
    for i in range(n):
        prev_edge = shifted_edges[i - 1]
        curr_edge = shifted_edges[i]
        if prev_edge[0] == "v":
            x = prev_edge[1]
            y = curr_edge[1]
        else:
            x = curr_edge[1]
            y = prev_edge[1]
        new_pts.append((x, y))
    new_pts.append(new_pts[0])
    return new_pts


# --- Bloques definidos SIN separacion (se tocan exactamente). ---
# Importante: TOP y BOTTOM tambien se tocan entre si en y=0 (x de 0.6 a 3),
# no solo a traves de LEFT, para que la erosion deje la MISMA separacion
# entre cualquier par de bloques (1-2, 1-3 y 2-3).
LEFT_BLOCK = [
    (-3.0, 1.5),
    (0.0, 1.5),
    (0.0, STEP_H),
    (0.6, STEP_H),
    (0.6, -STEP_H),
    (0.0, -STEP_H),
    (0.0, -1.5),
    (-3.0, -1.5),
]

TOP_BLOCK = [
    (0.0, 1.5),
    (3.0, 1.5),
    (3.0, 0.0),
    (0.6, 0.0),
    (0.6, STEP_H),
    (0.0, STEP_H),
]

BOTTOM_BLOCK = [(x, -y) for (x, y) in TOP_BLOCK]  # reflejo exacto en y=0

# Cada bloque es una L: una parte grande + una parte chica (el escalon).
# Un rectangulo interno va en la parte grande y otro en la parte chica.
LEFT_BIG_AREA = (-3.0, 0.0, -1.5, 1.5)
LEFT_SMALL_AREA = (0.0, 0.6, -STEP_H, STEP_H)

TOP_BIG_AREA = (0.6, 3.0, 0.0, 1.5)
TOP_SMALL_AREA = (0.0, 0.6, STEP_H, 1.5)

BOTTOM_BIG_AREA = (0.6, 3.0, -1.5, 0.0)
BOTTOM_SMALL_AREA = (0.0, 0.6, -1.5, -STEP_H)

# Encoger los 3 la misma distancia -> separacion final uniforme = GAP.
LEFT_BLOCK = erode_orthogonal_polygon(LEFT_BLOCK, GAP / 2)
TOP_BLOCK = erode_orthogonal_polygon(TOP_BLOCK, GAP / 2)
BOTTOM_BLOCK = erode_orthogonal_polygon(BOTTOM_BLOCK, GAP / 2)


def inset_rect_sides(area, left, right, bottom, top):
    """Encoge un area (xmin,xmax,ymin,ymax) con un margen distinto por lado."""
    xmin, xmax, ymin, ymax = area
    return (xmin + left, xmax - right, ymin + bottom, ymax - top)


def rect_to_points(rect):
    xmin, xmax, ymin, ymax = rect
    return [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin), (xmin, ymax)]


# Un rectangulo interno por sub-parte (grande y chica) de cada bloque.
# En el lado donde la parte grande y la chica son vecinas (dentro del mismo
# bloque) cada rectangulo solo retrocede GAP/2, para que la separacion final
# entre ambos sea GAP (no el doble). En el resto de lados, que dan al
# contorno del bloque, retroceden GAP completo.
LEFT_RECT_A = inset_rect_sides(LEFT_BIG_AREA, GAP, GAP / 2, GAP, GAP)
LEFT_RECT_B = inset_rect_sides(LEFT_SMALL_AREA, GAP / 2, GAP, GAP, GAP)

TOP_RECT_A = inset_rect_sides(TOP_BIG_AREA, GAP / 2, GAP, GAP, GAP)
TOP_RECT_B = inset_rect_sides(TOP_SMALL_AREA, GAP, GAP / 2, GAP, GAP)

BOTTOM_RECT_A = inset_rect_sides(BOTTOM_BIG_AREA, GAP / 2, GAP, GAP, GAP)
BOTTOM_RECT_B = inset_rect_sides(BOTTOM_SMALL_AREA, GAP, GAP / 2, GAP, GAP)

fig, ax = plt.subplots(figsize=(6.4, 4.2))


def draw_block(points, color):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, linewidth=3, solid_joinstyle="miter")


draw_block(LEFT_BLOCK, "darkred")
draw_block(TOP_BLOCK, "orange")
draw_block(BOTTOM_BLOCK, "blue")

for rect in (LEFT_RECT_A, LEFT_RECT_B):
    draw_block(rect_to_points(rect), "darkred")
for rect in (TOP_RECT_A, TOP_RECT_B):
    draw_block(rect_to_points(rect), "orange")
for rect in (BOTTOM_RECT_A, BOTTOM_RECT_B):
    draw_block(rect_to_points(rect), "blue")

ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-2.0, 2.0)

fig.tight_layout()
fig.savefig("scratch_shape.png", dpi=150)
print("Guardado: scratch_shape.png")
