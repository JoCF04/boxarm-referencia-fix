# Fundamentos matemáticos consolidados del conteo de paletizado

Este es el **único documento matemático normativo** del módulo. Conserva sin
resumir la teoría validada que antes vivía en `palletizing_math.md` y añade
las mejoras propuestas en `new_math.md`, corregidas contra el comportamiento
real de `src/boxarm/vision/palletizing/`.

## Estado de consolidación y decisiones

- **Conservado:** homografía, footprint canónico, mediana/MAD, razones de
  área, monotonicidad, escala aparente, oclusión, reconciliación e
  identificabilidad del documento anterior.
- **Sustituido:** el soporte top-2 rígido se generaliza mediante polígono de
  soporte con todos los contactos válidos.
- **Conservado como respaldo:** cobertura acumulada y balance $K/\phi$ se
  usan cuando el hull es degenerado o la geometría no está bien condicionada.
- **Corregido:** el hull **no rechaza automáticamente $K=1$**. Un único
  contacto puede contener el centroide; se exigen dos soportes independientes.
- **No llevado al runtime:** Lie/$PGL(3)$, teoría espectral, energía
  variacional, Bayes, cohomología y homología persistente quedan como marco
  teórico o diagnóstico offline.
- **Retirado:** no se usa un $K_{max}$ por catálogo ni umbrales por SKU. Los
  límites efectivos derivan de geometría y tolerancia de raster/localización.

---

Este documento es la teoría, sola. Define objetos, enuncia proposiciones y las
demuestra. No dice qué está implementado ni cómo se llama la función que lo
hace — eso vive en [`palletizing_counting.md`](palletizing_counting.md), que
referencia cada resultado de aquí por número de sección.

Convención: `Definición`, `Proposición`/`Teorema`, `Demostración`, `Corolario`,
`Observación`. Toda cota numérica (0.70, 0.75, 2.0, …) es un parámetro de
calibración, no una constante matemática; se escribe con su símbolo
($\tau,\rho_{max},\dots$) y el valor operativo vive en el otro documento.

---

## 1. Objetos primitivos

**Definición 1.1 (Detección).** Para un frame $t$, una detección es una
tupla

$$
d=(x_1,y_1,x_2,y_2,c,p),\qquad B=[x_1,x_2]\times[y_1,y_2]\subset\mathbb R^2,
$$

con clase $c\in\mathcal K=\{0,\dots,K-1\}$ y confianza $p\in[0,1]$. El
centro de $d$ es

$$
q=\left(\frac{x_1+x_2}2,\frac{y_1+y_2}2\right),
$$

y sus dimensiones crudas son $w=x_2-x_1$, $h=y_2-y_1$.

**Definición 1.2 (Rectificación).** Sea $H\in\mathbb R^{3\times3}$ la
homografía determinada por cuatro puntos de control del ROI y
$\pi([a,b,c]^\top)=(a/c,b/c)$ la proyección homogénea. La rectificación de
un punto es

$$
(u,v)=\pi(H[x,y,1]^\top)\in[0,1]^2.
$$

Aplicada a las cuatro esquinas de $B$, produce un footprint rectangular
normalizado $F=(u,v,d_u,d_v)$ con área $\mu(F)=d_u d_v$.

**Definición 1.3 (Footprint canónico).** Para eliminar la ambigüedad
introducida por rotaciones de 90°, se define

$$
a=\min(d_u,d_v)\quad(\text{lado corto}),\qquad
\ell=\max(d_u,d_v)\quad(\text{lado largo}),
$$

$$
o=\mathbf 1\{d_u\ge d_v\}\quad(\text{orientación}).
$$

El par $(a,\ell)$ es invariante bajo rotación de 90°; $o$ no lo es y se
conserva por separado. Un rectángulo físico de tamaño $(a,\ell)$ tiene
entonces dos representaciones posibles como conjunto en $\mathbb R^2$:

$$
Q^H=\left[-\tfrac\ell2,\tfrac\ell2\right]\times\left[-\tfrac a2,\tfrac a2\right],
\qquad
Q^V=\left[-\tfrac a2,\tfrac a2\right]\times\left[-\tfrac\ell2,\tfrac\ell2\right].
$$

Toda caja física es una traslación de $Q^H$ o $Q^V$: el tamaño no es una
incógnita libre una vez fijada la clase y el nivel; las incógnitas son
centro, orientación y nivel.

---

## 2. Estimadores robustos

**Definición 2.1 (Mediana superior).** Para una muestra ordenada
$x_{(1)}\le\cdots\le x_{(n)}$,

$$
\operatorname{med}(x)=
\begin{cases}
x_{((n+1)/2)}, & n\text{ impar},\\
x_{(n/2+1)}, & n\text{ par (mediana superior)}.
\end{cases}
$$

La segunda rama es una elección deliberada: para $n$ par, la mediana
superior es siempre un valor de la muestra. El promedio de las dos centrales
no lo es, e inventaría una medida nunca observada.

**Definición 2.2 (MAD).** $MAD_x=\operatorname{med}\,|x_i-\operatorname{med}(x)|$.

**Proposición 2.3 (Punto de ruptura de la mediana).** Sea $x_1,\dots,x_n$
una muestra y sea $m$ el número de valores que un adversario puede alterar
arbitrariamente. La mediana permanece dentro del rango original de la
muestra no contaminada si y solo si

$$
m<\left\lceil\frac n2\right\rceil.
$$

*Demostración.* Ordénese la muestra. Alterar $m$ valores solo puede mover
como máximo $m$ posiciones del orden hacia cualquier extremo. La posición
mediana es $\lfloor n/2\rfloor+1$; para desplazarla fuera del rango
original de los $n-m$ valores no contaminados, el adversario necesita
controlar más de la mitad de las posiciones alrededor del centro, es decir
$m\ge\lceil n/2\rceil$. Para $m<\lceil n/2\rceil$ la posición mediana
sigue cayendo dentro de valores no contaminados. $\blacksquare$

En contraste, la media aritmética $\bar x=\frac1n\sum x_i$ tiene punto de
ruptura $1/n$: un solo valor arbitrariamente grande la desplaza sin cota.
Esto justifica usar $\operatorname{med}$ en vez de $\bar x$ para
cualquier tamaño canónico estimado a partir de bboxes potencialmente
recortados o inflados.

**Definición 2.4 (Filtro de compatibilidad).** Una observación $(a_j,\ell_j)$
es compatible con el consenso $(\widetilde a,\widetilde\ell)$ si

$$
|a_j-\widetilde a|\le\lambda\,MAD_a+\varepsilon
\quad\text{y}\quad
|\ell_j-\widetilde\ell|\le\lambda\,MAD_\ell+\varepsilon.
$$

Solo las observaciones compatibles pueden actualizar
$(\widetilde a,\widetilde\ell)$; esto hace el estimador un M-estimador de
punto fijo: el consenso se recalcula únicamente sobre su propia vecindad de
tolerancia, lo que evita que una única observación extrema, aunque pase el
filtro inicial de tamaño de la Sección 3, siga inflando el consenso en
iteraciones sucesivas.

---

## 3. Razones de área

**Definición 3.1.** Para $A,B\subset\mathbb R^2$ medibles con
$\mu(A),\mu(B)>0$:

$$
IoU(A,B)=\frac{\mu(A\cap B)}{\mu(A\cup B)},\qquad
R_{min}(A,B)=\frac{\mu(A\cap B)}{\min(\mu(A),\mu(B))}.
$$

**Proposición 3.2.** $IoU(A,B)\le R_{min}(A,B)$ siempre, con igualdad si y
solo si $\mu(A)=\mu(B)$.

*Demostración.* $\mu(A\cup B)=\mu(A)+\mu(B)-\mu(A\cap B)\ge\max(\mu(A),\mu(B))\ge
\min(\mu(A),\mu(B))$, y la igualdad en la primera desigualdad ocurre exactamente
cuando $\mu(A\cap B)=\min(\mu(A),\mu(B))$ — es decir, cuando el más pequeño
está contenido en el más grande —, mientras que la igualdad
$\max=\min$ ocurre exactamente cuando $\mu(A)=\mu(B)$. Ambas desigualdades
se combinan como cota, y en general $\mu(A\cup B)\ge\min(\mu(A),\mu(B))$
directamente. Dividiendo $\mu(A\cap B)$ por un denominador mayor o igual da
$IoU\le R_{min}$; la igualdad exacta de las dos razones requiere
$\mu(A\cup B)=\min(\mu(A),\mu(B))$, lo que fuerza $\mu(A)=\mu(B)$ y
$A\subseteq B$ o $B\subseteq A$ con esa misma área. $\blacksquare$

**Consecuencia práctica.** Si $B$ es un recorte de una caja $A$ mucho más
grande ($\mu(B)\ll\mu(A)$, $B\subset A$), entonces $R_{min}(A,B)=1$
mientras que $IoU(A,B)=\mu(B)/\mu(A)\to0$. Por eso $R_{min}$ es el
estadístico correcto para "¿este fragmento pertenece a esta identidad?" y
$IoU$ lo es para "¿estas dos observaciones consecutivas del mismo tamaño
son el mismo objeto?" — son pruebas de hipótesis distintas y no
intercambiables.

---

## 4. Identidad espacial y monotonicidad

**Definición 4.1.** Una identidad física es un par $I=(g,z)\in\mathcal G\times\mathbb N$,
celda espacial y nivel. Su ocupación es $\chi:\mathcal G\times\mathbb N\to\{0,1\}$.

**Axioma 4.2 (Monotonicidad de ocupación).** La única transición válida de
$\chi(g,z)$ en el tiempo es $0\to1$. No existe $1\to0$.

**Axioma 4.3 (No degradación de nivel).** Si una detección en el tiempo
$t_2>t_1$ coincide espacialmente con una identidad confirmada en nivel
$i$ en $t_1$, el nivel resultante $z$ satisface $z\ge i$.

Estos dos axiomas convierten el conteo en una función monótona no
decreciente del tiempo: $total(t)$ es no decreciente, y cada $(g,z)$ una
vez marcado en 1 permanece en 1. Es la propiedad que hace válida la fórmula
de acumulación $total=initial+placed$ de la Sección 6 del documento
cliente: basta con nunca restar.

**Alcance del Axioma 4.2: una paleta, no el proceso completo.** La
monotonicidad rige *dentro* de la vida de una carga física. Cuando el
operador retira toda la carga y aparece una paleta nueva, `chi` completo se
reinicia a 0 como una única transición nombrada
(`GridCounter.reset_pallet()`, disparada por `gate.empty_pallet_debounce_frames`
frames `COUNTING` seguidos sin ninguna detección habiendo cajas
confirmadas -- ver `configs/palletizing.md`). Esto NO es una excepción al
axioma sino el límite de su dominio: 4.2 describe la evolución de $\chi$
*para una identidad física dada* mientras esa paleta exista; el reinicio
es el borde entre dos funciones $\chi$ distintas, cada una monótona en su
propio intervalo de tiempo. La prueba de punto fijo de §25 sigue
aplicando sin cambios dentro de cada intervalo.

---

## 5. Soporte trabado (interlocking)

**Definición 5.1.** Para una caja candidata $C$ en el nivel $z+1$ y el
conjunto de cajas confirmadas $\{F_i\}$ en el nivel $z$, la fracción de
soporte es

$$
s_i=\frac{\mu(C\cap F_i)}{\mu(C)}\in[0,1].
$$

Ordénense en forma decreciente y sean $s_{(1)}\ge s_{(2)}\ge\cdots$ los
valores ordenados con $F_{(1)},F_{(2)}$ las cajas asociadas a los dos
mayores.

**Definición 5.2 (Cobertura top-2 y balance).**

$$
Coverage^{(2)}(C)=\frac{\mu\bigl(C\cap(F_{(1)}\cup F_{(2)})\bigr)}{\mu(C)},
\qquad
\rho(C)=\frac{s_{(1)}}{s_{(2)}}\ (s_{(2)}>0).
$$

En la implementación, las participaciones medidas contienen incertidumbre de
borde. Si $\varepsilon_r$ es el error geométrico relativo, se reparte entre
los dos apoyos la banda $u=\varepsilon_r/2$ y se decide con

$$
\rho_{\varepsilon}(C)=
\frac{\max(0,s_{(1)}-u)}{s_{(2)}+u}.
$$

Esta corrección evita rechazar apoyos reales como $55\%/23\%$ por unos
milímetros de error de localización, pero un contacto residual $90\%/10\%$
continúa desbalanceado. No depende de clase, moneda ni SKU.

#### 5.2.1 Derrame de etiquetado frente a soporte compartido real

Hay que separar dos fenómenos visualmente parecidos:

1. **Alineación uno-a-uno con ruido.** La caja superior está esencialmente
   encima de una sola inferior. El bbox segmentado sobresale unos píxeles y
   produce una intersección residual con una vecina. Ese derrame NO demuestra
   un patrón trabado.
2. **Soporte compartido real.** La caja está desplazada respecto de la rejilla
   inferior y una parte geométricamente significativa descansa sobre otra
   caja. Aunque el primer apoyo sea dominante (por ejemplo $58\%/17\%$), el
   segundo contacto puede ser físico y no ruido.

Sea $p$ el paso entre centros en el eje relevante y $\Delta$ el desplazamiento
de la caja superior respecto de la rejilla inferior. La celda base y la fase
del patrón son

$$
k=\left\lfloor\frac{\Delta}{p}\right\rfloor,
\qquad
f=\operatorname{frac}\!\left(\frac{\Delta}{p}\right)
=\frac{\Delta}{p}-\left\lfloor\frac{\Delta}{p}\right\rfloor.
$$

En el caso ideal unidimensional, las participaciones vecinas son
aproximadamente $1-f$ y $f$. Por tanto, exigir siempre $50/50$ o un cociente
fijo pequeño es incorrecto: la fase puede producir legítimamente $75/25$,
$83/17$, etc.

La implementación actual NO reconstruye $f$ explícitamente: calcula las
intersecciones rectangulares reales, que ya contienen esa información en dos
dimensiones, y corrige el balance por incertidumbre geométrica. Una mejora
posterior podría estimar la fase temporalmente para separar todavía mejor un
derrame inestable de un segundo apoyo estable. Hasta entonces se mantienen
juntas estas evidencias: nivel inferior lleno, cobertura suficiente, al menos
dos intersecciones y balance corregido; ninguna por separado confirma nivel.

**Proposición 5.3 (Equivalencia balance/margen).** Para $s_{(1)}\ge s_{(2)}>0$,

$$
\rho(C)\le\rho_{max}
\iff
\delta(C):=\frac{s_{(1)}-s_{(2)}}{s_{(1)}+s_{(2)}}\le\frac{\rho_{max}-1}{\rho_{max}+1}.
$$

*Demostración.* Sea $r=s_{(1)}/s_{(2)}\ge1$. Entonces
$\delta=\frac{r-1}{r+1}$, función estrictamente creciente de $r$ en
$[1,\infty)$ (su derivada es $2/(r+1)^2>0$). Por monotonicidad,
$r\le\rho_{max}\iff\delta\le\frac{\rho_{max}-1}{\rho_{max}+1}$. Con
$\rho_{max}=2$ el umbral en $\delta$ es $1/3$. $\blacksquare$

**Proposición 5.4 (Necesidad de ambas condiciones).** Ni $Coverage^{(2)}\ge\tau$
ni $\rho\le\rho_{max}$ implica la otra; ambas son necesarias.

*Demostración por contraejemplos.* (a) Cuatro soportes iguales
$s_1=s_2=s_3=s_4=1/4$: $\rho=1\le\rho_{max}$ para cualquier $\rho_{max}\ge1$,
pero $Coverage^{(2)}=1/2<0.75$. El balance no implica cobertura. (b) Dos
soportes $s_{(1)}=0.9,\ s_{(2)}=0.1$ con las dos cajas ajenas cubriendo
toda $C$: $Coverage^{(2)}=1\ge0.75$, pero $\rho=9>2$. La cobertura no
implica balance. $\blacksquare$

**Interpretación.** El caso (a) es el patrón "un cuarto de caja sobre cada
una de cuatro vecinas": geométricamente compatible con estar apoyado, pero
sin evidencia dominante de una pareja específica — se descarta porque el
patrón trabado real (ladrillo sobre dos) exige que dos cajas expliquen la
mayoría del área. El caso (b) es casi una coincidencia uno-a-uno con una
segunda caja casi irrelevante — no es "apoyo sobre dos", es "apoyo sobre una,
con ruido".

---

## 6. Escalera de escala aparente

**Definición 6.1.** Bajo proyección en perspectiva con altura de cámara
$c_z$ y separación entre niveles $h$, la escala aparente de un objeto de
tamaño fijo en el nivel $z$ respecto a una referencia $s_{ref}$ en el
nivel 0 es

$$
s(z)=s_{ref}\,\frac{c_z-h}{c_z-(z+1)h}.
$$

*Derivación.* Bajo un modelo de cámara pinhole con foco $f$ y altura de
cámara $c_z$ sobre el plano del nivel $z=-1$ (piso), un objeto físico de
tamaño $\Delta$ a altura $y=zh$ está a distancia focal
$c_z-zh$ de la cámara, y su tamaño en píxeles es
$\sigma(z)=f\Delta/(c_z-zh)$. Definiendo el nivel 0 como referencia
efectiva a altura $h$ (la caja apoya sobre el piso pero su centro óptico
relevante está a la altura de una caja),

$$
\frac{s(z)}{s_{ref}}=\frac{\sigma(z)}{\sigma(0)}
=\frac{c_z-h}{c_z-(z+1)h}.\qquad\blacksquare
$$

**Proposición 6.2 (Asignación de nivel por vecino más cercano).** Una escala
observada $s$ se asigna al nivel

$$
\hat z=\arg\min_z\left|\frac{s-s(z)}{s(z)}\right|,
$$

y se acepta solo si $\left|\frac{s-s(\hat z)}{s(\hat z)}\right|\le\tau_{rung}$.
Esto es una cuantización de $s$ sobre la sucesión $\{s(z)\}$, válida
como criterio de asignación si y solo si la sucesión es estrictamente
separable, es decir si los intervalos de tolerancia
$\bigl[s(z)(1-\tau_{rung}),\,s(z)(1+\tau_{rung})\bigr]$ son disjuntos dos a
dos. Como $s(z)$ es estrictamente creciente en $z$ (para $h>0$,
$c_z-(z+1)h$ decrece), la separabilidad depende solo de que
$\tau_{rung}$ sea menor que la mitad de la brecha relativa mínima entre
peldaños consecutivos — una condición verificable numéricamente para
$(c_z,h)$ dados, no garantizada en general.

---

## 7. Álgebra de oclusión

### 7.1 El problema

Con el nivel $i-1$ lleno a capacidad $n$ y $k$ cajas confirmadas
encima, sea $p$ el número de cajas de $i-1$ parcialmente visibles. $p$
no es función de $k$ solo: depende de la disposición de las $k$ cajas
superiores.

**Proposición 7.1 (Cotas triviales).** $2\le p\le\min(n,4k)$ para $k\ge1$.

*Demostración (cota superior).* Cada caja rectangular superior, colocada de
forma genérica, puede solapar como máximo 4 vecinas inferiores (una
esquina puede caer sobre hasta 4 celdas de una grilla regular), luego
$p\le4k$; y $p\le n$ trivialmente porque no hay más de $n$ cajas en el
nivel. *(cota inferior).* El caso mínimo, $k=1$ centrada exactamente sobre
el borde compartido de dos celdas, tapa parcialmente exactamente 2.
$\blacksquare$

Las cotas no determinan $p$; hace falta una variable de disposición. Dos
caminos la eliminan: integrar sobre área (§7.2, no depende de la
disposición) o contarla exactamente vía el grafo de incidencia (§7.3, la
usa explícitamente).

### 7.2 Invariante de área

**Teorema 7.2 (Conservación de área bajo oclusión).** Sea el nivel $i-1$
con capacidad $n$, cada footprint de área $A$, completamente cubierto por
el plano del nivel (sin huecos estructurales). Sean $k$ cajas del nivel
$i$ colocadas sin sobresalir del perímetro de la paleta. Entonces

$$
\sum_{j\in\text{nivel }i-1}\mu_{\text{visible}}(F_j)=(n-k)\,A.
$$

*Demostración.* Sea $U_i=\bigcup_{t\in\text{nivel }i}F_t$ la región tapada
por el nivel superior. Por hipótesis $U_i$ no sale del perímetro de la
paleta y cada $F_t$ tiene área $A$ sin solaparse con otra $F_{t'}$
(disjunción esencial, Definición 8.2 más abajo), luego $\mu(U_i)=kA$.
El área total del nivel $i-1$ es $nA$ y se reparte exactamente entre
área visible y área tapada por $U_i$ (toda el área del nivel $i-1$ está,
por hipótesis de cobertura completa, o visible o bajo alguna caja
superior):

$$
\sum_j\mu_{\text{visible}}(F_j)=\mu\bigl((\textstyle\bigcup_j F_j)\setminus U_i\bigr)
=nA-\mu(U_i)=nA-kA=(n-k)A.\qquad\blacksquare
$$

Nótese que la demostración **no usa** la posición de las $k$ cajas ni la
disposición $r$ de la Sección 7.3 — solo sus áreas y la ausencia de
sobresalido. Esto explica por qué el criterio de "nivel lleno" puede
evaluarse por área sin enumerar disposiciones.

**Corolario 7.3 (Criterio de nivel lleno sin conteo por celda).**

$$
k=n-\frac1A\sum_j\mu_{\text{visible}}(F_j),
\qquad
Full(i-1)\iff\sum_j\mu_{\text{visible}}(F_j)+kA\ge nA-\varepsilon.
$$

### 7.3 Identidad exacta de doble conteo

**Teorema 7.4 (Identidad de incidencia).** Constrúyase el grafo bipartito
$\mathcal I$ con partes "arriba" ($k$ cajas del nivel $i$) y "abajo"
(cajas del nivel $i-1$), con arista $(t,j)$ si $\mu(F_t\cap F_j)>0$.
Sea $m_t$ el grado de la caja $t$ de arriba y $d_j$ el grado de la caja
$j$ de abajo. Entonces el número $p$ de cajas de abajo con $d_j\ge1$
satisface

$$
p=\sum_t m_t-\sigma,\qquad
\sigma:=\sum_{j:d_j\ge1}(d_j-1).
$$

*Demostración.* El número de aristas admite dos lecturas de doble conteo:
$E=\sum_t m_t=\sum_{j:d_j\ge1}d_j$. Reescribiendo el lado derecho,
$\sum_{j:d_j\ge1}d_j=\sum_{j:d_j\ge1}\bigl[(d_j-1)+1\bigr]=p+\sigma$.
Igualando ambas expresiones de $E$: $\sum_t m_t=p+\sigma$, de donde
$p=\sum_t m_t-\sigma$. $\blacksquare$

Esta identidad es exacta con o sin ciclos en $\mathcal I$, y sin ninguna
hipótesis sobre el patrón geométrico: es álgebra de grafos pura.

**Corolario 7.5 (Caso bosque).** Si $\mathcal I$ restringido a la partición
"arriba" es un bosque (ninguna caja de abajo es compartida por 3 o más de
arriba de forma que se cierre un ciclo), entonces $\sigma=k-r$, donde
$r$ es el número de componentes conexas ("racimos") de $\mathcal I$.

*Demostración.* Un bosque con $k$ vértices de un lado, $p$ del otro, y
$r$ componentes conexas tiene exactamente $k+p-r$ aristas (cada
componente conexa con $n_c$ vértices totales y forma de árbol tiene
$n_c-1$ aristas; sumando sobre las $r$ componentes,
$E=(k+p)-r$). Por otro lado $E=\sum_t m_t$. Sustituyendo en el Teorema 7.4:

$$
p=\sum_t m_t-\sigma=E-\sigma
\;\Longrightarrow\;
\sigma=E-p=(k+p-r)-p=k-r.\qquad\blacksquare
$$

**Corolario 7.6 (Patrón trabado, $m_t\equiv2$).** Si además cada caja de
arriba pisa exactamente 2 de abajo ($m_t=2\ \forall t$):

$$
p=2k-\sigma=2k-(k-r)=k+r,
\qquad
completas(i-1)=n-p=n-k-r.
$$

Las cotas de la Proposición 7.1 se recuperan como casos extremos de $r$:
$r=1$ (todas las $k$ cajas contiguas, un único racimo) da el mínimo
$p=k+1$; $r=k$ (todas aisladas) da el máximo $p=2k$.

### 7.4 Corrección por perspectiva

La rectificación $H$ (Definición 1.2) se calibra sobre el plano físico del
nivel 0; una caja del nivel $i>0$ está más cerca de la cámara y su
proyección aparece dilatada. El factor de dilatación de área es el cuadrado
del factor lineal de la escalera de escala (§6):

$$
\gamma(i)=\left(\frac{s(i)}{s(i-1)}\right)^{\!2}.
$$

El Teorema 7.2 corregido reemplaza el área nominal $kA$ tapada por el área
realmente proyectada $k\gamma A$:

$$
\sum_j\mu_{\text{visible}}(F_j)=(n-k\gamma)A
\;\Longrightarrow\;
k=\frac{n-\frac1A\sum_j\mu_{\text{visible}}(F_j)}{\gamma}.
$$

Con los valores de referencia $c_z=3.0$, $h=0.30$ (Sección 6),
$\gamma\approx1.23$: cada caja superior tapa aproximadamente 23% más área
de la que le correspondería a un objeto del tamaño nominal del nivel 0.
Ignorar $\gamma$ ($\gamma=1$) sobreestima $k$ sistemáticamente, nunca
al azar, porque el sesgo tiene el mismo signo en cada frame.

### 7.5 Tabla determinista de patrones alternantes

Para una clase calibrada existen dos conjuntos ordenados de rectangulos

$$
T^A=(R^A_0,\ldots,R^A_{n-1}),\qquad
T^B=(R^B_0,\ldots,R^B_{n-1}),
$$

donde el indice es la identidad espacial `cell`, no una coordenada de
cuadricula. La fase desconocida de la paleta es $p\in\{0,1\}$ y la plantilla
del nivel absoluto $z$ es

$$
T_z(p)=T^{(p+z)\bmod 2}.
$$

Asi, A/B no denota paridad absoluta. El primer piso fisico puede usar A o B;
una vez identificada $p$, todos los pisos posteriores quedan determinados por
alternancia. Para cada fase, la incidencia de oclusion y la tabla $p(k)$ son
funciones deterministas de las posiciones de $T_z(p)$ y pueden precomputarse
una sola vez por clase.

Una imagen con $n$ detecciones completas no prueba que represente una
plantilla: las detecciones pueden provenir de $T_z(p)\cup T_{z+1}(p)$. La
validez de una fuente exige pertenencia a una sola capa o evidencia externa
de su topologia.

### 7.6 Orden de autoridad entre los tres criterios

1. El **Teorema 7.2** (con la corrección 7.4) es válido para cualquier
   disposición y cualquier topología de $\mathcal I$, incluyendo ciclos:
   es el criterio de mayor generalidad.
2. La **tabla $p(k)$** (§7.5) es válida solo bajo el supuesto de patrón
   fijo, pero no requiere medir área directamente: predice y verifica.
3. La **identidad de incidencia** (Teorema 7.4) es exacta pero requiere
   construir $\mathcal I$ explícitamente; se usa como diagnóstico cuando 1
   y 2 discrepan, porque la discrepancia señala una detección faltante o un
   ciclo no anticipado en $\mathcal I$.

---

## 8. Reconciliación combinatoria del estado inicial

### 8.1 El problema es conjunto, no secuencial

Cuando la observación inicial mezcla más de un nivel físico, decidir bbox
por bbox "¿a qué nivel pertenece?" es circular: promover una caja exige que
el nivel inferior esté lleno (Corolario 7.3), pero probar que está lleno
requiere ya saber cuáles cajas inferiores están ocultas por las superiores
— que es precisamente lo que se quiere determinar. La resolución correcta
es un problema de **factibilidad conjunta**, no una cadena de decisiones
locales.

### 8.2 Formalización

Sea $P\subset\mathbb R^2$ la paleta rectificada, $\mu$ la medida de
Lebesgue, y para cada nivel $z$ sea $\mathcal L_z$ el conjunto (por
determinar) de cajas físicas asignadas a ese nivel, cada una una traslación
de $Q^H$ o $Q^V$ (Definición 1.3). La incógnita es la tupla completa

$$
X=(\mathcal L_0,\mathcal L_1,\dots,\mathcal L_{Z-1}).
$$

**Definición 8.1 (Disjunción esencial).** $F,G\in\mathcal L_z$, $F\ne G$,
son esencialmente disjuntas si $\operatorname{int}(F)\cap\operatorname{int}(G)=\varnothing$
(equivalentemente $\mu(F\cap G)=0$); con ruido de etiquetado se relaja a
$\mu(F\cap G)/\min(\mu(F),\mu(G))\le\varepsilon_{same}$. Esto convierte
cada $\mathcal L_z$ en una instancia de *rectangle packing*: el problema
de acomodar rectángulos de tamaño fijo sin superposición de interior.

**Definición 8.2 (Configuración admisible).** $X$ es admisible si:

- **(A) Contención:** $F\subseteq P$ para todo $F\in\mathcal L_z$, todo $z$.
- **(B) Capacidad exacta:** $Full(z)\Rightarrow|\mathcal L_z|=n$ exactamente
  (no "al menos $n$").
- **(C) Explicación única:** cada bbox completo observado corresponde a
  exactamente una $F\in\bigcup_z\mathcal L_z$, y cada $F$ explica como
  máximo un bbox por frame — matching bipartito uno a uno.
- **(D) Compatibilidad de fragmentos:** todo fragmento $r$ satisface
  $\mu(r\setminus F)\le\varepsilon_{fit}\,\mu(r)$ para algún
  $F\in\bigcup_z\mathcal L_z$.
- **(E) Consistencia de visibilidad:** si $F\in\mathcal L_z$ se declara
  totalmente oculta, $\mu(F\setminus O_{z+1})\le\varepsilon_{vis}\,\mu(F)$,
  donde $O_{z+1}=\bigcup_{G\in\mathcal L_{z+1}}G$.
- **(F) Soporte top-2:** toda $N\in\mathcal L_{z+1}$ satisface las dos
  condiciones de la Sección 5 ($Coverage^{(2)}\ge\tau_{support}$ y
  $\rho(N)\le\rho_{max}$) frente a $\mathcal L_z$.
- **(G) Cierre por imposibilidad de inserción:** si el nivel $z$ se
  declara cerrado, $\nexists\,q\in P,\,o\in\{H,V\}: (q+Q^o)\subseteq P\setminus U_z$,
  con $U_z=\bigcup_{F\in\mathcal L_z}F$.

#### 8.2.1 Especializacion finita por plantillas calibradas

Sea $D=\{d_1,\ldots,d_m\}$ el conjunto de rectangulos completos observados.
Para fase $p$, nivel base $b$ e hipotesis $q\in\{1,2\}$ niveles, el universo
de slots es

$$
S(p,b,q)=\bigcup_{r=0}^{q-1}\{(b+r,j,R^{(p+b+r)\bmod 2}_j):0\le j<n\}.
$$

Una asignacion admisible es una inyeccion $a:D\hookrightarrow S(p,b,q)$:
dos observaciones no pueden ocupar el mismo par `(nivel, cell)`. Si
$o(d_i)$ es la orientacion observada y $c_i,w_i,h_i$ su centro y lados, el
costo local contra un slot $R_j$ es

$$
e(d_i,R_j)=\|c_i-c_j\|_2+
\lambda\left(\left|\log\frac{w_i}{w_j}\right|+
\left|\log\frac{h_i}{h_j}\right|\right),
$$

finito solo si coinciden orientaciones, la distancia no supera
$\tau_{cell}$ y cada razon lateral pertenece a
$[\rho_{min},\rho_{min}^{-1}]$. El ajuste global es

$$
E(p,b,q)=\min_{a\ \mathrm{inyectiva}}\frac1m
\sum_{i=1}^{m}e(d_i,R_{a(i)}).
$$

La minimizacion es un matching bipartito uno-a-uno. Con $n=15$ puede
resolverse exactamente mediante programacion dinamica sobre mascaras de
ocupacion. No es correcto hacer matching voraz bbox por bbox porque una
decision local puede consumir el unico slot compatible de otra observacion.

Los fragmentos no entran como cajas independientes en $D$. Actuan como
restricciones de visibilidad: una hipotesis de dos niveles solo es admisible
si la union de cajas superiores explica el recorte observado de una caja
inferior. La mera existencia o ausencia de fragmentos no determina $q$.

**Definicion 8.2a (Fase identificable).** La fase se acepta solamente si el
minimo estable de $E(p,b,q)$ es unico fuera del margen de incertidumbre
$\delta$. Si

$$
|E(0,b,q_0)-E(1,b,q_1)|\le\delta,
$$

la fase es no identificable con la evidencia actual y el estado debe
permanecer sin mutar.

**Corolario 8.2b (Cierre inferior).** Si la solucion aceptada contiene al
menos un slot en $b+1$, entonces el nivel $b$ debe estar completo por la
regla de promocion. Por tanto se materializan los $n$ slots de $T_b(p)$,
incluidos aquellos sin evidencia visual directa. Esta inferencia procede de
la plantilla y de la regla de cierre, no de contar bboxes visibles.

### 8.3 Estructura combinatoria

La mediana canónica más las fronteras observadas (bboxes completos y
fragmentos) generan un conjunto **finito** de colocaciones candidatas
$\mathcal Q=\{Q_1,\dots,Q_M\}$, con variable indicadora
$x_m\in\{0,1\}$ por colocación. Las restricciones (A)-(G) son entonces:

- (B), (C), (D) son restricciones de **exact cover**: cada observación
  (bbox completo o fragmento) debe quedar explicada por exactamente una
  colocación seleccionada, y la cardinalidad de colocaciones seleccionadas
  por nivel debe igualar la capacidad declarada.
- (A), (F), (G) son restricciones de **set packing** / factibilidad
  geométrica: colocaciones incompatibles entre sí (que se interpenetran, que
  exceden el perímetro, que dejan un hueco insertable) no pueden coexistir
  en la solución.

El problema conjunto es, tras discretizar $\mathcal Q$, un **programa
entero binario** con restricciones de cobertura exacta y empaquetamiento —
NP-difícil en general (contiene exact cover, uno de los 21 problemas de
Karp), pero de tamaño acotado por $n$ celdas y observaciones de un solo
frame, tratable por poda (branch and bound: descartar ramas que violan (A) o
exceden capacidad antes de expandirlas).

### 8.4 Identificabilidad

**Definición 8.3.** $\Omega(\mathcal D)=\{X: X\text{ satisface (A)–(G) y explica }\mathcal D\}$,
el conjunto de configuraciones compatibles con las detecciones $\mathcal D$.
Dos configuraciones $X\sim Y$ son equivalentes si existe una biyección
entre sus cajas que conserva nivel y orientación y desplaza cada centro a lo
sumo la tolerancia geométrica de reconstrucción. En el caso calibrado, la
equivalencia tambien debe conservar `cell` y fase $p$: intercambiar A por B no
es una renumeracion inocua, sino otra explicacion fisica del piso inicial.

**Teorema 8.4 (Trico­tomía de identificabilidad).** Exactamente uno de los
siguientes tres casos ocurre:

$$
|\Omega/\!\sim|=0\ \Rightarrow\ \text{datos o calibración incompatibles (ninguna configuración explica }\mathcal D\text{)};
$$
$$
|\Omega/\!\sim|=1\ \Rightarrow\ \text{reconstrucción identificable: la preimagen física es única};
$$
$$
|\Omega/\!\sim|>1\ \Rightarrow\ \text{el problema inverso no es identificable con esta evidencia}.
$$

*Demostración.* Trivial por ser $|\cdot|$ una función bien definida sobre
un conjunto finito cociente por una relación de equivalencia: toma
exactamente un valor en $\{0,1,2,\dots\}$, y las tres categorías cubren
todos los valores posibles de $\mathbb N$ sin solaparse. El contenido no
trivial es la interpretación: solo $|\Omega/\!\sim|=1$ permite mutar el
estado físico (crear cajas inferidas, promover niveles) sin arbitrariedad,
porque es el único caso en que "la" solución es una noción bien definida en
vez de una elección entre alternativas igualmente compatibles con los
datos. $\blacksquare$

Esta es la misma noción de identificabilidad usada en problemas inversos
generales: observar una proyección no basta para reconstruir la causa si
existe más de una causa compatible con la misma proyección.

### 8.5 Ejemplo resuelto: 14 completas + 2 fragmentos, capacidad 15

Antes del ejemplo, una advertencia operativa: `15 completas` tampoco implica
`un nivel completo`. Una proyeccion cenital puede mostrar simultaneamente
cajas completas de dos niveles. La cardinalidad solo participa despues del
matching global contra $T_b(p)$ y $T_{b+1}(p)$.

Sean 14 bboxes completos y 2 fragmentos de aproximadamente media caja. Para
una hipótesis que promueve $t$ de las 14 completas al nivel superior y
reconstruye $p$ identidades del nivel inferior a partir de los
fragmentos, el número de cajas inferiores **todavía sin ninguna evidencia
visual** (totalmente ocultas) es

$$
h=n-\bigl[(14-t)+p\bigr].
$$

Una hipótesis sobrevive solo si $h\ge0$ y, además, existen exactamente
$h$ colocaciones para esas cajas ocultas que sean simultáneamente:
esencialmente disjuntas de toda caja ya asignada al nivel (Definición 8.1),
consistentes con la visibilidad (restricción (E) respecto al nivel
superior), y compatibles con el soporte top-2 exigido por las cajas del
nivel superior que se apoyan sobre ellas (restricción (F)).

Con $n=15$, $t=2$, $p=2$:

$$
h=15-\bigl[(14-2)+2\bigr]=15-14=1.
$$

La hipótesis es entonces: 12 cajas del nivel inferior observadas completas,
2 reconstruidas desde fragmentos, 1 totalmente oculta e inferida por
diferencia — y 2 cajas promovidas al nivel superior. Esta hipótesis
particular pertenece a $\Omega(\mathcal D)$ si y solo si la colocación de
la caja oculta satisface (A)-(G); pertenece a la solución **aceptada** solo
si, además, es la única clase de $\Omega/\!\sim$ que lo hace (Teorema 8.4).
La igualdad de conteo $h=1$ es necesaria pero no suficiente.

### 8.6 Invariante de identidad bajo oclusión

Sea $R_i$ el rectángulo canónico de una caja ya confirmada y sea $F_t$ la
detección visible que pretende redetectar esa misma identidad en el instante
$t$. Una oclusión puede reducir la región observable, pero no puede trasladar,
girar ni agrandar físicamente la caja confirmada. Por tanto debe cumplirse,
salvo la incertidumbre geométrica de borde $\varepsilon_r$,

$$
\frac{\mu(F_t\cap R_i)}{\mu(F_t)} \ge 1-2\varepsilon_r.
$$

Aquí $\varepsilon_r$ es el error absoluto de raster/localización dividido
por el lado corto observado, limitado por la misma cota usada en las reglas
de soporte. El factor dos cubre la incertidumbre combinada de los bordes
opuestos. La fracción se normaliza por el área de **la detección**, no por
el rectángulo menor: un fragmento genuino contenido obtiene valor cercano a
uno, mientras dos cajas completas cruzadas pueden tener mucho solape sobre
el menor y aun así una contención baja.

**Consecuencia.** Si una caja horizontal nueva cruza una caja vertical
confirmada, no puede heredar su identidad solo por solaparla. Debe continuar
como candidata nueva y resolver su nivel mediante capacidad y soporte. El
centro, `footprint`, orientación y nivel de una identidad confirmada son
inmutables. Las observaciones posteriores validan visibilidad, pero nunca
reescriben la geometría persistida.

Cuando dos observaciones reclaman la misma identidad, la de mayor área
conserva el emparejamiento principal. La menor puede ser evidencia adicional
de oclusión solo si está contenida en el rectángulo canónico y satisface, con
tolerancia de raster $\varepsilon_r$,

$$w(F_t) \leq w(R_i)+\varepsilon_r,\qquad
  h(F_t) \leq h(R_i)+\varepsilon_r,$$

además de tener área estrictamente menor. Este segundo vínculo no representa
otra caja: es una validación uno-a-muchos de la misma identidad inferior.

---

## 9. Límites teóricos de una cámara cenital 2D

**Observación 9.1.** Si un nivel está totalmente tapado
($\sum_j\mu_{\text{visible}}(F_j)=0$ para todo $j$ en ese nivel), el
Teorema 7.2 es compatible con **cualquier** número de cajas superiores que
produzca esa área tapada total — el sistema pierde un grado de libertad
observacional. Ninguna transformación algebraica de una sola imagen 2D
recupera esa información faltante; se requeriría una segunda vista o una
medida de profundidad independiente.

**Observación 9.2.** El Teorema 7.4 y sus corolarios asumen que cada caja
del nivel superior es correctamente detectada; una detección faltante
introduce una discrepancia entre la predicción del Teorema 7.2 (que no
depende de contar cajas individuales) y la del Teorema 7.4 (que sí). Esa
discrepancia es la señal de diagnóstico descrita en §7.6, no ruido a
promediar.

**Observación 9.3.** El factor de perspectiva $\gamma$ (§7.4) depende de
$(c_z,h)$, parámetros de calibración óptica externos a la geometría del
problema. Un error sistemático en su medición sesga $k$ de forma
sistemática y direccional, no aleatoria: no se corrige promediando sobre más
frames.

---

---

# PARTE II — MEJORAS: generalización del soporte trabado

## 10. Por qué generalizar

El criterio top-2 (§5) asume $m_t\equiv2$ (Corolario 7.6): cada caja de
arriba pisa exactamente dos de abajo. Deja de ser cierto con (a) tamaños de
caja mixtos, o (b) desplazamiento de apilado variable por nivel. La
generalización reemplaza "las dos mayores intersecciones" por dos criterios
independientes, de robustez y fundamento distintos — usados en cascada.

## 11. Mejora A — K dinámico (acumulador de área generalizado)

**Definición 11.1 (K de explicación mínima).**

$$
K(C)=\min\Bigl\{k: \textstyle\sum_{i=1}^k s_{(i)}\ge\tau_{support}\Bigr\}.
$$

**Axioma 11.2 (Nunca sobre una sola).** $K(C)=1\Rightarrow$ rechazar,
siempre — categoría aparte, no caso límite de balance.

**Definición 11.3 (Balance generalizado).**
$\phi(C)=s_{(1)}/\sum_{i=1}^K s_{(i)}$, aceptar si
$\phi\le\rho_{max}/(\rho_{max}+1)$.

**Proposición 11.4 (Consistencia con $K=2$).** Para $K=2$, el criterio de
11.3 es equivalente a $\rho\le\rho_{max}$ (Proposición 5.3) — no hay
regresión sobre el caso ya calibrado en producción.

**Proposición 11.5 (Cota geométrica, tamaños iguales).** Con $C$ traslación
pura (sin rotar) del mismo tamaño que las celdas de abajo,
$K(C)\in\{1,2,4\}$: $1$ sii alineación perfecta en ambos ejes, $2$ sii
offset en un solo eje, $4$ sii offset en ambos.

*Demostración.* $C$ desplazada menos de un lado de celda cruza como máximo
una línea de rejilla por eje; cruzar $c_u,c_v\in\{0,1\}$ líneas parte a $C$
en $(c_u+1)(c_v+1)$ subregiones, cada una en una celda distinta.
$\blacksquare$

$K=3$ no es régimen estable — solo aparece en la frontera de medida cero,
contaminación de borde a filtrar, no apoyo real.

**Corolario 11.6 (Tamaños distintos).**

$$
K(C)\le\Bigl(\lfloor w/a\rfloor+1\Bigr)\Bigl(\lfloor \ell'/\ell\rfloor+1\Bigr),
$$

mismo argumento de líneas de rejilla cruzables. Valor operativo por
catálogo: $K_{max}=4$ (mismo tamaño) hasta $K_{max}=(n+1)^2$ (caja $n\times$
más grande en un eje).

**Observación 11.7 (Rotación ±θ).** No introduce cota nueva — reusa el piso
de ruido $\varepsilon_r$ ya calibrado: para $\theta$ pequeño, los slivers de
esquina tienen área $O(\theta^2)$, filtrados si $s_i<\varepsilon_r$ antes de
construir $K$.

**Definición 11.8 (Régimen de gap).** $\gamma_{gap}=g/\min(a,\ell)$
(gap nominal por clase, no por instancia).

- **Régimen A** ($\gamma_{gap}\le\kappa$): inflar footprints por $g/2$ antes
  de medir $s_i$; cota $K_{max}$ de 11.6 sigue válida.
- **Régimen B** ($\gamma_{gap}>\kappa$): patrón de separación de diseño, no
  packing denso — calibración propia ($\tau_{support}$, $K_{max}$ distintos),
  nunca heredada de A.

**Proposición 11.9.** El régimen A/B es propiedad de clase de caja
($g,a,\ell$ de catálogo), no de instancia — se calibra una vez por clase.

## 12. Mejora B — Polígono de soporte (criterio primario, mecánica de sólidos)

Tomado de robótica de bin-packing: en vez de umbralizar fracciones de área,
usa la condición **necesaria** de equilibrio estático.

**Definición 12.1.** $R_i=C\cap F_i$; $\mathcal S(C)$ = vértices de todos
los $R_i$ con área $>0$; $\mathcal P(C)=\mathrm{ConvexHull}(\mathcal S(C))$.

**Axioma 12.2 (Mecánica clásica).** Bajo gravedad y fricción de Coulomb, una
condición necesaria de equilibrio es $\bar c(C)\in\mathcal P(C)$ — si el
centroide cae fuera del hull de contacto, no existe combinación de fuerzas
normales que cancele el torque neto; el cuerpo vuelca. Es el criterio
estándar en bin-packing robótico (test punto-en-polígono sobre el support
polygon).

**Criterio 12.3.** $\text{ACEPTAR}(C)\iff\bar c(C)\in\mathrm{int}(\mathcal P(C))$.

**Proposición 12.4 (El polígono no demuestra entrelazado por sí solo).**
Con un único soporte, $\mathcal P(C)=R_1$. El centroide de $C$ puede quedar
dentro de $R_1$ aun cuando el solape sea parcial; por tanto el test mecánico
puede demostrar estabilidad estática local, pero **no** que exista soporte
trabado entre múltiples cajas.

*Contraejemplo.* Dos cajas cuadradas iguales, desplazadas menos de medio lado
en un eje, tienen una intersección que todavía contiene el centro de la caja
superior. El test del hull acepta esa geometría aunque $K=1$. $\blacksquare$

**Regla operativa 12.5.** El conteo de paletizado exige al menos dos regiones
de contacto independientes después de filtrar ruido. Esta regla conserva la
semántica de entrelazado del sistema anterior; el polígono decide estabilidad,
no multiplicidad de soportes.
**Proposición 12.6 (Sin techo de K).** El hull incorpora automáticamente
cualquier número de soportes con intersección $>0$ — no requiere
$\tau_{support}$ ni $K_{max}$ como parámetros libres, solo la tolerancia de
tamaño de caja ya calibrada.

**Observación 12.7 (Advertencia multi-nivel, de la literatura de bin
packing robótico).** El test local (solo contra el nivel inmediato) no
verifica que ese nivel esté a su vez bien soportado.

**Definición 12.8 (Estabilidad recursiva).**

$$
\mathrm{Stable}(C)\iff\bar c(C)\in\mathrm{int}(\mathcal P(C))\ \wedge\ \forall F_i\in\text{soporte}(C):\mathrm{Stable}(F_i).
$$

**Proposición 12.9.** Se calcula en $O(N)$ recorriendo niveles de abajo
hacia arriba una sola vez — mismo recorrido que ya usa el sistema para
asignar nivel (Sección 15 de `palletizing_counting.md`), costo marginal
~0.

## 13. Fallback: cuándo usar A en vez de B

**Definición 13.1 (Hull degenerado).**
$\mathrm{area}(\mathcal P(C))<\varepsilon_{hull}\cdot\mu(C)$ — contactos casi
colineales, típicamente ruido severo de detección.

**Regla 13.2.** Si el hull es degenerado, el test punto-en-polígono es
numéricamente inestable (un par de píxeles de error voltea el resultado).
En ese caso, usar el criterio $K/\phi$ (Sección 11) como respaldo — es un
promedio de área, mucho más insensible a error pequeño que un test
geométrico sobre un hull casi plano.

**Proposición 13.3 (Consistencia en los extremos).** Ambos criterios
coinciden en el caso balanceado ($K=2$, $s_{(1)}\approx s_{(2)}$, hull no
degenerado) y en el caso muy desbalanceado — divergen solo en una franja
estrecha alrededor del umbral, donde ambos son igualmente sensibles a
calibración.

**Algoritmo unificado.**

```text
1. R_i = C ∩ F_i, footprints inflados por gap si Régimen A (Def 11.8)
2. descartar R_i con área < eps_r * área(C)  [piso de ruido único, reusado]
3. si no queda ningún R_i: RECHAZAR
4. P = ConvexHull(∪ vértices(R_i))
5. si área(P) < eps_hull * área(C):
       usar FALLBACK K/φ (Sección 11) → retornar ese resultado
6. si no:
       c_bar = centroide(C)
       ACEPTAR si c_bar ∈ int(P), RECHAZAR si no
7. [opcional, costo ~0] propagar Stable(C) recursivo (Def 12.8)
```

## 14. Comparación A vs B

| | K/φ (Mejora A) | Polígono de soporte (Mejora B) |
|---|---|---|
| Parámetros libres | $\tau_{support}$, $\rho_{max}$, $K_{max}$ | solo $\varepsilon_{hull}$ (más los de fallback) |
| Rechazo $K=1$ | regla explícita | regla explícita; el hull solo no basta (12.4–12.5) |
| Techo de soportes | necesita $K_{max}$ calibrado | ninguno — el hull crece solo |
| Robustez a ruido de 1–2 px | alta (promedia área) | baja cerca del borde — de ahí el fallback |
| Verifica estructura multi-nivel | no | sí, gratis (12.8–12.9) |
| Fundamento | calibrado empíricamente | condición necesaria de equilibrio estático |

---
---

# PARTE III — Qué se generaliza a $\mathbb R^n$/$\mathbb C$ y qué no

## 15. Sí se reduce limpio (map/reduce sobre un eje batch)

- **Footprint (Def 1.1–1.3):** tensor $\mathbb R^{N\times4}$; rectificación
  = una contracción matricial con $H$ para las $4N$ esquinas del frame, no
  $N$ llamadas.
- **Orientación como acción de grupo en $\mathbb C$:** $\zeta=d_u+id_v$;
  $\sigma(\zeta)=i\bar\zeta$ intercambia $d_u\leftrightarrow d_v$ — es
  literalmente "girar 90°". $(a,\ell)$ es el invariante del cociente
  $\mathbb C/\mathbb Z_2$; $o$ es qué representante de la órbita era $\zeta$.
  Vectorizado: `min`/`max` sobre partes real e imaginaria de
  $\boldsymbol\zeta\in\mathbb C^N$, sin condicional por caja.
- **Mediana/MAD (Def 2.1–2.4):** `segment_median` agrupado por $(g,z)$ —
  tensor de valores + tensor de índices de grupo, una operación para todas
  las clases y niveles a la vez.
- **$R_{min}$/IoU (Def 3.1):** matriz $N\times M$ vía outer product de
  áreas — exactamente como se vectoriza IoU en librerías de detección de
  objetos.
- **Matching uno a uno (§10 counting):** asignación óptima sobre la matriz
  de costos anterior (Hungarian / `linear_sum_assignment`) — sigue siendo
  combinatorio en sí mismo, pero la construcción del costo es tensorial.
- **Escalera de escala (Prop 6.2):** resta broadcast $N\times Z$ + `argmin`
  sobre el eje $Z$, sin loop por observación ni por nivel.
- **K dinámico (Def 11.1):** `sort` descendente + `cumsum` + `searchsorted`
  por fila — tres primitivas tensoriales estándar, para todas las
  candidatas del frame a la vez.

## 16. NO se reduce a álgebra tensorial pura (irreducible)

- **Polígono de soporte (Def 12.1):** `ConvexHull` es combinatorio
  ($O(h\log h)$, QuickHull/gift wrapping) — batcheable ($N$ hulls pequeños
  en paralelo) pero no una fórmula cerrada en $\mathbb R^n$.
- **Matching bipartito exacto (restricción (C) de §8.2):** programación
  entera — la tensorización ayuda a construir el costo (Sección 15), no a
  resolver la asignación.
- **Reconciliación combinatoria (§8, Teorema 8.4):** exact cover + set
  packing, NP-difícil por diseño (Karp). La *evaluación* de cada hipótesis
  se tensoriza; la *búsqueda* sobre hipótesis sigue siendo discreta —
  branch and bound, no gradiente ni álgebra lineal.

**Regla general.** Si la Proposición dice "para todo $i$" sobre una
propiedad evaluada independientemente por caja, es un `map`/`reduce`
disfrazado de cuantificador universal — tensoriza directo. Si dice "existe
una configuración que satisface simultáneamente" (cuantificador
existencial sobre una estructura combinatoria conjunta), no colapsa a
álgebra en $\mathbb R^n$ sin perder generalidad — es optimización discreta,
y punto.

## 17. Qué NO cambia entre las dos partes

- El footprint canónico, centro, orientación y nivel de una identidad
  confirmada son inmutables en toda versión (§4, §8.6 de la Parte I) — la
  Parte II solo cambia cómo se *decide* soporte, nunca reescribe geometría
  persistida.
- Para el caso ya validado en producción ($K=2$ balanceado), Mejora A
  coincide exactamente con el criterio base de §5 (Proposición 11.4) y
  Mejora B coincide con Mejora A en los extremos (Proposición 13.3) — cero
  regresión de comportamiento conocido.

## 18. Fuentes

- [Counting Stacked Objects (arXiv 2411.19149)](https://arxiv.org/pdf/2411.19149)
- [CountNet3D (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Jenkins_CountNet3D_A_3D_Computer_Vision_Approach_To_Infer_Counts_of_WACV_2023_paper.pdf)
- [Counting Through Occlusion (arXiv 2511.12702)](https://arxiv.org/pdf/2511.12702)
- [Inclusion–exclusion principle (Wikipedia)](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle)
- [Palletizing Pallet Pattern Charts (Robotiq)](https://blog.robotiq.com/palletizing-pallet-pattern-charts)
- Wikipedia, *Support polygon* — definición formal y condición necesaria de
  equilibrio bajo fricción de Coulomb.
- *SDF-Pack: Towards Compact Bin Packing with Signed-Distance-Field
  Minimization* — test de estabilidad por convex hull + centro de masa.
- *Online 3D Bin Packing with Fast Stability Validation and Stable
  Rearrangement Planning* — insuficiencia del test local en pilas
  multinivel (origen de §12.7–12.9).
- Patente *Method and apparatus for palletizing packages of random size and
  weight* — implementación industrial real, umbral de soporte dependiente
  del peso, verificación combinada centro-de-gravedad + polígono de soporte.

---


Continúa `paletizado_completo.md` (Partes I–III). Cada sección de aquí toma
un resultado ya probado y lo reubica dentro de una estructura matemática más
general — no cambia ninguna conclusión operativa, muestra por qué esas
conclusiones eran casos particulares de algo más grande. Numeración
continúa desde §18.

---

## 19. La simetría del footprint es una representación de grupo, no solo $\mathbb Z_2$

La Sección 2 de este documento usó $\mathbb Z_2$ actuando en $\mathbb C$
para las rotaciones de 90°. Eso es el caso discreto de algo más grande: el
grupo diedral $D_4$ (simetrías del cuadrado) actuando sobre $\mathbb C$, y
en el límite continuo, el grupo ortogonal $O(2)$.

**Definición 19.1 (Acción de $D_4$).** $D_4=\langle r,\sigma\mid r^4=\sigma^2=1,\ \sigma r\sigma=r^{-1}\rangle$
actúa sobre $\zeta\in\mathbb C$ por $r(\zeta)=i\zeta$ (rotación de 90°) y
$\sigma(\zeta)=\bar\zeta$ (reflexión). El footprint canónico de la Parte I
usa solo el subgrupo $\{1,\sigma r\}\cong\mathbb Z_2$ (la reflexión
diagonal), porque el problema físico solo produce ambigüedad de 90°, nunca
de reflexión especular real (una caja no se refleja, se rota).

**Proposición 19.2 (El invariante completo bajo $\langle r\rangle\cong\mathbb Z_4$).**
Para el subgrupo cíclico de rotaciones puras $\langle r\rangle$, el
invariante completo de la órbita $\{\zeta,i\zeta,-\zeta,-i\zeta\}$ es
$\zeta^4\in\mathbb C$ — el mapa $\zeta\mapsto\zeta^4$ es exactamente el
cociente $\mathbb C/\mathbb Z_4$.

*Demostración.* $\mathbb Z_4$ actúa por multiplicación por raíces cuartas
de la unidad; dos puntos están en la misma órbita sii difieren por una raíz
cuarta de la unidad, sii tienen la misma cuarta potencia — estándar en
teoría de invariantes de grupos cíclicos actuando por multiplicación
escalar. $\blacksquare$

**Por qué esto no se usa en el sistema (y por qué es importante saberlo).**
El sistema solo necesita $\mathbb Z_2$ porque una caja rectangular con
$a\ne\ell$ NO tiene simetría de orden 4 (girarla 90° la cambia de $Q^H$ a
$Q^V$, dos formas distintas, no la misma). El invariante $\zeta^4$ sería
correcto para objetos con simetría cuadrada real (cajas cuadradas,
$a=\ell$) — en ese caso degenerado, $(a,\ell)$ colapsa a un solo número y
la orientación $o$ deja de tener sentido observacional: **una caja cuadrada
no tiene orientación medible por footprint**, consecuencia directa de que
su grupo de estabilizador bajo $D_4$ es todo $D_4$, no solo $\{1,\sigma r\}$.
Si el catálogo llega a incluir cajas cuadradas, el sistema debe detectar
$a\approx\ell$ y suprimir la variable $o$, no inferirla de ruido.

## 20. La homografía como elemento del grupo de Lie $PGL(3,\mathbb R)$

**Definición 20.1.** $H\in\mathbb R^{3\times3}$ invertible, módulo escala,
es un elemento del grupo proyectivo lineal $PGL(3,\mathbb R)=GL(3,\mathbb
R)/\mathbb R^\times$, que actúa sobre $\mathbb P^2$ y, restringido a la
carta afín $\{[x:y:1]\}$, da la rectificación de la Definición 1.2.

**Proposición 20.2 (Perturbación infinitesimal, álgebra de Lie).** Un error
de calibración pequeño en $H$ se escribe $H_\epsilon=H\exp(\epsilon X)$ con
$X\in\mathfrak{pgl}(3,\mathbb R)$ (el álgebra de Lie, matrices $3\times3$
de traza cero módulo escala) y $\epsilon\to0$. A primer orden,

$$
(u,v)_\epsilon = (u,v) + \epsilon\, J_H(u,v)\,X\, [x,y,1]^\top + O(\epsilon^2),
$$

donde $J_H$ es el jacobiano de $\pi\circ H(\cdot)$ evaluado en el punto.

**Consecuencia práctica.** Esto formaliza por qué $\varepsilon_r$ (el error
geométrico relativo usado en toda la Parte II) no es constante en el ROI:
$J_H$ varía punto a punto (más grande cerca de los bordes del ROI, donde la
proyección homogénea amplifica el error), así que un $\varepsilon_r$ único
para todo el frame es una aproximación de primer orden válida solo cerca
del centro de calibración. Una mejora real (no solo más compleja, sino
correcta) sería $\varepsilon_r(u,v)\propto\|J_H(u,v)\|_{op}$, el error
geométrico escalado por la norma de operador del jacobiano local — más
grande donde $H$ distorsiona más.

**Proposición 20.3 (Por qué $\gamma$ de §7.4 es exactamente esto en 1D).**
El factor de dilatación de área $\gamma(i)=(s(i)/s(i-1))^2$ de la Sección
7.4 es el caso particular de la Proposición 20.2 restringido a la familia de
un parámetro de homografías generadas por altura de cámara — $s(z)$ es la
imagen de un boost proyectivo unidimensional (subgrupo de un parámetro de
$PGL(3,\mathbb R)$ generado por la traslación en la dirección de
profundidad), y $\gamma$ es literalmente el jacobiano al cuadrado de ese
subgrupo actuando sobre área en vez de longitud.

## 21. Teoría espectral del grafo de incidencia (generaliza el Corolario 7.5)

El Corolario 7.5 asumía $\mathcal I$ un **bosque** (sin ciclos) para derivar
$\sigma=k-r$. Sin esa hipótesis, $\sigma$ se relaciona con la topología del
grafo vía su primer número de Betti.

**Definición 21.1 (Primer número de Betti de un grafo).** Para un grafo
(no necesariamente bosque) con $V$ vértices, $E$ aristas y $r$ componentes
conexas, $b_1(\mathcal I)=E-V+r$ — el rango del espacio de ciclos
independientes (dimensión del primer grupo de homología simplicial del
grafo visto como complejo de dimensión 1).

**Teorema 21.2 (Generalización de 7.5/7.6 vía Betti).** Con $V=k+p$
(vértices arriba + abajo), $\sigma=E-p$ (Teorema 7.4), y $b_1=E-(k+p)+r$:

$$
\sigma = b_1(\mathcal I) + k - r.
$$

*Demostración.* Sustituyendo $E=b_1+(k+p)-r$ en $\sigma=E-p$:
$\sigma=b_1+(k+p)-r-p=b_1+k-r$. $\blacksquare$

**Corolario 21.3 (El Corolario 7.5 es el caso $b_1=0$).** Un grafo es
bosque sii $b_1=0$ — recuperando $\sigma=k-r$ exactamente.

**Interpretación física.** $b_1(\mathcal I)>0$ significa que existe al
menos una caja de abajo compartida por 3 o más de arriba de forma que se
cierra un ciclo en el grafo de contacto — geométricamente, un patrón de
apilado más denso que el "ladrillo simple" (donde cada caja de arriba pisa
exactamente 2, sin compartir triples). $b_1$ cuenta exactamente cuántos
"contactos redundantes" de ese tipo hay, y el Teorema 21.2 dice que cada
unidad de $b_1$ añade una unidad a $\sigma$ más allá de lo que predice la
estructura de árbol — **una corrección aditiva exacta, no una
aproximación**, calculable sin enumerar el patrón si ya se conoce $b_1$ del
diseño de apilado (ej. patrones "pinwheel" de la literatura de paletizado
tienen $b_1$ característico distinto de los patrones "ladrillo").

## 22. Formulación variacional unificada: un solo funcional de energía

Las Partes I–III tratan cobertura, balance, hull, capacidad y conservación
de área como pruebas booleanas separadas aplicadas en cascada. Se pueden
unificar en un único problema de optimización con restricciones, del cual
cada criterio anterior es una condición de Karush-Kuhn-Tucker (KKT).

**Definición 22.1 (Funcional de plausibilidad).** Para una configuración
candidata $X=(\mathcal L_0,\dots,\mathcal L_{Z-1})$ (notación de §8.2),
defínase

$$
\mathcal E(X) = \underbrace{\sum_z\lambda_{area}\bigl(\mu(U_z)-(n-k_z)A\bigr)^2}_{\text{Teorema 7.2, penalización}}
+\underbrace{\sum_C\lambda_{sup}\max(0,\ \tau_{support}-Coverage(C))^2}_{\text{cobertura, Def 5.2/11.1}}
+\underbrace{\sum_C\lambda_{bal}\max(0,\ \phi(C)-\tfrac{\rho_{max}}{\rho_{max}+1})^2}_{\text{balance, Def 3.2/11.3}}
+\underbrace{\sum_C\lambda_{hull}\,d\bigl(\bar c(C),\mathcal P(C)\bigr)^2}_{\text{polígono de soporte, §12}},
$$

donde $d(\cdot,\mathcal P)$ es la distancia (con signo, negativa si dentro)
del centroide al polígono de soporte, y las $\lambda$ son multiplicadores
de peso relativo entre criterios.

**Proposición 22.2 (Las reglas de aceptación de la Parte II son el caso
$\mathcal E=0$ activo).** Cada término de $\mathcal E$ es una penalización
tipo *hinge* (cero cuando la restricción se satisface, positivo si se
viola). $\mathcal E(X)=0$ para todos los términos simultáneamente es
exactamente la conjunción de: Corolario 7.3 (nivel lleno por área),
Definición 5.2/11.3 (cobertura y balance), y Criterio 12.3 (centroide en
hull) — todas las condiciones de aceptación de las Partes I y II, a la vez,
como un único punto factible de $\mathcal E$.

**Observación 22.3 (Por qué esto es más que notación).** Con $\mathcal E$
como funcional único, el problema de reconciliación de §8 (elegir entre
$\Omega(\mathcal D)$) se convierte en

$$
X^\star=\arg\min_{X\in\text{admisibles}}\mathcal E(X),
$$

y el Teorema 8.4 (tricotomía) se reinterpreta: $|\Omega/\!\sim|=1$ sii
$X^\star$ es un mínimo global **estrictamente aislado** (sin otro mínimo
global empatado); $|\Omega/\!\sim|>1$ sii hay múltiples minimizadores
globales con $\mathcal E=0$ — la no-identificabilidad es literalmente
degeneración del mínimo, no solo una definición combinatoria aparte. Esto
abre la puerta a diagnosticar identificabilidad por la **curvatura** de
$\mathcal E$ alrededor de $X^\star$ (el Hessiano, si $X$ se relaja a
variables continuas) en vez de enumerar $\Omega$ combinatoriamente — más
caro por evaluación, pero evita la explosión combinatoria de §8.3 para
casos grandes.

## 23. Generalización bayesiana de la tricotomía de identificabilidad

**Definición 23.1 (Verosimilitud geométrica).** Para cada $X\in\Omega(\mathcal D)$
admisible, defínase $L(X\mid\mathcal D)=\exp(-\mathcal E(X)/T)$ para una
"temperatura" $T>0$ de calibración (análogo a un modelo de Gibbs/Boltzmann
sobre configuraciones).

**Definición 23.2 (Posterior sobre configuraciones).**

$$
P(X\mid\mathcal D)=\frac{L(X\mid\mathcal D)}{\sum_{X'\in\Omega(\mathcal D)}L(X'\mid\mathcal D)}.
$$

**Proposición 23.3 (La tricotomía como límite de temperatura cero).**
Cuando $T\to0^+$, $P(X\mid\mathcal D)$ se concentra en los minimizadores
globales de $\mathcal E$: si hay un único minimizador, $P\to\delta_{X^\star}$
(recupera $|\Omega/\!\sim|=1$, Teorema 8.4); si hay varios empatados, la masa
se reparte uniformemente entre ellos (recupera $|\Omega/\!\sim|>1$); si
$\Omega(\mathcal D)=\varnothing$, $P$ no está definida (recupera
$|\Omega/\!\sim|=0$).

*Justificación.* Estándar de mecánica estadística: el límite de temperatura
cero de una distribución de Gibbs colapsa al conjunto argmin del hamiltoniano,
uniformemente si hay degeneración. $\blacksquare$

**Corolario 23.4 (Por qué esto es útil operativamente, no solo elegante).**
Con $T>0$ finito y calibrado (no el límite), $P(X\mid\mathcal D)$ da algo que
el Teorema 8.4 no da: una **cuantificación de cuán cerca** está el sistema de
ser no-identificable, vía la entropía

$$
H(\mathcal D)=-\sum_X P(X\mid\mathcal D)\log P(X\mid\mathcal D)\ge0.
$$

$H\approx0$: reconstrucción confiablemente única, aunque técnicamente
$|\Omega/\!\sim|$ pudiera ser 1 con un segundo candidato cerca pero peor —
información que la tricotomía binaria descarta. $H$ grande: aunque
$|\Omega/\!\sim|=1$ formalmente (un solo candidato pasa todas las
restricciones duras), hay configuraciones casi-admisibles con energía
similar — señal de alerta de fragilidad que el criterio binario no ve. Esto
es una generalización estrictamente más informativa del mismo teorema, no
un capricho: cuantifica el margen, no solo el veredicto.

## 24. Qué de esta Parte IV es implementable ya y qué es exploratorio

| Sección | Estado |
|---|---|
| §19 (rep. de grupo, $D_4$/$\mathbb Z_4$) | Diagnóstico útil ya: detectar $a\approx\ell$ (cajas cuadradas) y suprimir $o$ — trivial de añadir |
| §20 (Lie/$PGL(3)$, $\varepsilon_r(u,v)$ local) | Implementable: requiere el jacobiano de $H$, ya disponible si $H$ se conoce en forma cerrada |
| §21 (Betti/espectral) | Diagnóstico/análisis offline de patrones de catálogo — no bloquea producción, sirve para caracterizar patrones nuevos antes de calibrarlos |
| §22 (funcional unificado) | Reformulación útil para razonar, cara de evaluar en tiempo real si $X$ es de dimensión alta — mejor como herramienta de diseño/debug que como runtime |
| §23 (bayesiano/entropía) | El más especulativo: útil como métrica de confianza adicional (loggeable), no como reemplazo del criterio duro — requiere calibrar $T$, que no tiene análogo físico directo como sí lo tienen $\tau_{support}$ o $\rho_{max}$ |

**Advertencia explícita, para no inflar esto sin criterio:** ninguna sección
de la Parte IV cambia una sola decisión de aceptar/rechazar del sistema tal
como está calibrado hoy — son reformulaciones que exponen estructura
matemática ya presente, útiles para depurar, diagnosticar patrones nuevos
de catálogo, o justificar teóricamente parámetros que hoy son solo
empíricos. Si el objetivo es producción estable, §19 y §20 son las únicas
con relación costo/beneficio clara; §21–23 son para cuando el sistema ya
esté maduro y quieras entender *por qué* falla en un caso raro, no para
correr en cada frame.

---


Continúa `paletizado_parte4.md`. Cuatro estructuras más, cada una
reemplazando un resultado ya probado por el teorema general del que es
instancia. Numeración desde §25.

---

## 25. Monotonicidad de ocupación como punto fijo en un retículo completo (Knaster–Tarski)

El Axioma 4.2 ("$\chi$ solo va $0\to1$") se enunció como regla operativa.
Es, exactamente, la hipótesis del teorema de punto fijo de Knaster–Tarski.

**Definición 25.1 (Retículo de estados).** Sea $\mathcal X=\{0,1\}^{\mathcal G\times\mathbb N}$
el conjunto de todas las funciones de ocupación posibles, ordenado
puntualmente: $\chi\le\chi'$ sii $\chi(g,z)\le\chi'(g,z)$ para todo $(g,z)$.
$(\mathcal X,\le)$ es un retículo completo (ínfimo/supremo puntuales de
cualquier familia existen trivialmente en $\{0,1\}$).

**Definición 25.2 (Operador de actualización).** Sea $\Delta_t:\mathcal X\to\mathcal X$
el operador que, dado el estado $\chi$ y las detecciones del frame $t$,
produce el nuevo estado tras aplicar matching, confirmación y promoción de
nivel (Secciones 9–15 de `palletizing_counting.md`).

**Axioma 25.3 (Monotonicidad del operador, reformulación de 4.2–4.3).**
$\Delta_t$ es monótono: $\chi\le\chi'\Rightarrow\Delta_t(\chi)\le\Delta_t(\chi')$,
y **extensivo**: $\chi\le\Delta_t(\chi)$ siempre (el operador nunca reduce
ocupación).

**Teorema 25.4 (Knaster–Tarski, especializado).** La secuencia
$\chi_0\le\Delta_1(\chi_0)\le\Delta_2(\Delta_1(\chi_0))\le\cdots$ es una
cadena creciente en el retículo completo $\mathcal X$, y por tanto:

(a) converge (en el sentido de que $\chi_t(g,z)$ se estabiliza en 1 para
cada $(g,z)$ finalmente ocupado, ya que $\{0,1\}$ no admite cadenas
infinitas estrictamente crecientes por coordenada);

(b) el límite $\chi_\infty=\bigvee_t\chi_t$ es el menor punto fijo de
$\Delta_\infty$ (el operador límite) que domina a $\chi_0$ — es decir, la
paleta completa reconstruida es, formalmente, el mínimo punto fijo de
Knaster–Tarski del sistema, exactamente como la semántica de programas
recursivos en dominios de Scott usa el mismo teorema para garantizar que un
bucle monótono termina en un resultado bien definido.

*Demostración.* Extensividad + retículo completo con cadenas finitas por
coordenada da (a) directo. (b) es la construcción estándar de Kleene para el
mínimo punto fijo de un operador monótono continuo en un dcpo (dominio
completo dirigido), que $\{0,1\}^{\mathcal G\times\mathbb N}$ satisface
trivialmente al ser producto de retículos finitos. $\blacksquare$

**Por qué importa más allá de la estética.** Esto es la prueba formal de
que "$total=initial+placed$ y basta con nunca restar" (comentario informal
tras el Axioma 4.3) **no es solo una conveniencia de implementación** — es
consecuencia necesaria de que el sistema completo es un cálculo de punto
fijo monótono, la misma clase matemática que garantiza terminación en
teoría de dominios (Scott) y en análisis de programas por interpretación
abstracta. Si en algún refactor se introduce una operación que rompe
monotonicidad (por ejemplo, "corregir" una identidad bajando su nivel), se
pierde la garantía de terminación/estabilidad del Teorema 25.4, no solo una
propiedad deseable — el sistema deja de tener la estructura que garantiza
que $\chi_t$ se estabilice en absoluto.

---

## 26. Identificabilidad como cohomología de Čech (generaliza el Teorema 8.4)

El problema de reconciliación (§8) cubre la paleta con "vistas locales" (una
hipótesis de nivel por frontera $N_i\to N_{i+1}$) que deben pegarse en una
solución global. Esto es exactamente la estructura de un problema de
gavillas (sheaves): secciones locales compatibles en las intersecciones que
sí o no se extienden a una sección global.

**Definición 26.1 (Cubrimiento por fronteras).** Sea $\mathcal U=\{U_i\}$
el cubrimiento de la paleta por vecindades de cada frontera
$N_i\to N_{i+1}$ (§18 del documento cliente, secuencia de fronteras
consecutivas). Sobre cada $U_i$, una hipótesis local es una asignación de
nivel a las cajas observadas en esa frontera, satisfaciendo (A)–(G) de la
Definición 8.2 restringidas a $U_i$.

**Definición 26.2 (Gavilla de hipótesis compatibles).** Sea $\mathcal F$ la
gavilla que asigna a cada abierto $U\subseteq$ paleta el conjunto de
hipótesis locales admisibles sobre $U$, con restricción dada por
recortar la hipótesis a subconjuntos. Dos hipótesis en $U_i,U_j$ son
compatibles en la superposición $U_i\cap U_j$ si asignan el mismo nivel a
toda caja fragmento compartida (exactamente la condición "los fragmentos
consumidos al reconstruir $N_i$ se marcan para que jamás se reutilicen al
resolver $N_{i+1}$" de §18).

**Teorema 26.3 (Obstrucción de Čech).** Existe una sección global de
$\mathcal F$ sobre toda la paleta (i.e., una reconstrucción completa
consistente) si y solo si la clase de cohomología de obstrucción en
$\check H^1(\mathcal U,\mathcal F)$ (calculada a partir de las
incompatibilidades por pares en las superposiciones) se anula.

*Justificación (estándar de teoría de gavillas).* La secuencia exacta corta
de Čech $0\to H^0(\mathcal U,\mathcal F)\to\prod_i\mathcal F(U_i)\to\prod_{i,j}\mathcal F(U_i\cap U_j)$
tiene como núcleo del segundo mapa (diferencia de restricciones en
solapes) exactamente las familias de secciones locales que coinciden en las
intersecciones — es decir, las que se pegan. $H^0=0$ obstruido por una
clase no nula en el cociente es la formalización estándar de "localmente
consistente, globalmente no pegable". $\blacksquare$

**Corolario 26.4 (El Teorema 8.4 es el caso de complejo simplicial 1D).**
Con fronteras consecutivas en secuencia lineal $N_0\to N_1\to\cdots\to N_{Z-1}$
(sin ramificación — cada frontera se resuelve antes de pasar a la
siguiente, como especifica §18), el cubrimiento $\mathcal U$ es una cadena
de intervalos superpuestos en $\mathbb R$, cuyo primer grupo de cohomología
de Čech es siempre trivial cuando cada superposición par a par es
compatible — recuperando exactamente que "cada frontera se resuelve de
forma independiente, secuencialmente, sin reabrir la anterior" (la
propiedad operativa ya implementada en `_reconcile_initial_layers`) es
suficiente para garantizar pegado global, sin necesitar el aparato
cohomológico completo — el caso 1D lineal es donde $H^1$ se anula
automáticamente por la estructura de cadena, que es precisamente por qué
el algoritmo secuencial de §18 funciona sin enumerar $\Omega$ completo.

**Por qué esto es más que traducir vocabulario.** Si el sistema alguna vez
necesita **más de una cámara** viendo la misma paleta desde ángulos
distintos (cubrimiento no lineal, con superposiciones que no forman una
cadena simple), el Teorema 26.3 dice exactamente qué se necesita verificar:
ya no basta con resolver fronteras en secuencia — hay que verificar la
clase de cohomología del cubrimiento completo, porque con más de dos vistas
solapadas simultáneas la triangulación de consistencia deja de ser
automáticamente trivial (aparecen ciclos de compatibilidad de 3 vistas que
pueden ser localmente consistentes por pares y globalmente inconsistentes
— el fenómeno clásico de gavillas no triviales).

---

## 27. Matching uno a uno como transporte óptimo discreto (generaliza §10)

El matching de la Sección 10 (`palletizing_counting.md`) usa $R_{min}$ +
distancia de centroides como criterio de emparejamiento uno a uno. Esto es
la instancia discreta y balanceada del problema de Monge–Kantorovich.

**Definición 27.1 (Problema de transporte).** Sean $\mu=\sum_i\delta_{c_i}$
(medida empírica sobre centroides de detecciones) y $\nu=\sum_j\delta_{f_j}$
(centroides de identidades confirmadas), con costo
$c(i,j)=1-R_{min}(C_i,F_j)$ (costo bajo si el solape es alto). El problema
de transporte óptimo balanceado es

$$
\pi^\star=\arg\min_{\pi\in\Pi(\mu,\nu)}\sum_{i,j}\pi_{ij}\,c(i,j),
$$

con $\Pi(\mu,\nu)$ el conjunto de matrices de transporte con marginales
$\mu,\nu$ (en el caso $N=M$ y transporte 0/1, es exactamente una matriz de
permutación).

**Proposición 27.2 (El matching uno a uno de §10 es el caso 0/1 del
transporte óptimo).** Restringir $\Pi(\mu,\nu)$ a matrices de permutación
(en vez de acoplamientos fraccionarios generales) recupera exactamente el
problema de asignación bipartita resuelto por el algoritmo Húngaro — el
transporte óptimo balanceado sobre medidas puntuales con soporte 0/1 es,
por el teorema de Birkhoff–von Neumann, equivalente a la asignación
bipartita: los extremos del politopo de Birkhoff (matrices doblemente
estocásticas) son exactamente las matrices de permutación, así que el
óptimo de un funcional lineal sobre ese politopo siempre se alcanza en un
vértice — una permutación.

**Observación 27.3 (Por qué la generalización SÍ importa aquí, no solo
elegancia).** El caso $N\ne M$ (más detecciones que identidades, o
viceversa — exactamente lo que pasa en cada frame real: candidatas nuevas,
confirmadas que no se redetectan por oclusión) es **transporte no
balanceado** (unbalanced optimal transport), que tiene solución vía
relajación con términos de penalización por masa no transportada
(formulación de Kantorovich–Rubinstein relajada, o Sinkhorn no balanceado).
Esto formaliza correctamente lo que hoy el sistema maneja con reglas ad-hoc
("candidata sin match → nueva", "confirmada sin match → sigue existiendo,
$\chi$ no baja") — son exactamente los términos de penalización por masa
sobrante en transporte no balanceado, con el costo de "no matchear" fijado
implícitamente por las reglas del Axioma 4.2/4.3 en vez de explícito como
un hiperparámetro $\tau_{unbalanced}$.

---

## 28. Homología persistente para la evolución temporal de la oclusión (generaliza §7 a una secuencia)

El §7 del documento base trata un frame estático. La secuencia completa de
frames de bootstrap (§18 del cliente) es una filtración, y su topología a
través del tiempo se puede rastrear con homología persistente en vez de
solo áreas acumuladas.

**Definición 28.1 (Complejo de oclusión por frame).** Para el frame $t$,
constrúyase el complejo simplicial $K_t$ con un vértice por caja confirmada
visible y una arista $(j,j')$ si sus footprints están a distancia menor a
un radio $\epsilon$ de vecindad de rejilla — el complejo de Vietoris–Rips
del conjunto de centroides visibles en $t$.

**Definición 28.2 (Filtración temporal).** $K_1\subseteq K_2\subseteq
\cdots$ NO es cierto en general (cajas pueden aparecer sin orden de
inclusión estricta), pero sí lo es la filtración por **radio de vecindad**
$\epsilon$ dentro de un frame fijo — el objeto correcto es la homología
persistente 2-paramétrica $(t,\epsilon)$: para cada frame $t$, un
diagrama de persistencia estándar en $\epsilon$; comparando diagramas entre
frames consecutivos, la distancia de bottleneck $d_B(K_t,K_{t+1})$ mide
cuánto cambió la estructura topológica de "qué está agrupado con qué" de un
frame al siguiente.

**Proposición 28.3 (Uso diagnóstico).** Un salto grande en
$d_B(K_t,K_{t+1})$ sin cambio correspondiente en el conteo total es señal de
reordenamiento espurio de identidades (posible bug de matching) más
sensible que monitorear solo $total(t)$, porque $total$ puede permanecer
exactamente igual mientras la topología de agrupamiento cambia
completamente (ej. dos identidades intercambiadas por error de matching —
invisible en el conteo, visible en $d_B$).

**Por qué esto es genuinamente el borde de lo razonable para producción.**
A diferencia de §25–27 (que reformulan cosas ya implementadas o casi
implementables), esta sección es diagnóstico de investigación pura: calcular
diagramas de persistencia por frame tiene costo no trivial y su valor es
para *auditoría offline* de una corrida completa (detectar corrupciones de
matching post-hoc), nunca para decisión en tiempo real.

---

## 29. Balance final: mapa de rendimiento matemático vs. costo de implementación

| Sección | Qué generaliza | Rendimiento real | Costo |
|---|---|---|---|
| §25 (Knaster–Tarski) | Axioma 4.2/4.3 | Alto — es la prueba formal de por qué el sistema no puede "romperse" si se preserva monotonicidad; guía de qué NO tocar en refactors | Cero (no cambia código, cambia por qué confías en el código) |
| §26 (cohomología de Čech) | Teorema 8.4, §18 | Alto SOLO si se agregan múltiples cámaras/vistas; nulo si la paleta sigue viéndose desde una sola cámara cenital | Cero hoy, relevante el día que haya multi-cámara |
| §27 (transporte óptimo) | Matching §10 | Medio — el caso balanceado ya está bien resuelto (Húngaro); la versión no balanceada formaliza reglas que hoy son ad-hoc, útil si el ad-hoc empieza a fallar en casos raros | Medio si se implementa Sinkhorn no balanceado |
| §28 (homología persistente) | §7 (estático) → temporal | Bajo para producción, alto para auditoría offline de bugs de matching difíciles de reproducir | Alto, y no es tiempo real |

**Ya en el límite razonable.** Con esto se cierra la cadena: cada capa
matemática de las Partes IV–V explica *por qué* algo que el sistema ya hace
funciona (§25, ya implementado, ahora con garantía formal), *cuándo* algo
adicional se volvería necesario (§26, si aparece multi-cámara; §27, si el
ad-hoc de matching empieza a fallar), o es honestamente investigación para
depurar, no para producción (§28). Seguir apilando estructura después de
este punto — categorías derivadas, teoría de haces superior, homotopía —
deja de tener retorno: el problema físico es 2D, estático por frame salvo
una racha corta de 3 (§9 base), y de un tamaño acotado por 15–25 cajas. La
sofisticación matemática correcta para *ese* problema ya está cubierta;
más allá es matemática por sí misma, no por el paletizado.

---

## 30. Síntesis operativa final: por qué el polígono de soporte elimina la necesidad de $K$

Recapitulando el hilo de todo el documento en una sola decisión de diseño:

- El criterio original (Sec. 5, top-2 fijo) necesitaba $\tau_{support}$ y $\rho_{max}$, y asum?a $m_t\equiv 2$; esto falla con tama?os mixtos.
- La Mejora A (§11, $K$ dinámico) generaliza a cualquier número de soportes, pero **introduce un parámetro nuevo, $K_{max}$**, que hay que calibrar por catálogo (Corolario 11.6) — sigue siendo "elegir un número".
- La Mejora B (§12, polígono de soporte) no fija un techo $K_{max}$: el hull usa todos los contactos válidos. Sin embargo, §12.4 demuestra que el hull no distingue por sí solo un apoyo único estable de un entrelazado; se conserva explícitamente el mínimo de dos soportes independientes.

**Por eso el orden de autoridad correcto es:** usar §12 (polígono de soporte) como único criterio en el camino feliz; caer a §11 ($K/\phi$) *solo* cuando el hull es degenerado (Definición 13.1) — ahí, y solo ahí, $K$ vuelve a ser necesario, porque el test geométrico deja de estar bien condicionado. $K$ no desaparece del sistema: se convierte en un mecanismo de respaldo para el 5–10% de casos con ruido severo, no en el criterio que corre en cada caja de cada frame.

Con esto el documento queda cerrado: Parte I es lo que ya corre en producción sin tocar; Parte II es la mejora con criterio de cuándo usar cada rama; Partes III–V son, en orden decreciente de utilidad inmediata, todo lo que sostiene matemáticamente por qué esa jerarquía es correcta y hasta dónde vale la pena llevarla.
