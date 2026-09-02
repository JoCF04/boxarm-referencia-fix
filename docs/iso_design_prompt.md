# Prompt de dirección visual — visor ISO

Este prompt define la intención visual del gemelo digital. Sirve para revisar
la interfaz o generar referencias conceptuales; la implementación real sigue
siendo Canvas/HTML/CSS y debe conservar la geometría de `SceneState`.

> **Objeto principal:** Un gemelo digital 3D interactivo de una paleta
> industrial con cajas colocadas por un brazo robótico, mostrando con claridad
> cada caja, su celda y su nivel de apilado.
>
> **Estilo visual:** Interfaz industrial premium y minimalista, inspirada en
> los paneles de control de vehículos eléctricos avanzados y centros de
> automatización modernos. Precisa, sobria, tecnológica y funcional; sin
> apariencia de videojuego, sin decoración innecesaria y sin saturación.
>
> **Materiales y texturas:** Cajas con superficies mate, neutras y casi opacas,
> contorno oscuro de separación y variaciones claras de luminosidad por celda.
> El color apagado del nivel aparece solo como acento en las aristas y la
> leyenda, no como una masa de color sobre toda la cara. Las diferencias de
> nivel deben ser inequívocas sin usar un arcoíris intenso. Paleta representada
> como una plataforma oscura con retícula de ingeniería discreta.
>
> **Iluminación:** Iluminación de estudio fría y suave, contraste controlado,
> sombras sutiles y reflejos mínimos. Las aristas activas pueden tener un acento
> cian moderado, nunca un resplandor neón excesivo.
>
> **Fondo o entorno:** Sala de control digital oscura, fondo grafito casi negro,
> cuadrícula técnica tenue, paneles translúcidos de bajo contraste y suficiente
> espacio negativo para que la carga sea el foco.
>
> **Interacción y datos:** Arrastre directo y natural para orbitar, rueda para
> inclinar, presets ISO/frontal/lateral/planta, estado de conexión, cámara,
> azimut y elevación. HUD compacto con total, inventario inicial y cajas
> colocadas por el robot. Etiquetas legibles `celdaNnivel`, leyenda por nivel y
> respuesta fluida mediante `requestAnimationFrame`. La vista planta aísla
> automáticamente el nivel ocupado superior para que la proyección no mezcle
> capas con el mismo XY; un control permite recorrer cada nivel o volver a ver
> todas las capas en las vistas tridimensionales.
>
> **Restricciones negativas:** No usar colores chillones por cada caja, glass
> excesivo, gradientes decorativos fuertes, reflejos cromados, tipografía
> futurista ilegible, animaciones constantes, ruido visual ni elementos que
> oculten la posición o el nivel real de las cajas.

## Criterio de aceptación

En menos de dos segundos un operador debe poder identificar:

1. cuántas cajas existen;
2. qué niveles están ocupados;
3. qué caja corresponde a cada etiqueta;
4. desde qué cámara llega el estado;
5. si la geometría está sincronizada.
