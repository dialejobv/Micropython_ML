# Micropython_ML
Desarrollo sencillo de una red forward-back propagation en micropython orientada directamente con el archivo visualizado "pop_corn" del repositorio generado por el [profesor Gerardo Muñoz](https://github.com/GerardoMunoz/pop_corn).

Para este caso utilicé dos clases la primera la denominé matrix que es una versión que revisé del repositorio del profesor Gerardo Muñoz y posteriormente la adapté a las necesidades de mi proyecto, la segunda clase la denominé Red Neuronal que es un desarrollo que generé partiendo de las funciones definidas de la clase matrix. La clase matrix la visualicé de la siguiente manera:
1. Clase Matrix:
   - Visualicé el manejo de operaciones esenciales como la multiplicación, suma, transposición y producto Hadamard.
   - Desarrollé el anexo de funciones de mapeo para aplicar funciones como sigmoid.
   - Tuve presente from_list() y to_list() para la conversión entre formatos y randomize() para la ininiaclización de pesos.
2. Para acoplar el componente de Matrix en el caso de la red neuronal generé:
   - Inicialización: Los pesos y sesgos los convertí a objetos e la clase Matrix
   - Propagación hacia adelante:
     - Operaciones matriciales reemplazan cálculos con listas.
     - Matrix.multiply() para producto matricial.
     - map() aplica la función de activación.
   - Backpropagation:
     - Cálculo de errores mediante operaciones matriciales.
     - Uso de transposición para propagación de errores.
     - Actualización de pesos mediante operaciones matriciales.
     


