# ann.py
import math
from matrix import Matrix  # Importar la clase Matrix

class RedNeuronal:
    def __init__(self, entrada, oculta, salida):
        # Configuración de capas
        self.entrada = entrada
        self.oculta = oculta
        self.salida = salida
        
        # Inicialización de pesos con matrices
        self.pesos_oculta = Matrix(oculta, entrada)
        self.pesos_oculta.randomize()
        
        self.pesos_salida = Matrix(salida, oculta)
        self.pesos_salida.randomize()
        
        # Inicialización de sesgos
        self.sesgo_oculta = Matrix(oculta, 1)
        self.sesgo_oculta.randomize()
        
        self.sesgo_salida = Matrix(salida, 1)
        self.sesgo_salida.randomize()
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def derivada_sigmoid(x):
        return x * (1 - x)
    
    def propagacion_adelante(self, entradas):
        # Convertir entrada a matriz
        self.entrada_mat = Matrix.from_list(entradas)
        
        # Capa oculta
        self.salida_oculta = Matrix.multiply(self.pesos_oculta, self.entrada_mat)
        self.salida_oculta.add(self.sesgo_oculta)
        self.salida_oculta.map(self.sigmoid)
        
        # Capa salida
        self.salida_final = Matrix.multiply(self.pesos_salida, self.salida_oculta)
        self.salida_final.add(self.sesgo_salida)
        self.salida_final.map(self.sigmoid)
        
        return self.salida_final.to_list()
    
    def entrenar(self, entradas, objetivo, tasa_aprendizaje):
        # Paso 1: Propagación hacia adelante
        self.propagacion_adelante(entradas)
        objetivo_mat = Matrix.from_list(objetivo)
        
        # Paso 2: Calcular errores
        error_salida = Matrix.subtract(objetivo_mat, self.salida_final)
        pesos_salida_t = Matrix.transpose(self.pesos_salida)
        error_oculta = Matrix.multiply(pesos_salida_t, error_salida)
        
        # Paso 3: Calcular deltas
        derivada_salida = Matrix.map_static(self.salida_final, self.derivada_sigmoid)
        derivada_salida.hadamard(error_salida)
        derivada_salida.scale(tasa_aprendizaje)
        
        derivada_oculta = Matrix.map_static(self.salida_oculta, self.derivada_sigmoid)
        derivada_oculta.hadamard(error_oculta)
        derivada_oculta.scale(tasa_aprendizaje)
        
        # Paso 4: Actualizar pesos y sesgos
        # Capa salida
        salida_oculta_t = Matrix.transpose(self.salida_oculta)
        delta_salida = Matrix.multiply(derivada_salida, salida_oculta_t)
        self.pesos_salida.add(delta_salida)
        self.sesgo_salida.add(derivada_salida)
        
        # Capa oculta
        entrada_t = Matrix.transpose(self.entrada_mat)
        delta_oculta = Matrix.multiply(derivada_oculta, entrada_t)
        self.pesos_oculta.add(delta_oculta)
        self.sesgo_oculta.add(derivada_oculta)

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar red (2 entradas, 2 ocultas, 1 salida)
    red = RedNeuronal(9, 4, 1)
    
    # Datos de entrenamiento (XOR)
    datos_entrenamiento = [

        ([0.6,	0.2,	0.6,	0.2,	0.2,	0.2,	0.2,	0.2,	0.3],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.6,	0.2,	0.6,	0.6,	0.4,	0.4,	0.4,	0.4,	0.45],	[0]),
        ([0.2,	0.2,	0.4,	0.2,	0.4,	0.2,	0.4,	0.2,	0.275],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.4,	0.2,	0.6,	0.2,	0.2,	0.2,	0.6,	0.2,	0.325],	[0]),
        ([0.2,	0.2,	0.2,	0.6,	0.2,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.4,	0.2,	0.6,	0.4,	0.2,	0.2,	0.4,	0.2,	0.325],	[0]),
        ([0.4,	0.6,	0.8,	0.4,	0.4,	0.6,	0.8,	0.2,	0.525],	[1]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.2,	0.2,	0.2,	0.4,	0.4,	0.4,	0.4,	0.4,	0.325],	[0]),
        ([0.2,	0.2,	0.2,	0.6,	0.2,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.4,	0.2,	0.2,	0.225],	[0]),
        ([0.2,	0.4,	0.4,	0.2,	0.2,	0.2,	0.2,	0.4,	0.275],	[0]),
        ([0.2,	0.2,	0.6,	0.4,	0.2,	0.2,	0.6,	0.2,	0.325],	[0]),
        ([0.2,	0.2,	0.2,	0.6,	0.6,	0.8,	0.2,	0.2,	0.375],	[0]),
        ([0.2,	0.2,	0.4,	0.4,	0.2,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.2,	0.2,	0.2,	0.4,	0.4,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.2,	0.2,	0.6,	0.6,	0.4,	0.4,	0.4,	0.2,	0.375],	[0]),
        ([0.4,	0.2,	0.4,	0.4,	0.6,	0.2,	0.6,	0.4,	0.4],	[0]),
        ([0.6,	0.6,	0.6,	0.6,	0.6,	0.6,	0.6,	0.2,	0.55],	[1]),
        ([0.4,	0.2,	0.2,	0.6,	0.6,	0.4,	0.2,	0.2,	0.35],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.4,	0.6,	0.4,	0.6,	0.2,	0.4,	0.2,	0.2,	0.375],	[0]),
        ([0.2,	0.2,	0.4,	0.4,	0.2,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.2,	0.2,	0.4,	0.2,	0.2,	0.2,	0.4,	0.4,	0.275],	[0]),
        ([0.2,	0.2,	0.4,	0.2,	0.2,	0.2,	0.4,	0.2,	0.25],	[0]),
        ([0.6,	0.4,	0.8,	0.8,	0.4,	0.4,	0.4,	0.4,	0.525],	[0]),
        ([0.2,	0.2,	0.6,	0.6,	0.2,	0.2,	0.6,	0.2,	0.35],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.2,	0.2,	0.4,	0.2,	0.2,	0.2,	0.2,	0.2,	0.225],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.4,	0.2,	0.4,	0.4,	0.2,	0.2,	0.4,	0.2,	0.3],	[0]),
        ([0.4,	0.4,	0.6,	0.4,	0.2,	0.2,	0.2,	0.2,	0.325],	[0]),
        ([0.4,	0.2,	0.6,	0.4,	0.2,	0.2,	0.4,	0.2,	0.325],	[0]),
        ([0.4,	0.6,	0.4,	0.4,	0.2,	0.6,	0.4,	0.2,	0.4],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.4,	0.6,	0.6,	0.8,	0.6,	0.6,	0.8,	0.4,	0.6],	[1]),
        ([0.4,	0.2,	0.2,	0.4,	0.2,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.2,	0.2,	0.2,	0.4,	0.2,	0.2,	0.2,	0.2,	0.225],	[0]),
        ([0.2,	0.4,	0.6,	0.2,	0.2,	0.2,	0.2,	0.2,	0.275],	[0]),
        ([0.4,	0.2,	0.6,	0.6,	0.2,	0.4,	0.2,	0.2,	0.35],	[0]),
        ([0.4,	0.2,	0.2,	0.6,	0.2,	0.2,	0.2,	0.2,	0.275],	[1]),
        ([0.2,	0.2,	0.6,	0.6,	0.2,	0.2,	0.2,	0.2,	0.3],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.2,	0.2,	0.6,	0.4,	0.2,	0.2,	0.2,	0.2,	0.275],	[1]),
        ([0.6,	0.2,	0.2,	0.4,	0.2,	0.2,	0.2,	0.2,	0.275],	[0]),
        ([0.4,	0.4,	0.6,	0.2,	0.2,	0.2,	0.2,	0.2,	0.3],	[0]),
        ([0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2,	0.2],	[0]),
        ([0.6,	0.2,    0.4,	0.6,	0.2,	0.2,	0.6,	0.2,	0.375],	[0]),
        ([0.2,	0.2,    0.2,	0.6,	0.2,	0.2,	0.2,	0.2,	0.25],	[0]),
        ([0.2,	0.2,	0.6,	0.6,	0.2,	0.2,	0.2,	0.2,	0.3],	[0])
    ]
    
    # Entrenamiento
    for epoca in range(10000):
        for entradas, objetivo in datos_entrenamiento:
            red.entrenar(entradas, objetivo, 0.1)
    
    # Probar la red
    for entradas, _ in datos_entrenamiento:
        salida = red.propagacion_adelante(entradas)[0]
        print(f"{entradas} -> {salida:.3f}")