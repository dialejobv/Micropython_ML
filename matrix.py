# matrix.py
import random

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def from_list(lst):
        m = Matrix(len(lst), 1)
        for i in range(len(lst)):
            m.data[i][0] = lst[i]
        return m
    
    def to_list(self):
        return [item for row in self.data for item in row]
    
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(-1, 1)
    
    def add(self, other):
        if isinstance(other, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += other.data[i][j]
        else:  # Escalar
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += other
        return self
    
    @staticmethod
    def subtract(a, b):
        result = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result
    
    @staticmethod
    def multiply(a, b):
        if a.cols != b.rows:
            raise ValueError("Dimensiones incompatibles para multiplicación")
        result = Matrix(a.rows, b.cols)
        for i in range(a.rows):
            for k in range(a.cols):
                for j in range(b.cols):
                    result.data[i][j] += a.data[i][k] * b.data[k][j]
        return result
    
    def hadamard(self, other):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] *= other.data[i][j]
        return self
    
    @staticmethod
    def transpose(m):
        result = Matrix(m.cols, m.rows)
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[j][i] = m.data[i][j]
        return result
    
    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])
        return self
    
    @staticmethod
    def map_static(m, func):
        result = Matrix(m.rows, m.cols)
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[i][j] = func(m.data[i][j])
        return result
    
    def scale(self, scalar):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] *= scalar
        return self