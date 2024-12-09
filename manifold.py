import numpy as np

class Manifold:

    def coords(self, x):
        return self.chart(self.chartSelect(x), x)
    
    def tangentCoords(self, x):
        return self.frame(self.chartSelect(x), x)

    def tangentCoordsInv(self, x):
        return self.frameInv(self.chartSelect(x), x)
    
    def orthoCoords(self, x):
        return self.orthoFrame(self.chartSelect(x), x)

    def orthoCoordsInv(self, x):
        return self.orthoFrameInv(self.chartSelect(x), x)

    def transportMatrix(self, x, v):
        transport = self.transport(x, v)
        coordsInv = self.orthoCoordsInv(x)
        if len(v.shape) == 1:
            coords = self.orthoCoords(self.exp(x, v))
        elif len(v.shape) == 2:
            coordsInv = np.expand_dims(coordsInv, axis=0)
            coords = np.stack([self.orthoCoords(self.exp(x, v1)) for v1 in v])
        return np.matmul(np.matmul(coordsInv, transport), coords)

class Sphere(Manifold):
    
    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def chartSelect(self, x):
        poles = np.concatenate((np.eye(self.n+1), -np.eye(self.n+1)))
        return np.argmin(np.linalg.norm(x-poles, axis=1))
    
    def chart(self, i, x):
        if i < self.n+1:
            return np.delete(x, i)/(1+x[i])
        else:
            i -= self.n+1
            return np.delete(x, i)/(1-x[i])
    
    def chartInv(self, i, y):
        a = 2/(1+np.linalg.norm(y)**2)
        J = a*y
        if i < self.n+1:
            return np.insert(J, i, -1+a)
        else:
            i -= self.n+1
            return np.insert(J, i, 1-a)
    
    def frame(self, i, x):
        J = self.frameInv(i, x)
        return np.transpose(J/np.linalg.norm(J, axis=1)**2)
    
    def frameInv(self, i, x):
        y = self.chart(i, x)
        a = 2/(1+np.linalg.norm(y)**2)
        J = a*np.eye(self.n) - a**2*np.tensordot(y, y, axes=0)
        if i < self.n+1:
            return np.insert(J, i, -a**2*y, axis=1)
        else:
            i -= self.n+1
            return np.insert(J, i, a**2*y, axis=1)
    
    def orthoFrame(self, i, x):
        J = self.orthoFrameInv(i, x)
        return np.transpose(J)

    def orthoFrameInv(self, i, x):
        y = self.chart(i, x)
        a = 2/(1+np.linalg.norm(y)**2)
        J = self.frameInv(i, x)/a
        if (i < self.n+1 and i%2 == 0) or (i >= self.n+1 and (i-self.n-1)%2 == 1):
            return J
        else:
            diag = np.diag([1]*(self.n-1)+[-1])
            return np.matmul(diag, J)

    def exp(self, x, v):
        theta = np.linalg.norm(v, axis=-1, keepdims=True)+1e-15
        v_ = v/theta
        return np.cos(theta)*x + np.sin(theta)*v_
    
    def log(self, x, x1):
        theta = np.arccos(np.clip(np.matmul(x1, x), -1, 1))+1e-15
        theta = np.expand_dims(theta, axis=-1)
        return theta/np.sin(theta) * (x1 - np.cos(theta)*x)
    
    def transport(self, x, v):
        theta = np.linalg.norm(v, axis=-1, keepdims=True)+1e-15
        v_ = v/theta
        theta = np.expand_dims(theta, axis=-1)
        x1, x2, v1, v2 = np.expand_dims(x, axis=-1), np.expand_dims(x, axis=-2), np.expand_dims(v_, axis=-1), np.expand_dims(v_, axis=-2)
        xx, vv, xv, vx = x1*x2, v1*v2, x1*v2, v1*x2
        return np.eye(self.n+1) - (1-np.cos(theta))*(xx+vv) + np.sin(theta)*(xv-vx)

class EuclideanSpace(Manifold):
    
    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def chartSelect(self, x):
        return 0
    
    def chart(self, i, x):
        return x
    
    def chartInv(self, i, y):
        return y
    
    def frame(self, i, x):
        return np.eye(self.n)
    
    def frameInv(self, i, x):
        return np.eye(self.n)
    
    def orthoFrame(self, i, x):
        return np.eye(self.n)

    def orthoFrameInv(self, i, x):
        return np.eye(self.n)

    def exp(self, x, v):
        return x+v
    
    def log(self, x, x1):
        return x1-x
    
    def transport(self, x, v):
        if len(v.shape) == 2:
            return np.eye(self.n)
        elif len(v.shape) == 3:
            return np.tile(np.expand_dims(np.eye(self.n), axis=0), (x.shape[0], 1, 1))