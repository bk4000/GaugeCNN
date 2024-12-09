import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class Mesh2D:

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.edges = set()
        self.faces = set()
        for f in faces:
            for i in range(len(f)-1):
                self.edges.add(tuple(sorted((f[i], f[i+1]))))
            self.edges.add(tuple(sorted((f[-1], f[0]))))
            self.faces.add(f)
    
    def __len__(self):
        return len(self.vertices)
    
    def __getitem__(self, i):
        return self.vertices[i]

    def graphNeighbor(self, max_dist):
        adj = np.zeros((len(self), len(self)), dtype=np.uint32)
        for e in self.edges:
            adj[e[0], e[1]] = 1
            adj[e[1], e[0]] = 1
        sum = np.eye(len(self), dtype=np.uint32)
        for _ in range(max_dist):
            sum = np.matmul(sum, adj) + np.eye(len(self), dtype=np.uint32)
        neighbor = (sum > 0).astype(np.uint8)
        return neighbor
    
    def toSphereMesh(self):
        return SphereMesh(self.vertices, self.faces)
    
    def toPlaneMesh(self):
        return PlaneMesh(self.vertices, self.faces)

class SphereMesh(Mesh2D):

    def __init__(self, vertices, faces):
        vertices = vertices/np.linalg.norm(vertices, axis=-1, keepdims=True)
        super().__init__(vertices, faces)
    
    def geodesicNeighbor(self, max_dist):
        dist = np.linalg.norm(np.expand_dims(self.vertices, axis=1) - np.expand_dims(self.vertices, axis=0), axis=-1)
        neighbor = (dist <= 2*np.sin(max_dist/2)).astype(np.uint8)
        return neighbor
    
    def samplingCoords(self):
        phi = 2/np.pi * np.arccos(np.clip(self.vertices[:, 2], -1, 1)) - 1
        theta = 1/np.pi * np.arctan2(self.vertices[:, 1], self.vertices[:, 0]) - 1
        return np.expand_dims(np.stack((phi, theta), axis=-1), axis=(0, 1))

class PlaneMesh(Mesh2D):

    def __init__(self, vertices, faces):
        super().__init__(vertices, faces)
    
    def geodesicNeighbor(self, max_dist):
        dist = np.linalg.norm(np.expand_dims(self.vertices, axis=1) - np.expand_dims(self.vertices, axis=0), axis=-1)
        neighbor = (dist <= max_dist).astype(np.uint8)
        return neighbor
    
    def samplingCoords(self):
        return np.expand_dims(self.vertices/2-1, axis=(0, 1))

class TriangularMesh(Mesh2D):

    def subdivision(self, n):
        vertices = list(self.vertices)
        faces = []
        edge_ind = {}
        i = len(vertices)

        def label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, f_ind, n, j, k):
            if j == 0 and k == 0:
                return f[0]
            elif j == n and k == 0:
                return f[1]
            elif j == 0 and k == n:
                return f[2]
            elif k == 0:
                if o0:
                    return e0_ind + j-1
                else:
                    return e0_ind + n-j-1
            elif j+k == n:
                if o1:
                    return e1_ind + k-1
                else:
                    return e1_ind + n-k-1
            elif j == 0:
                if o2:
                    return e2_ind + n-k-1
                else:
                    return e2_ind + k-1
            else:
                return f_ind + (k-1)*(2*n-k-2)//2 + j-1
        
        for e in sorted(self.edges):
            for j in range(1, n):
                v = (vertices[e[0]]*(n-j) + vertices[e[1]]*j) / n
                vertices.append(v) # append subdivision points in the edge interior
            edge_ind[e] = i
            i += n-1
        
        for f in sorted(self.faces):
            o0 = (f[0], f[1]) in edge_ind
            o1 = (f[1], f[2]) in edge_ind
            o2 = (f[2], f[0]) in edge_ind
            e0_ind = edge_ind[(f[0], f[1])] if o0 else edge_ind[(f[1], f[0])]
            e1_ind = edge_ind[(f[1], f[2])] if o1 else edge_ind[(f[2], f[1])]
            e2_ind = edge_ind[(f[2], f[0])] if o2 else edge_ind[(f[0], f[2])]
            for k in range(1, n):
                for j in range(1, n-k):
                    v = (vertices[f[0]]*(n-j-k) + vertices[f[1]]*j + vertices[f[2]]*k) / n
                    vertices.append(v) # append subdivision points in the face interior
            for k in range(0, n):
                for j in range(0, n-k):
                    v0_ind = label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, i, n, j, k)
                    v1_ind = label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, i, n, j+1, k)
                    v2_ind = label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, i, n, j, k+1)
                    faces.append((v0_ind, v1_ind, v2_ind)) # append subdivided faces
            for k in range(1, n):
                for j in range(1, n-k+1):
                    v0_ind = label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, i, n, j, k-1)
                    v1_ind = label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, i, n, j-1, k)
                    v2_ind = label(f, o0, o1, o2, e0_ind, e1_ind, e2_ind, i, n, j, k)
                    faces.append((v0_ind, v1_ind, v2_ind)) # append subdivided faces that are upside-down
            i += (n-1)*(n-2)//2
        
        vertices = np.stack(vertices)
        return TriangularMesh(vertices, faces)

class SquareMesh(Mesh2D):

    def subdivision(self, n):
        vertices = list(self.vertices)
        faces = []
        edge_ind = {}
        i = len(vertices)

        def label(f, o0, o1, o2, o3, e0_ind, e1_ind, e2_ind, e3_ind, f_ind, n, j, k):
            if j == 0 and k == 0:
                return f[0]
            elif j == n and k == 0:
                return f[1]
            elif j == n and k == n:
                return f[2]
            elif j == 0 and k == n:
                return f[3]
            elif k == 0:
                if o0:
                    return e0_ind + j-1
                else:
                    return e0_ind + n-j-1
            elif j == n:
                if o1:
                    return e1_ind + k-1
                else:
                    return e1_ind + n-k-1
            elif k == n:
                if o2:
                    return e2_ind + n-j-1
                else:
                    return e2_ind + j-1
            elif j == 0:
                if o3:
                    return e3_ind + n-k-1
                else:
                    return e3_ind + k-1
            else:
                return f_ind + (k-1)*(n-1) + j-1
        
        for e in sorted(self.edges):
            for j in range(1, n):
                v = (vertices[e[0]]*(n-j) + vertices[e[1]]*j) / n
                vertices.append(v)
            edge_ind[e] = i
            i += n-1
        
        for f in sorted(self.faces):
            o0 = (f[0], f[1]) in edge_ind
            o1 = (f[1], f[2]) in edge_ind
            o2 = (f[2], f[3]) in edge_ind
            o3 = (f[3], f[0]) in edge_ind
            e0_ind = edge_ind[(f[0], f[1])] if o0 else edge_ind[(f[1], f[0])]
            e1_ind = edge_ind[(f[1], f[2])] if o1 else edge_ind[(f[2], f[1])]
            e2_ind = edge_ind[(f[2], f[3])] if o2 else edge_ind[(f[3], f[2])]
            e3_ind = edge_ind[(f[3], f[0])] if o3 else edge_ind[(f[0], f[3])]
            for k in range(1, n):
                for j in range(1, n):
                    v = (vertices[f[0]]*(n-j)*(n-k) + vertices[f[1]]*j*(n-k) + vertices[f[2]]*j*k + vertices[f[3]]*(n-j)*k) / (n**2)
                    vertices.append(v)
            for k in range(0, n):
                for j in range(0, n):
                    v0_ind = label(f, o0, o1, o2, o3, e0_ind, e1_ind, e2_ind, e3_ind, i, n, j, k)
                    v1_ind = label(f, o0, o1, o2, o3, e0_ind, e1_ind, e2_ind, e3_ind, i, n, j+1, k)
                    v2_ind = label(f, o0, o1, o2, o3, e0_ind, e1_ind, e2_ind, e3_ind, i, n, j+1, k+1)
                    v3_ind = label(f, o0, o1, o2, o3, e0_ind, e1_ind, e2_ind, e3_ind, i, n, j, k+1)
                    faces.append((v0_ind, v1_ind, v2_ind, v3_ind))
            i += (n-1)**2
        
        vertices = np.stack(vertices)
        return SquareMesh(vertices, faces)

def IcosahedronGrid(n):
    vertices = []
    ico_cos, ico_sin = 1/5**0.5, 2/5**0.5
    vertices.append(np.array([1, 0, 0], dtype=np.float32))
    for i in range(5):
        theta = 2*i*np.pi/5
        vertices.append(np.array([ico_cos, ico_sin*np.cos(theta), ico_sin*np.sin(theta)], dtype=np.float32))
    for i in range(5):
        theta = (2*i+1)*np.pi/5
        vertices.append(np.array([-ico_cos, ico_sin*np.cos(theta), ico_sin*np.sin(theta)], dtype=np.float32))
    vertices.append(np.array([-1, 0, 0], dtype=np.float32))
    vertices = np.stack(vertices)

    faces = []
    for i in range(5):
        faces.append((0, i+1, (i+1)%5+1))
        faces.append((i+1, (i+1)%5+1, i+6))
        faces.append((i+1, (i-1)%5+6, i+6))
        faces.append((i+6, (i+1)%5+6, 11))
    
    icosahedron = TriangularMesh(vertices, faces)
    return icosahedron.subdivision(n).toSphereMesh()

def SquareGrid(n):
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    faces = [(0, 1, 2, 3)]
    square = SquareMesh(vertices, faces)
    return square.subdivision(n).toPlaneMesh()

def sampling(x, coords):
    batch_size, num_channel = x.shape[0], x.shape[3]
    grid_size = coords.shape[2]
    x = torch.permute(x, (0, 3, 1, 2))
    coords = np.tile(coords, (batch_size, 1, 1, 1))
    coords = torch.tensor(coords, dtype=torch.float32, requires_grad=False)
    sample = torch.nn.functional.grid_sample(x, coords, mode='bilinear', padding_mode='border', align_corners=False)
    sample = torch.reshape(sample, (batch_size, num_channel, grid_size))
    sample = torch.permute(sample, (2, 0, 1))
    return sample