import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
import numpy as np
from manifold import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class KernelPrecomputation:

    def __init__(self, manifold, grid, neighbor, num_theta, num_circle, radius, sigma, bandwidth):
        self.manifold = manifold
        self.len_grid = len(grid)
        self.num_theta = num_theta
        self.num_circle = num_circle
        self.bandwidth = bandwidth

        edge_index = [[], []]
        for i in range(len(grid)):
            for j in range(len(grid)):
                if neighbor[i, j]:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        self.edge_index = np.array(edge_index, dtype=np.uint32)
        self.len_edge = len(self.edge_index[0])
        
        transport = []
        for i in range(len(grid)):
            x = grid[i]
            x_nb = grid[[j for j in range(len(grid)) if neighbor[i, j]]]
            tp = self.manifold.transportMatrix(x, self.manifold.log(x, x_nb))
            cos_tp, sin_tp = tp[:, 0, 0], tp[:, 0, 1]
            theta_tp = np.arctan2(sin_tp, cos_tp)
            transport.append(-theta_tp)
        self.transport = np.concatenate(transport) # shape is E
        
        self.coefficient = []
        for c in range(1, num_circle+1):
            coefficient = []
            theta = np.linspace(0, 2*np.pi, num_theta*c, endpoint=False)
            cos, sin = np.cos(theta), np.sin(theta)
            w = c/num_circle*radius * np.stack((cos, sin), axis=-1)
            for i in range(len(grid)):
                x = grid[i]
                x_nb = grid[[j for j in range(len(grid)) if neighbor[i, j]]]
                v = np.matmul(w, self.manifold.orthoCoordsInv(x))
                y = self.manifold.exp(x, v)
                dist = np.linalg.norm(np.expand_dims(x_nb, axis=1) -np.expand_dims(y, axis=0), axis=-1)
                exp = np.exp(-(dist/sigma)**2)
                coeff = exp/np.sum(exp, axis=0, keepdims=True)
                coefficient.append(coeff)
            self.coefficient.append(np.concatenate(coefficient)/(num_circle*num_theta)) # shape is E x (num_theta*c)
        
        self.basis = {}
        
    def getBasis(self, n_in, n_out):

        if (n_in, n_out) in self.basis:
            return self.basis[(n_in, n_out)]
        print('Computing the kernel basis for ', n_in, '->', n_out)
        
        kernel_list = []
        dim_in = SO2RepDim(n_in)
        dim_out = SO2RepDim(n_out)

        # Kernel basis at the origin
        if n_in != n_out:
            kernel = np.zeros((self.len_edge, dim_in, dim_out, dim_in*dim_out))
        else:
            row, col = self.edge_index
            eq = np.reshape((row == col).astype(np.float32), (self.len_edge, 1, 1, 1))
            if n_in == 0:
                kernel = eq
            elif n_in > 0:
                K0 = np.eye(2)
                K1 = np.array([[0, -1], [1, 0]])
                K2 = np.zeros((2, 2))
                K3 = np.zeros((2, 2))
                K = np.stack((K0, K1, K2, K3), axis=-1)
                kernel = eq*K
        kernel_list.append(kernel) # shape is E x d_in x d_out x (d_in*d_out)

        # Kernel basis at each circle
        for c in range(1, self.num_circle+1):
            if n_in == 0 and n_out == 0:
                kernel = np.reshape(np.sum(self.coefficient[c-1], axis=1), (-1, 1, 1, 1))
            else:
                theta = np.linspace(0, 2*np.pi, self.num_theta*c, endpoint=False)
                if n_in == 0 or n_out == 0:
                    if n_in == 0:
                        theta_n = n_out*theta
                    elif n_out == 0:
                        theta_n = n_in*theta
                    cos_n, sin_n = np.cos(theta_n), np.sin(theta_n)
                    K0 = np.stack((cos_n, sin_n), axis=-1)
                    K1 = np.stack((-sin_n, cos_n), axis=-1)
                    K = np.stack((K0, K1), axis=-1)
                else:
                    theta_m, theta_p = (n_out-n_in)*theta, (n_out+n_in)*theta
                    cos_m, sin_m, cos_p, sin_p = np.cos(theta_m), np.sin(theta_m), np.cos(theta_p), np.sin(theta_p)
                    K0 = np.stack((cos_m, sin_m, -sin_m, cos_m), axis=-1)
                    K1 = np.stack((sin_m, -cos_m, cos_m, sin_m), axis=-1)
                    K2 = np.stack((cos_p, sin_p, sin_p, -cos_p), axis=-1)
                    K3 = np.stack((-sin_p, cos_p, cos_p, sin_p), axis=-1)
                    K = np.stack((K0, K1, K2, K3), axis=-1)
                K = np.reshape(K, (self.num_theta*c, dim_in, dim_out, dim_in*dim_out))
                kernel = np.einsum('ij,jklm->iklm', self.coefficient[c-1], K, optimize='optimal')
                if n_in > 0:
                    theta_tp = n_in * self.transport
                    cos_tp, sin_tp = np.cos(theta_tp), np.sin(theta_tp)
                    tp = np.stack((cos_tp, sin_tp, -sin_tp, cos_tp), axis=-1)
                    tp = np.reshape(tp, (self.len_edge, 2, 2))
                    kernel = np.einsum('ijk,iklm->ijlm', tp, kernel, optimize='optimal')
            kernel_list.append(kernel) # shape is E x d_in x d_out x (d_in*d_out)
        
        # Kernel basis on the plane
        kernel_cat = np.stack(kernel_list, axis=-1) # shape is E x d_in x d_out x (d_in*d_out) x (n_circle+1)
        r = np.linspace(0, np.pi, self.num_circle+1, endpoint=True)
        coswave = np.cos(np.expand_dims(r, axis=1) * np.expand_dims(np.arange(self.bandwidth+1), axis=0))
        sinwave = np.sin(np.expand_dims(r, axis=1) * np.expand_dims(np.arange(1, self.bandwidth+1), axis=0))
        wave = np.concatenate((coswave, sinwave), axis=-1)
        if n_in != n_out:
            kernel = np.reshape(np.matmul(kernel_cat, sinwave), (self.len_edge, dim_in, dim_out, dim_in*dim_out*self.bandwidth))
        elif n_in == 0:
            kernel = np.reshape(np.matmul(kernel_cat, wave), (self.len_edge, dim_in, dim_out, 2*self.bandwidth+1))
        elif n_in > 0:
            kernel1 = np.reshape(np.matmul(kernel_cat[:, :, :, :2, :], wave), (self.len_edge, dim_in, dim_out, 4*self.bandwidth+2))
            kernel2 = np.reshape(np.matmul(kernel_cat[:, :, :, 2:, :], sinwave), (self.len_edge, dim_in, dim_out, 2*self.bandwidth))
            kernel = np.concatenate((kernel1, kernel2), axis=-1)
        self.basis[(n_in, n_out)] = kernel
        return self.basis[(n_in, n_out)] # shape is E x d_in x d_out x basis_size

class SphereConv(MessagePassing):

    def __init__(self, rep_in, rep_out, precomputation):
        super().__init__(aggr='add', flow='target_to_source')
        self.dim_in = SO2RepDim(rep_in)
        self.dim_out = SO2RepDim(rep_out)
        self.edge_index = torch.tensor(precomputation.edge_index, dtype=torch.long, requires_grad=False)
        self.len_grid = precomputation.len_grid
        self.len_edge = precomputation.len_edge

        indices = []
        values = []
        b0 = 0
        i0 = 0
        for n_in, mult_in in sorted(rep_in.items()):
            dim_in = SO2RepDim(n_in)
            for _ in range(mult_in):
                j0 = 0
                for n_out, mult_out in sorted(rep_out.items()):
                    dim_out = SO2RepDim(n_out)
                    for __ in range(mult_out):
                        kernel = precomputation.getBasis(n_in, n_out)
                        index = np.transpose([[(e*self.dim_in+i0+i)*self.dim_out+j0+j, b0+b] for e in range(precomputation.len_edge) for i in range(dim_in) for j in range(dim_out) for b in range(kernel.shape[-1])])
                        value = kernel.flatten()
                        indices.append(index)
                        values.append(value)
                        b0 += kernel.shape[-1]
                        j0 += dim_out
                i0 += dim_in
        indices = np.concatenate(indices, axis=-1)
        values = np.concatenate(values)
        self.basis = torch.sparse_coo_tensor(indices, values, (self.len_edge*self.dim_in*self.dim_out, b0), dtype=torch.float32, requires_grad=False)
        self.lin = nn.Linear(b0, 1, bias=False)
    
    def forward(self, x):
        # Input shape : len_grid x batch_size x dim_in
        # Output shape : len_grid x batch_size x dim_out
        batch_size = x.shape[1]
        x = torch.reshape(x, (self.len_grid*batch_size, self.dim_in))
        edge_index = torch.cat([self.edge_index+self.len_grid*i for i in range(batch_size)], dim=-1)
        conv = self.propagate(edge_index=edge_index, x=x)
        conv = torch.reshape(conv, (self.len_grid, batch_size, self.dim_out))
        return conv

    def message(self, x_j):
        batch_size = len(x_j)//self.len_edge
        kernel = torch.reshape(self.lin(self.basis), (self.len_edge, self.dim_in, self.dim_out))
        kernel = kernel.repeat(batch_size, 1, 1) # shape is batch_size x d_in x d_out
        return torch.einsum('ij,ijk->ik', x_j, kernel)

class IrrepReLU(nn.Module):

    def __init__(self, n):
        super(IrrepReLU, self).__init__()
        self.n = n
        dim = 1 if n == 0 else 2
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if self.n == 0:
            return self.relu(x+self.b)
        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
            relu_norm = self.relu(norm+self.b)/(norm+1e-15)
            return relu_norm*x

def SO2RepDim(rep):
    if type(rep) == int:
        return 1 if rep == 0 else 2
    elif type(rep) == dict:
        dim = 0
        for n, mult in rep.items():
            dim += (1 if n == 0 else 2) * mult
        return dim