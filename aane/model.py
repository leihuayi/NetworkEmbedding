#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from math import ceil

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   AANE MODEL                                                                                  #
#   Implementation of AANE algorithm                                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
class AANE:
    def __init__(self, G_mat, Attri, args):

        [self.num_of_nodes, m] = Attri.shape  # num_of_nodes = Total num of nodes, m = attribute category num

        Gmat = sparse.lil_matrix(G_mat)
        Gmat.setdiag(np.zeros(self.num_of_nodes))
        Gmat = csc_matrix(Gmat)
        Attri = csc_matrix(Attri)

        self.maxiter = int(args.iter)  # Max num of iteration
        self.lambd = float(args.lambd)  # Initial regularization parameter
        self.rho = float(args.rho)  # Initial penalty parameter
        self.dim = int(args.dim) # Embedding dimension
        splitnum = 1  # number of pieces we split the SA for limited cache

        sumcol = Gmat.sum(0)

        # Initialize H and Z
        self.H = svds(Gmat[:, sorted(range(self.num_of_nodes), key=lambda k: sumcol[0, k], reverse=True)[0:min(10 * self.dim, self.num_of_nodes)]], self.dim)[0]
        self.Z = self.H.copy()

        # Use blocks for big matrices operations
        self.block = min(int(ceil(float(self.num_of_nodes) / splitnum)), 5000)  # Treat at least each 5000 nodes as a block
        self.splitnum = int(ceil(float(self.num_of_nodes) / self.block))
        with np.errstate(divide='ignore'):  # inf will be ignored
            self.Attri = Attri.transpose() * sparse.diags(np.ravel(np.power(Attri.power(2).sum(1), -0.5)))
        
        self.affi = -1  # Index for affinity matrix sa
        self.U = np.zeros((self.num_of_nodes, self.dim))
        self.nexidx = np.split(Gmat.indices, Gmat.indptr[1:-1])
        self.Gmat = np.split(Gmat.data, Gmat.indptr[1:-1])


    # Update matrix embedding representation H
    def updateH(self):
        xtx = np.dot(self.Z.transpose(), self.Z) * 2 + self.rho * np.eye(self.dim)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks
            if self.affi != blocki:
                indices = np.r_[indexblock : indexblock + min(self.num_of_nodes - indexblock, self.block)]
                self.sa = self.Attri[:, indices].transpose() * self.Attri
                self.affi = blocki
            sums = self.sa.dot(self.Z) * 2
            
            for i in range(indexblock, indexblock + min(self.num_of_nodes - indexblock, self.block)):
                neighbor = self.Z[self.nexidx[i], :]  # the set of adjacent nodes of node i

                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.H[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index

                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Gmat[i][nzidx]) / normi_j[nzidx]
                        self.H[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.dim), sums[i - indexblock, :] + 
                            (neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (self.Z[i, :] - self.U[i, :]))
                    else:
                        self.H[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + self.rho * (self.Z[i, :] - self.U[i, :]))


    # Update Z (copy of H)           
    def updateZ(self):
        xtx = np.dot(self.H.transpose(), self.H) * 2 + self.rho * np.eye(self.dim)
        for blocki in range(self.splitnum):  # Split nodes into different Blocks
            indexblock = self.block * blocki  # Index for splitting blocks

            if self.affi != blocki:
                indices = np.r_[indexblock : indexblock + min(self.num_of_nodes - indexblock, self.block)]
                self.sa = self.Attri[:, indices].transpose() * self.Attri
                self.affi = blocki
            sums = self.sa.dot(self.H) * 2

            for i in range(indexblock, indexblock + min(self.num_of_nodes - indexblock, self.block)):
                neighbor = self.H[self.nexidx[i], :]  # the set of adjacent nodes of node i

                for j in range(1):
                    normi_j = np.linalg.norm(neighbor - self.Z[i, :], axis=1)  # norm of h_i^k-z_j^k
                    nzidx = normi_j != 0  # Non-equal Index

                    if np.any(nzidx):
                        normi_j = (self.lambd * self.Gmat[i][nzidx]) / normi_j[nzidx]
                        self.Z[i, :] = np.linalg.solve(xtx + normi_j.sum() * np.eye(self.dim), sums[i - indexblock, :] + 
                            (neighbor[nzidx, :] * normi_j.reshape((-1, 1))).sum(0) + self.rho * (self.H[i, :] + self.U[i, :]))
                    else:
                        self.Z[i, :] = np.linalg.solve(xtx, sums[i - indexblock, :] + self.rho * (self.H[i, :] + self.U[i, :]))


    # Return dictionary of nodes and their embeddings 
    def embedding_mapping(self):
        return {i: self.H[i] for i in range(self.H.shape[0])}

    # Run algorithm
    def run(self):
        self.updateH()

        print("AANE Iteration :")
        for i in range(self.maxiter):
            print("%s/%s"%(i+1, self.maxiter))
            self.updateZ()
            self.U = self.U + self.H - self.Z
            self.updateH()
        return self.embedding_mapping()



