import string
from qiskit import *
from qiskit import QuantumCircuit, Aer
import numpy as np
from scipy import linalg
from functools import reduce
import itertools

# Decompose a unitary into uniformly controlled single-qubit rotations using the Cosine-Sine Decomposition.
# Returns a final sequence L D R L D R L D R ... L D R, where L and R are uniformly controlled gates on the last qubit and
# D are uniformly controlled Ry gates on the second-to-last qubit.
def factor(input):
    def diag(t,b):
        zero = np.zeros((t.shape[0], b.shape[1]))
        return np.block([
            [t, zero],
            [zero.transpose(), b]
        ])

    def CSD(u):
        dim = u.shape[0]
        dim2 = int(dim//2)
        if dim <= 2:
            return [u]
        else:
            L,D,R = linalg.cossin(u, dim2, dim2)

            L1 = L[:dim2,:dim2]
            L2 = L[dim2:,dim2:]
            R1 = R[:dim2,:dim2]
            R2 = R[dim2:,dim2:]

            l1_f = CSD(L1)
            l2_f = CSD(L2)
            r1_f = CSD(R1)
            r2_f = CSD(R2)

            return [diag(l1, l2) for l1,l2 in zip(l1_f, l2_f)] + [D] + [diag(r1, r2) for r1,r2 in zip(r1_f, r2_f)]

    if type(input) == np.ndarray:
        unitary = input
    else:
        if type(input) == str:
            qc = QuantumCircuit.from_qasm_file(input)
        else:
            qc = input
        qc.remove_final_measurements()

        backend = Aer.get_backend('unitary_simulator')
        job = execute(qc, backend)
        result = job.result()
        unitary = np.array(result.get_unitary(qc, decimals=3))
    
    mats = CSD(unitary)
    reconstruct = reduce(np.matmul, mats)
    fid = np.abs(np.trace(np.transpose(np.conjugate(reconstruct)) @ unitary))**2 / unitary.shape[0]**2
    print("Number of ops:",len(mats))
    print("Fidelity:", fid)
    return mats

i2 = np.array([[1, 0], [0, 1]])