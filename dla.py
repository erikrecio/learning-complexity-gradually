import numpy as np
import jax.random
import jax.numpy as jnp
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)
import pennylane as qml

def get_generators(gen_type: str, nqubits: int) -> list[np.ndarray]:
    """
    Returns a list of generators for a given type of quantum circuit and number of qubits.

    Parameters:
    gen_type (str): The type of the quantum circuit (e.g. "a0", "a1", "a2") referencing
        at the label of table 1 page 7 paper: https://arxiv.org/pdf/2309.05690.pdf
    nqubits (int): The number of qubits in the circuit.

    Returns:
    list[np.ndarray]: A list of 2D numpy arrays representing the generators of the circuit.
    """
    generators = []
    if gen_type == "a0":    
        for q in range(nqubits):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.Z(q)
            generators.append(m)

        for q in range(nqubits-1):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.X(q) @ qml.X(q+1)
            generators.append(m)
    
    elif gen_type == "a1":
        for q in range(nqubits):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.Z(q)
            generators.append(m)

        for q in range(nqubits-1):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.X(q) @ qml.Y(q+1)
            generators.append(m)

    elif gen_type == "a2":
        for q in range(nqubits):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.Z(q)
            generators.append(m)

        for q in range(nqubits-1):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.X(q) @ qml.Y(q+1)
            generators.append(m)

        for q in range(nqubits-1):
            m = qml.I(0)
            for i in range(nqubits):
                if i != q:
                    m = m @ qml.I(i)
                else:
                    m = m @ qml.Y(q) @ qml.X(q+1)
            generators.append(m)

    return generators

def pauli_basis(generators: list, nqubits: int) -> list:
    """
    Given a list of generators of a Lie algebra, returns a basis for the Lie algebra
    as a list of 2^nqubits x 2^nqubits complex matrices. Only works for Pauli Lie algebras.

    The generators are assumed to be Hermitian.

    :param generators: The generators of the Lie algebra as a list of PennyLane
                       operators.
    :param nqubits: The number of qubits in the system.
    :return: A list of 2^nqubits x 2^nqubits complex matrices representing a basis
             for the Lie algebra.
    """
    zeros = np.zeros((2**nqubits, 2**nqubits))

    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Computes the commutator of two matrices.

        :param A: The first matrix.
        :param B: The second matrix.
        :return: The commutator of A and B.
        """
        return np.matmul(A, B) - np.matmul(B, A)

    def compare(A: np.ndarray, B: np.ndarray) -> bool:
        """
        Checks if two matrices are proportional.

        :param A: The first matrix.
        :param B: The second matrix.
        :return: True if the matrices are proportional, False otherwise.
        """
        r = None
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i,j] == 0 and B[i,j] == 0:
                    pass
                elif A[i,j] == 0 and B[i,j] != 0 or A[i,j] != 0 and B[i,j] == 0:
                    return False
                elif r is None:
                    r = A[i,j]/B[i,j]
                elif not A[i,j]/B[i,j] == r:
                    return False
        return True

    g_matrices = [qml.matrix(g) for g in generators]
    g_all = g_matrices.copy()
    g_prev = g_matrices.copy()
    num_new = 1

    while num_new != 0:
        g_new = []
        for g in g_matrices:
            for gp in g_prev:
                gn = commutator(g, gp)
                if not (gn == zeros).all():
                    gn_new = True
                    for ga in g_all:
                        if compare(ga,gn):
                            gn_new = False
                            break
                    if gn_new:
                        g_all.append(gn)
                        g_new.append(gn)
        num_new = len(g_new)
        # g_matrices += g_new
        g_prev = g_new

    normalized_basis = []
    for g in g_all:
        normalized_basis.append(1/np.sqrt(np.trace(np.matrix(g).getH() @ g))*g)

    return normalized_basis

def get_gen_basis(gen_type: str, nqubits: int) -> list[np.ndarray]:
    """
    Get the basis of generators of the Lie algebra for a given type of quantum circuit and number of qubits.

    Parameters:
    gen_type (str): The type of the quantum circuit (e.g. "a0", "a1", "a2") referencing
        at the label of table 1 page 7 paper: https://arxiv.org/pdf/2309.05690.pdf
    nqubits (int): The number of qubits in the circuit.

    Returns:
    list[np.ndarray]: A list of 2D numpy arrays representing the basis of generators of the circuit.
    """
    generators = get_generators(gen_type, nqubits)
    return pauli_basis(generators, nqubits)

def purity(state: np.ndarray, gen_basis: list[np.ndarray]) -> float:
    """
    Compute the purity of a quantum state given a basis of generators of a Lie algebra.

    Parameters:
    state (np.ndarray): The quantum state as a 1D numpy array.
    gen_basis (List[np.ndarray]): A list of 2D numpy arrays representing a basis for the generators of the Lie algebra.

    Returns:
    float: The purity of the quantum state.
    """
    rho = jnp.tensordot(state, jnp.conjugate(state), axes=0)
    return sum([jnp.absolute(jnp.trace(jnp.transpose(jnp.conjugate(g)) @ rho))**2 for g in gen_basis])
