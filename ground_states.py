import numpy as np
import jax.random
import jax.numpy as jnp
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)
import pennylane as qml
from shapely.geometry import Polygon, Point

# @jax.jit
def ground_state(j1: float, j2: float, nqubits: int, ham: str) -> np.ndarray:
    """
    Generates the ground state of the Hamiltonian specified by the parameters.

    Parameters:
    j1 (float): The coupling constant for the nearest neighbor interaction.
    j2 (float): The coupling constant for the next nearest neighbor interaction.
    nqubits (int): The number of qubits in the Hamiltonian.
    ham (str): The type of Hamiltonian to generate. Must be either "gch" or "ssh".

    Returns:
    np.ndarray: The ground state of the Hamiltonian as a 1D numpy array.
    """
    
    # Initialize the Hamiltonian to zero
    hamiltonian = 0
    
    # Construct the Hamiltonian based on the specified type
    if ham == "gch":
        # Loop over each qubit to add terms to the Hamiltonian
        for i in range(nqubits):
            hamiltonian += qml.Z(i)
            hamiltonian -= j1 * qml.X(i) @ qml.X((i + 1) % nqubits)
            hamiltonian -= j2 * qml.X((i - 1) % nqubits) @ qml.Z(i) @ qml.X((i + 1) % nqubits)
        
    elif ham == "ssh":
        # Add interaction terms within pairs of qubits
        for i in range(nqubits // 2):
            hamiltonian += 0.5 * (qml.X(2 * i) @ qml.X(2 * i + 1) + 
                                  qml.Y(2 * i) @ qml.Y(2 * i + 1) + 
                                  j2 * qml.Z(2 * i) @ qml.Z(2 * i + 1))
        # Add interaction terms between adjacent pairs
        for i in range((nqubits - 1) // 2):
            hamiltonian += j1 / 2 * (qml.X(2 * i + 1) @ qml.X(2 * i + 2) + 
                                     qml.Y(2 * i + 1) @ qml.Y(2 * i + 2) + 
                                     j2 * qml.Z(2 * i + 1) @ qml.Z(2 * i + 2))
        
        # Add periodic boundary conditions
        hamiltonian += j1 / 2 * (qml.X(nqubits - 1) @ qml.X(0) + 
                                 qml.Y(nqubits - 1) @ qml.Y(0) + 
                                 j2 * qml.Z(nqubits - 1) @ qml.Z(0))
    
    # Convert the Hamiltonian to a matrix
    ham_matrix = qml.matrix(hamiltonian)
    
    # Compute the eigenvalues and eigenvectors of the Hamiltonian matrix
    _, eigvecs = jnp.linalg.eigh(ham_matrix)
    
    # Return the ground state, which corresponds to the eigenvector of the lowest eigenvalue
    return eigvecs[:, 0]


# Define coordinates of the points of each region
region01_coords = np.array([(-2, 1), (2, 1), (4, 3), (4, 4), (-4, 4), (-4, 3)])    # Class 0
region02_coords = np.array([(-3, -4), (0, -1), (3, -4)])                           # Class 0
region1_coords = np.array([(0, -1), (3, -4), (4, -4), (4, 3)])                     # Class 1
region2_coords = np.array([(0, -1), (-3, -4), (-4, -4), (-4, 3)])                  # Class 2
region3_coords = np.array([(-2, 1), (2, 1), (0, -1)])                              # Class 3

region1_coords2 = np.array([(0, 0), (1.1, 0), (1.1, 1.5), (0.75, 4), (0, 4)])
region2_coords2 = np.array([(0.75, 4), (1.1, 1.5), (2.3, 4)])
region3_coords2 = np.array([(1.1, 0), (1.1, 1.5), (2.3, 4), (3, 4), (3, 0)])

e = 0.1
# Define coordinates of the points of each region far from the borders
region01e_coords = np.array([(-2+(np.sqrt(2)-1)*e, 1+e), (2-(np.sqrt(2)-1)*e, 1+e), (4, 3+np.sqrt(2)*e), (4, 4), (-4, 4), (-4, 3+np.sqrt(2)*e)])    # Class 0 with epsilon
region02e_coords = np.array([(-3+np.sqrt(2)*e, -4), (0, -1-np.sqrt(2)*e), (3-np.sqrt(2)*e, -4)])                                                    # Class 0 with epsilon
region1e_coords = np.array([(0+np.sqrt(2)*e, -1), (3+np.sqrt(2)*e, -4), (4, -4), (4, 3-np.sqrt(2)*e)])                                              # Class 1 with epsilon
region2e_coords = np.array([(0-np.sqrt(2)*e, -1), (-3-np.sqrt(2)*e, -4), (-4, -4), (-4, 3-np.sqrt(2)*e)])                                           # Class 2 with epsilon
region3e_coords = np.array([(-2+e/np.tan(np.pi/8), 1-e), (2-e/np.tan(np.pi/8), 1-e), (0, -1+np.sqrt(2)*e)])                                         # Class 3 with epsilon


def labeling(x: float, y: float, ham: str) -> int:
    """
    Determines the label for a given point (x, y) based on its location within
    predefined region polygons.

    Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        ham (str): The type of Hamiltonian to consider. Must be either "gch" or "ssh".
    Returns:
        int: A label (0, 1, 2, or 3) corresponding to the region the point belongs to,
             or raises an exception if the point is not in any region.

    Note:
        Labels are assigned as follows:

        For GCH Hamiltonian (ham = "gch"):
        0: Region 01
        0: Region 02
        1: Region 1
        2: Region 2
        3: Region 3

        For SSH Hamiltonian (ham = "ssh"):
        1: Region 1
        2: Region 2
        3: Region 3
    """
    
    p = Point(x, y)  # Create a point object for the given coordinates

    if ham == "gch":
        # Define polygons for each region in GCH Hamiltonian
        region01_poly = Polygon(region01_coords)
        region02_poly = Polygon(region02_coords)
        region1_poly = Polygon(region1_coords)
        region2_poly = Polygon(region2_coords)
        region3_poly = Polygon(region3_coords)
        
        # Check which polygon contains the point and return corresponding label
        if region01_poly.contains(p) or region02_poly.contains(p):
            return 0
        elif region1_poly.contains(p):
            return 1
        elif region2_poly.contains(p):
            return 2
        elif region3_poly.contains(p):
            return 3
        else:
            raise Exception(f"Point ({x}, {y}) is not in any region")  # Point not in any region
    
    elif ham == "ssh":
        # Define polygons for each region in SSH Hamiltonian
        region1_poly = Polygon(region1_coords2)
        region2_poly = Polygon(region2_coords2)
        region3_poly = Polygon(region3_coords2)
        
        # Check which polygon contains the point and return corresponding label
        if region1_poly.contains(p):
            return 1
        elif region2_poly.contains(p):
            return 2
        elif region3_poly.contains(p):
            return 3
        else:
            raise Exception(f"Point ({x}, {y}) is not in any region")  # Point not in any region

    
def labeling_epsilon(x: float, y: float) -> int | None:
    """
    Determines the label for a given point (x, y) based on its location within
    predefined epsilon-deflated region polygons. Only implemented for ham == "gch".

    Parameters:
    x (float): The x-coordinate of the point.
    y (float): The y-coordinate of the point.

    Returns:
    int | None: A label (0, 1, 2, or 3) corresponding to the region the point belongs to,
              or None if the point is not in any region.
    """
    # Create polygon objects for each region
    region01e_poly = Polygon(region01e_coords)
    region02e_poly = Polygon(region02e_coords)
    region1e_poly = Polygon(region1e_coords)
    region2e_poly = Polygon(region2e_coords)
    region3e_poly = Polygon(region3e_coords)
    
    # Create a point object for the given coordinates
    p = Point(x, y)
    
    # Check which polygon contains the point and return corresponding label
    if region01e_poly.contains(p) or region02e_poly.contains(p):
        return 0
    elif region1e_poly.contains(p):
        return 1
    elif region2e_poly.contains(p):
        return 2
    elif region3e_poly.contains(p):
        return 3
    else:
        return None # if the point is not in any region



# Generate ground states
def generate_gs(num_points: int, uniform: bool, epsilon: bool, nqubits: int, ham: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a specified number of ground states for a Hamiltonian system 
    based on provided parameters, either uniformly or with balanced sampling.

    Parameters:
    num_points (int): Number of ground states to generate.
    uniform (bool): Whether to sample uniformly from the j's space or balanced among classes.
    epsilon (bool): If True, uses epsilon-deflated regions for sampling.
    nqubits (int): The number of qubits in the Hamiltonian.
    ham (str): The type of Hamiltonian to generate. Must be either "gch" or "ssh".

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays:
        - gs_list: The generated ground states as a 2D numpy array.
        - labels_list: Labels corresponding to the region each ground state belongs to.
        - j_list: The coupling constants j1 and j2 used for each ground state as a 2D numpy array.
    """
    
    if uniform:  # Sample the ground states uniformly from the j's space
        if epsilon:
            j_list = []
            num = 0
            while num < num_points:
                # Randomly sample coupling constants j1 and j2
                j = np.random.uniform(-4, 4, 2)
                # Determine the label for the sampled point
                l = labeling_epsilon(j[0], j[1], ham)

                if l in [0, 1, 2, 3]:
                    num += 1
                    j_list.append(j)

            j_list = np.array(j_list)
            
        else:
            if ham == "gch":
                j_list = np.random.uniform(-4, 4, (num_points, 2))
            elif ham == "ssh":
                j_list = np.column_stack((np.random.uniform(0, 3, num_points), np.random.uniform(0, 4, num_points)))
    
    else:  # Sample the same number of ground states for each class
        
        if ham == "gch":
            npoints_class = num_points // 4
            npoints_02 = npoints_class // 2
            npoints_01 = npoints_class - npoints_02 + num_points % 4
            
            # This would be another definition of balanced, with the same number of points per area (so class 0 has double points)
            ## npoints_class = num_points//5
            ## npoints_02 = npoints_class
            ## npoints_01 = npoints_class + num_points%4
            
            j_list = []
            num_01, num_02, num_1, num_2, num_3 = 0, 0, 0, 0, 0
            
            while num_01 != npoints_01 or num_02 != npoints_02 or num_1 != npoints_class or num_2 != npoints_class or num_3 != npoints_class:
                # Randomly sample coupling constants j1 and j2
                j = np.random.uniform(-4, 4, 2)
                # Determine the label for the sampled point
                l = labeling_epsilon(j[0], j[1]) if epsilon else labeling(j[0], j[1], ham)

                if l == 0:
                    p = Point(j[0], j[1])
                    if Polygon(region01_coords).contains(p) and num_01 < npoints_01:
                        num_01 += 1
                        j_list.append(j)
                        
                    elif Polygon(region02_coords).contains(p) and num_02 < npoints_02:
                        num_02 += 1
                        j_list.append(j)
                    
                elif l == 1 and num_1 < npoints_class:
                    num_1 += 1
                    j_list.append(j)
                elif l == 2 and num_2 < npoints_class:
                    num_2 += 1
                    j_list.append(j)
                elif l == 3 and num_3 < npoints_class:
                    num_3 += 1
                    j_list.append(j)
                    
        elif ham == "ssh":
            npoints_1 = num_points // 4
            npoints_3 = num_points // 4
            npoints_2 = num_points - npoints_1 - npoints_3
            
            j_list = []
            num_1, num_2, num_3 = 0, 0, 0
            
            while num_1 != npoints_1 or num_2 != npoints_2 or num_3 != npoints_3:
                # Randomly sample coupling constants j1 and j2
                j = np.array([np.random.uniform(0, 3), np.random.uniform(0, 4)])
                # Determine the label for the sampled point
                l = labeling(j[0], j[1], ham)
                    
                if l == 1 and num_1 < npoints_1:
                    num_1 += 1
                    j_list.append(j)
                elif l == 2 and num_2 < npoints_2:
                    num_2 += 1
                    j_list.append(j)
                elif l == 3 and num_3 < npoints_3:
                    num_3 += 1
                    j_list.append(j)
            
        j_list = np.array(j_list)
    
    # Generate ground states using the coupling constants
    gs_list = jax.vmap(ground_state, in_axes=[0, 0, None, None])(j_list[:, 0], j_list[:, 1], nqubits, ham)
    
    labels_list = []
    for i in range(num_points):
        # Determine the label for each coupling constant pair
        labels_list.append(labeling(j_list[i, 0], j_list[i, 1], ham))
    
    gs_list = np.array(gs_list)
    labels_list = np.array(labels_list)
    
    return gs_list, labels_list, j_list
