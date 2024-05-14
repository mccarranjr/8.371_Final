import numpy as np
from scipy.linalg import svd

def density_matrix_to_MPO(rho):
    """
    Convert a density matrix to its MPO representation.
    
    Parameters:
    rho (np.ndarray): A density matrix.
    
    Returns:
    list of np.ndarray: A list of tensors representing the MPO.
    """
    
    # Check if the input is a valid density matrix
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("Density matrix must be Hermitian.")
    if not np.isclose(np.trace(rho), 1):
        raise ValueError("Trace of the density matrix must be 1.")
    
    # Initialize list to store MPO tensors
    mpo = []
    
    # Start with the density matrix
    current_matrix = rho
    
    # Perform SVD and reshape matrices into tensors to build the MPO
    while current_matrix.size > 1:
        # Reshape the current matrix into a bipartite system for SVD
        q = int(np.sqrt(current_matrix.shape[0]))
        print(q,'q\n')
        current_matrix = current_matrix.reshape(q, q, -1)
        U, S, Vh = svd(current_matrix.reshape(q**2, -1), full_matrices=False)
        
        # Create the tensor for the current site
        chi = S.size
        tensor = U.reshape(q, q, chi)
        
        # Add the tensor to the MPO list
        mpo.append(tensor)
        
        # Update the current matrix with the remaining singular values
        current_matrix = np.diag(S) @ Vh
    
    # Add the last matrix as a tensor
    mpo.append(current_matrix.reshape(chi, chi, 1))
    
    return mpo

# Example usage:
# Define a 2x2 density matrix (for a single qubit state as an example)
rho_example = np.array([[0.5, 0.5], [0.5, 0.5]])

# Convert the density matrix to its MPO representation
mpo_representation = density_matrix_to_MPO(rho_example)

# Display the result
print(mpo_representation)

