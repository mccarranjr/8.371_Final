import pennylane as qml
from pennylane import numpy as np
from shadow_functions import (
    find_neighbors, construct_hamiltonian 
    )
import argparse
from argparse import ArgumentParser
from matplotlib import pyplot as plt



np.random.seed(11)
def get_config():
    parser = ArgumentParser()


    parser.add_argument('--J', type = float, default=1)
    parser.add_argument('--h', type=float, default=1 )
    parser.add_argument('--nb_rows', type = int, default=3)
    parser.add_argument('--nb_columns', type = int, default=3)
    parser.add_argument('--periodic', type = bool, default=True)
    parser.add_argument('--num_snapshots', type = int, default=1000)
    parser.add_argument('--circuit', type=str, default='ansatz')
    parser.add_argument('--params', type=str, default='2D_TFIM')

    args = parser.parse_args()
    return args

    zero = np.array([[1,0]])
    one = np.array([[0,1]])
    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0],[0,1]])
    phase_z = np.array([[1,0],[0,-1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))
def state_dict(state):
    dic = {
        'zero' : np.array([[1,0]]),
        'one' : np.array([[0,1]]),
        'pi_0' : np.array([[1,0],[0,0]]),
        'pi_1' : np.array([[0,0],[0,1]]),
        'phase' : np.array([[1,0],[0,-1j]], dtype=complex),
        'H' : qml.matrix(qml.Hadamard(0)),
        'I' : qml.matrix(qml.Identity(0))
     }
    return dic[state]

dev = qml.device("default.qubit", wires=9, shots=1)
@qml.qnode(dev)
def ansatz_circuit(num_qubits, params, neighbors, obs): 
    """
    Single layer circuit taken from IBM efficient SU(2) documentation
    Returns list of expectation values for each observable passed in
    """
    for i in range(num_qubits):
        qml.RY(params[3*i], wires=i)
        qml.RX(params[3*i+1],wires=i)
        for j in neighbors[i]:
            qml.CNOT(wires=[i, j])
        qml.RY(params[3*i+2],wires=i)
    out = [qml.expval(o) for o in obs]
    return out 

def snapshot_state(b_list = None, obs_list = None, im_time_evo_op = None):

    num_qubits = len(b_list)

    zero = np.array([[1,0]])
    one = np.array([[0,1]])
    zero_state = np.array([[1,0],[0,0]])
    one_state = np.array([[0,0],[0,1]])
    phase_z = np.array([[1,0],[0,-1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    unitaries = [hadamard, hadamard@phase_z, identity]
    #rhos = []
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot




def calculate_classical_shadow(circuit_template = None, params = None, neighbors = None, shadow_size = None, num_qubits = None):

    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits)) #generate random measurement bases

    outcomes = np.zeros((shadow_size, num_qubits))

    for num_shadows in range(shadow_size): #pick random observalbes and calculate expectation values
        obs = [unitary_ensemble[int(unitary_ids[num_shadows,i])](i) for i in range(num_qubits)]
        #obs = [unitary_ensemble[unitary_id](i) for i, unitary_id in enumerate(unitary_ids[num_shadows])]
        outcomes[num_shadows, :] = circuit_template(num_qubits, params, neighbors, obs)
    
    return (outcomes, unitary_ids)


def shadow_state_reconstruction(shadow):

    num_snapshots, num_qubits = shadow[0].shape
    b_lists, obs_lists = shadow

    shadow_rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])#each snapshot state has dim (512,512)
    return shadow_rho / np.trace(shadow_rho) #num_snapshots


def create_classical_shadow(ham, circuit, params, nearest_neighbors, shadow_size, num_qubits):
    shadow = calculate_classical_shadow( 
        circuit_template = circuit, params = params, neighbors = nearest_neighbors,
        shadow_size = shadow_size, num_qubits = num_qubits
        )

    density_matrix = shadow_state_reconstruction(shadow)

    return density_matrix




def main():
    config = get_config()
    '''

    if config.circuit in ['ansatz']:
        if config.circuit == 'ansatz':
            num_qubits = config.nb_columns*config.nb_rows
            print(num_qubits)

            circuit = ansatz_circuit
            '''

    if config.params in ['2D_TFIM']:
        if config.J == config.h == 1:
            if config.nb_rows == config.nb_columns == 3:
                params = [-4.18276290e-07, -6.13030748e-08,  2.52708489e-01, -7.77878068e-08,
        -2.99356307e-09,  2.52721373e-01,  1.02800675e-02, -3.27243045e-08,
            2.53353876e-01, -2.92291622e-08, -9.68577237e-07,  2.52721884e-01,
            5.51256128e-08, -7.58435451e-08,  2.52734054e-01,  1.03054950e-02,
        -4.20498697e-07,  2.53958793e-01,  1.02802252e-02, -2.44150501e-07,
            2.53353775e-01,  1.03059920e-02, -2.80372170e-07,  2.53960352e-01,
            1.05437284e-01, -1.44041963e-06,  1.48563628e-01]

    nearest_neighbors = find_neighbors(config.nb_rows, periodic=config.periodic)
    ham = construct_hamiltonian(config.J, config.h, config.nb_rows, config.nb_columns, config.periodic, nearest_neighbors)

    rho_hat = create_classical_shadow(ham, ansatz_circuit, params, nearest_neighbors, config.num_snapshots, config.nb_rows*config.nb_columns)

    energy = np.trace(qml.matrix(ham)@rho_hat)

    print(energy)

if __name__ == '__main__':
    main()

    
    



#next --> i think MPS is kinda fucked
#implement --> shadow VQE and compare with tdvp