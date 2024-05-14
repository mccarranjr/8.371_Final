from matplotlib import pyplot as plt
import numpy as np
from numpy import array
import scipy
from scipy.optimize import minimize
import random
import math
import networkx as nx
import pennylane as qml
import itertools as it
import argparse
from argparse import ArgumentParser
import os

from cost_funcs import exact_cost_no_shadow, exact_cost_shadow, updated_cost_shadow
from functions import sigmoid, prob_dist, calculate_entropy

def get_config():
    parser = ArgumentParser()


    parser.add_argument('--beta', type = float, default=0.5)
    parser.add_argument('--qubit', type=int, default=3)
    parser.add_argument('--shadow', type=bool, default=False)
    parser.add_argument('--hamiltonian', type=str, default='ising')
    parser.add_argument('--num_snapshots', type = int, default=10)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--save_name', type=str, default='3_qubit_ising')
    parser.add_argument('--original',type=bool,default=True)


    args = parser.parse_args()
    return args

def create_hamiltonian_matrix(n, graph):
  pauli_x = np.array([[0, 1], [1, 0]])
  pauli_y = np.array([[0, -1j], [1j, 0]])
  pauli_z = np.array([[1, 0], [0, -1]])
  identity = np.array([[1, 0], [0, 1]])

  matrix = np.zeros((2**n, 2**n))

  for i in graph.edges:
    m = 1
    for j in range(0,n):
      if j == i[0] or j == i[1]:
        m = np.kron(m, pauli_x)
      else:
        m = np.kron(m, pauli_z)
    matrix = np.add(matrix, m)

  for i in range(0,n):
    m = 1
    for j in range(0,n):
      if j == i:
        m = np.kron(m, pauli_z)
      else:
        m = np.kron(m, identity)
    matrix = np.add(matrix, m)
  print(matrix.shape)

  return matrix

#ham_matrix = create_hamiltonian_matrix(qubit, interaction_graph)

def create_density_plot(data,save_name):

  array = np.array(data)
  plt.matshow(array)
  plt.colorbar()
  path = f'results/{save_name}'
  if not os.path.exists(path):
    os.makedirs(path)
  plt.savefig(path)


def create_target(qubit, beta, h):

  y = -1*float(beta)*h
  new_matrix = scipy.linalg.expm(np.array(y))
  norm = np.trace(new_matrix)
  final_target = new_matrix / norm

  entropy = -1*np.trace(np.matmul(final_target, scipy.linalg.logm(final_target)))
  ev = np.trace(np.matmul(final_target, h))
  real_cost = beta*ev - entropy

  print(f'Expectation with H: {ev}')
  print(f'Entropy: {entropy}')
  print(f'Cost function value: {real_cost}')
  return final_target

#final_density_matrix = create_target(qubit, beta, create_hamiltonian_matrix, interaction_graph)

#create_density_plot(final_density_matrix)
'''
def sigmoid(x):
  return np.exp(x) / (np.exp(x) + 1)

def prob_dist(params):
  return np.vstack([sigmoid(params), 1 - sigmoid(params)]).T
'''
def create_v_gate(prep_state):
  for i in range(0, len(prep_state)):
    if prep_state[i] == 1:
      qml.PauliX(wires = i)

def single_rotation(phi_params, q):
  qml.RZ(phi_params[0], wires = q)
  qml.RY(phi_params[1], wires = q)
  qml.RZ(phi_params[2], wires = q)

def ansatz_circuit(params, qubits, depth, graph, param_number,shadow=False,obs=None):

  param_number = int(param_number)
  number = param_number*len(qubits) + len(graph.edges)
  
  partition = []
  for i in range(0, int((len(params) / number))):
    partition.append(params[number*i:number*(i+1)])
  #print(number,'number')
  #print(len(partition),'len partition')
  #print(depth,'depth')
  #print(param_number,'param_number')


  for j in range(0, depth):
    sq = partition[j][0:number - len(graph.edges)]

    for i in qubits:
      single_rotation(sq[i*param_number:(i+1)*param_number], i)

    for count, i in enumerate(graph.edges):
      p = partition[j][(number - len(graph.edges)) : number]
      qml.CRX(p[count], wires = (i[0], i[1]))
  if shadow:
    return [qml.expval(o) for o in obs]
    

def quantum_circuit(params, qubits, depth, sample, param_number, interaction_graph, ham_matrix):

  create_v_gate(sample)

  ansatz_circuit(params, qubits, depth, interaction_graph, param_number)

  return qml.expval(qml.Hermitian(ham_matrix, wires = qubits))

def quantum_circuit_with_shadow(params, qubits, depth, sample, param_number, interaction_graph, obs,return_state=False,ham_matrix=None):

  create_v_gate(sample)
  if return_state==True:
    return qml.state()
  else:
    return ansatz_circuit(params, qubits, depth, interaction_graph, param_number,shadow=True,obs=obs)

  #if return_state:
    #return qml.state()
  #else:
    #return [qml.expval(o) for o in obs]

def updated_quantum_circuit_with_shadow(params, qubits, depth, sample, param_number, interaction_graph, obs,return_state=False):

  ansatz_circuit(params, qubits, depth, interaction_graph, param_number)

  if return_state:
    return qml.state()
  else:
    return [qml.expval(o) for o in obs]

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

def shadow_state_reconstruction(b_lists, obs_lists, num_snapshots):

    shadow_rho = np.zeros((2**qubit, 2**qubit), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])#each snapshot state has dim (512,512)
    return shadow_rho / np.trace(shadow_rho)

def prepare_shadow_state(qubit,params, qnode, device, depth, interaction_graph,shadow_size=None):

    # Initializes the density matrix

    final_density_matrix_2 = np.zeros((2**qubit, 2**qubit))

    # Prepares the optimal parameters, creates the distribution and the bitstrings

    dist_params = params[0:qubit]
    unitary_params = params[qubit:]

    distribution = prob_dist(dist_params)

    s = [[int(i) for i in list(bin(k)[2:].zfill(qubit))] for k in range(0, 2**qubit)]

    # Runs the circuit in the case of the optimal parameters, for each bitstring, and adds the result to the final density matrix

    for i in s:
        qnode(unitary_params, range(qubit), depth, i, qubit, interaction_graph, None,return_state=True)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]])*state
        final_density_matrix_2 = np.add(final_density_matrix_2, np.outer(state, np.conj(state)))

    return final_density_matrix_2

def updated_prepare_shadow_state(qubit,params, qnode, device, depth, interaction_graph,shadow_size=None):
  state = np.array([qnode(params, range(qubit), depth, None, qubit, interaction_graph, None,return_state=True)]).T
  #state = device.state()
  #state.reshape((state.shape[0],1))
  print(state.shape)
  rho_hat = state@state.conj().T
  print(rho_hat.shape)
  return rho_hat


#qnode = qml.QNode(quantum_circuit, dev)
'''
def calculate_entropy(distribution):

  total_entropy = []
  for i in distribution:
    total_entropy.append(-1*i[0]*np.log(i[0]) + -1*i[1]*np.log(i[1]))

  return total_entropy

def exact_cost(params):
  global iterations

  dist_params = params[0:qubit]
  params = params[qubit:]

  distribution = prob_dist(dist_params)

  combos = it.product([0,1], repeat = qubit)
  s = [list(c) for c in combos]


  final_cost = 0
  for i in s:
    result = qnode(params, qubits, i, 3)
    for j in range(0, len(i)):
      result = result*distribution[j][i[j]]
    final_cost += result

  entropy = calculate_entropy(distribution)
  final_final_cost = beta*final_cost - sum(entropy)
  #print(beta*final_cost,'beta * final')
  #print(sum(entropy))

  if (iterations%50 == 0):
    print("Cost at Step "+str(iterations)+": "+str(final_final_cost))

  iterations += 1

  return final_final_cost
'''


#params = [random.randint(-100, 100)/100 for i in range(0, (12*depth)+3)]
#out = minimize(exact_cost, x0=params, args=(qubit, qnode, beta), method="COBYLA", options={'maxiter':750})
#iterations += 1
#params = out['x']
#print(out)


def prepare_state(params, device, qubit, qnode, depth, interaction_graph, ham_matrix):

    # Initializes the density matrix

    final_density_matrix_2 = np.zeros((2**qubit, 2**qubit))

    # Prepares the optimal parameters, creates the distribution and the bitstrings

    dist_params = params[0:qubit]
    unitary_params = params[qubit:]

    distribution = prob_dist(dist_params)

    s = [[int(i) for i in list(bin(k)[2:].zfill(qubit))] for k in range(0, 2**qubit)]

    # Runs the circuit in the case of the optimal parameters, for each bitstring, and adds the result to the final density matrix

    for i in s:
        qnode(unitary_params, range(qubit), depth, i, qubit, interaction_graph, ham_matrix)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]])*state
        final_density_matrix_2 = np.add(final_density_matrix_2, np.outer(state, np.conj(state)))

        entropy = -1*np.trace(np.matmul(final_density_matrix_2, scipy.linalg.logm(final_density_matrix_2)))
        #print(entropy,'entropy final \n')
        ev = np.trace(np.matmul(final_density_matrix_2, ham_matrix))
        #print(ev,'EV final')
        real_cost = 0.5*ev - entropy

    return final_density_matrix_2

#final_density_matrix_2 = prepare_state(params, dev)

def trace_distance(one,two):
  return 0.5 * np.trace(np.absolute(np.add(one, -1*two)))

#print(f'Final fidelity = {trace_distance(final_density_matrix_2,final_density_matrix)}')



def main():
    config = get_config()
    qubit = config.qubit
    print(qubit)
    qubits = range(qubit)
    dev = qml.device('lightning.qubit', wires=len(qubits))#,shots=1)

    if config.hamiltonian == 'ising':
      if qubit == 3:
        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(range(0, qubit))
        interaction_graph.add_edges_from([(0,1),(1,2)])

      elif qubit == 4:
        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(range(0, qubit))
        interaction_graph.add_edges_from([(0,1),(1,2)])
        interaction_graph.add_edges_from([(0,1),(1,2),(2,3)])
      elif qubit == 2:
        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(range(0, qubit))
        interaction_graph.add_edges_from([(0,1)])

      ham_matrix = create_hamiltonian_matrix(qubit, interaction_graph)

    if config.shadow == False:
        cost_func = exact_cost_no_shadow
        circuit = quantum_circuit
        prep_state = prepare_state
        print('hi')
        params = [random.randint(-300, 300)/100 for i in range(0, (19*config.depth)+config.qubit)]

    else:
      if config.original == True:
        cost_func = exact_cost_shadow
        circuit = quantum_circuit_with_shadow
        prep_state = prepare_shadow_state
        params = np.load('rand_params.npy')#[random.randint(-100, 100)/100 for i in range(0, (19*config.depth)+config.qubit)]
      else:
        cost_func = updated_cost_shadow
        circuit = updated_quantum_circuit_with_shadow
        prep_state = updated_prepare_shadow_state
        params = np.load('ising_3_bench_params')#[random.randint(-100, 100)/200 for i in range(0, (19*config.depth)+config.qubit)]



    
    final_density_matrix = create_target(config.qubit, config.beta, ham_matrix)
    create_density_plot(final_density_matrix,f'exact_sol_{config.save_name}')

    qnode = qml.QNode(circuit, dev)#shots=1)
    print(len(params),'num params')
    #params = np.load('final_params_heisenberg.npy')
    out = minimize(cost_func, x0=params, args=(config.qubit, qnode, config.depth, config.beta, interaction_graph, ham_matrix), method="COBYLA", options={'maxiter':50})
    params = out['x']
    print(out)
    np.save('ising_3_bench_params',params)
    #np.save('rand_params',params)
    if not config.shadow:
        final_density_matrix_2 = prep_state(params, dev, config.qubit, qnode, config.depth, interaction_graph, ham_matrix)#prep_state(params, dev, config.qubit, qnode, config.depth, interaction_graph, ham_matrix, shadow_size=config.num_snapshots)#
    else:
      final_density_matrix_2 = prep_state(config.qubit, params, qnode, dev, config.depth, interaction_graph, shadow_size=config.num_snapshots)
      print(final_density_matrix_2.shape)
    #no shadow --> prepare_state(params, device, qubit, qnode, depth, interaction_graph, ham_matrix):
    #shadow def prepare_shadow_state(qubit,params, qnode, device, depth, interaction_graph,shadow_size=None):
    print(f'Final trace distance = {trace_distance(final_density_matrix_2,final_density_matrix)}')
    create_density_plot(final_density_matrix_2.real,f'shadow_{config.shadow}_vqt{config.save_name}')

    create_density_plot(np.abs(final_density_matrix-final_density_matrix_2),f'difference_{config.save_name}')



if __name__ == '__main__':
    main()
