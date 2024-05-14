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

from functions import sigmoid, prob_dist, calculate_entropy
#from shadow_vqt import shadow_state_reconstruction


iteration=0
def exact_cost_no_shadow(params, qubit, qnode, depth, beta, interaction_graph, ham_matrix,shadow_size=None):
  global iteration
  #global qubit
  #print(iterations)
  #print(iteration)
  dist_params = params[0:qubit]
  params = params[qubit:]

  distribution = prob_dist(dist_params)

  combos = it.product([0,1], repeat = qubit)
  s = [list(c) for c in combos]

  final_cost = 0
  exp_h, entropies = [], []
  for i in s:
    result = qnode(params, range(qubit), depth, i, qubit, interaction_graph, ham_matrix)
    #result = result@result.conj().T

    #exp_val = np.trace(ham_matrix@result)
    #print(exp_val,'result')
    for j in range(0, len(i)):
      result = result*distribution[j][i[j]]
    final_cost += result

  entropy = calculate_entropy(distribution)
  final_final_cost = beta*final_cost - sum(entropy)
  #print(beta*final_cost,'beta * final')
  #print(sum(entropy))

  if (iteration%50 == 0):
    '''
    if iteration == 0:
      with open('results/3_heisenberg_no_shadow/exp_h.txt','w') as file_h:
        file_h.write(f'{final_cost}\n')
      with open('results/3_heisenberg_no_shadow/entropies.txt', 'w') as file_e:
        file_e.write(f'{sum(entropy)}\n')
    else:
      with open('results/3_heisenberg_no_shadow/exp_h.txt','a') as file_h:
        file_h.write(f'{final_cost}\n')
      with open('results/3_heisenberg_no_shadow/entropies.txt', 'a') as file_e:
        file_e.write(f'{sum(entropy)}\n')
      '''
    print(sum(entropy),'entropy')
    print(beta*final_cost)
    exp_h.append(final_cost)
    entropies.append(sum(entropy))
    print("Cost at Step "+str(iteration)+": "+str(final_final_cost))
  iteration += 1

  

  #np.save('results/3_heisenberg_no_shadow/exp_h', exp_h)
  #np.save('results/3_heisenberg_no_shadow/entropies', entropies)
  #print(len(entropies),'len entropy')

  return final_final_cost

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
    rho_snapshot = np.array([1],dtype=complex)
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot

def shadow_state_reconstruction(qubit, b_lists, obs_lists, num_snapshots):

    shadow_rho = np.zeros((2**qubit, 2**qubit), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])#each snapshot state has dim (512,512)
    return shadow_rho / np.trace(shadow_rho)


def exact_cost_shadow(params, qubit, qnode, depth, beta, interaction_graph, ham_matrix, shadow_size=10):
    global iteration
    qubits = range(qubit)
    h = ham_matrix
    dist_params = params[:qubit]
    params = params[qubit:]
    print(beta,'beta')
    distribution = prob_dist(dist_params)
    combos = it.product([0,1], repeat = qubit)
    s = [list(c) for c in combos]

    final_cost = 0
    for i in s:
        unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
        unitary_ids = np.random.randint(0, 3, size=(shadow_size, qubit))
        outcomes = np.zeros((shadow_size, qubit))

        for num_shadows in range(shadow_size):
            obs = [unitary_ensemble[int(unitary_ids[num_shadows,k])](k) for k in range(qubit)]
            #print_ = qnode(params, qubits, depth, i, qubit, interaction_graph, obs,ham_matrix=h)
            outcomes[num_shadows, :] = qnode(params, qubits, depth, i, qubit, interaction_graph, obs)
        result = 0
        '''
        for shadow in range(shadow_size):
          snap_shot = snapshot_state(outcomes[shadow], unitary_ids[shadow])
          snap_shot /= np.trace(snap_shot)
          result += np.trace(snap_shot@h)
        result /= shadow_size
        print(result,'result')
        '''
        rho_hat = shadow_state_reconstruction(qubit, outcomes, unitary_ids, shadow_size)
        #eigs, _ = np.linalg.eig(rho_hat)
        #print(eigs,'eigs')
        #if iterations == 5:
          #print('hi')
          #create_density_plot(rho_hat.real)
        #rho_hat=rho_hat.real
        result = np.trace(np.matmul(h, rho_hat)) 
        #result = print_
        #print(result,'gay')
        print(result,'result')
        for j in range(len(i)):
           result = result*distribution[j][i[j]]
        final_cost += result
    print(final_cost)

    entropy = sum(calculate_entropy(distribution))
    #entropy = np.trace(rho_hat@scipy.linalg.logm(rho_hat))
    #print(final_cost,'final_cost') 
    print(beta*final_cost,'beta * final cost')
    #print(sum(entropy), 'entropy')
    print(entropy,'entropy')
    final_final_cost = beta*final_cost - entropy# + np.trace(np.matmul(rho_hat, scipy.linalg.logm(rho_hat))))/2

    if (iteration%1 == 0):
      print("Cost at Step "+str(iteration)+": "+str(final_final_cost))
      
      if (iteration%50 == 0):
        if iteration == 0:
          with open('results/3_ising_shadow/exp_h.txt','w') as file_h:
            #print(final_cost,'GAYYYYYYYYYY')
            file_h.write(f'{final_cost}\n')
          with open('results/3_ising_shadow/entropies.txt', 'w') as file_e:
            file_e.write(f'{entropy}\n')
        else:
          with open('results/3_ising_shadow/exp_h.txt','a') as file_h:
            file_h.write(f'{final_cost}\n')
            #print(final_cost,'GAYYYYYYYYYY')

          with open('results/3_ising_shadow/entropies.txt', 'a') as file_e:
            file_e.write(f'{entropy}\n')

      

    iteration += 1

    return final_final_cost

def make_psd(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Set negative eigenvalues to 0
    eigenvalues[eigenvalues.real < 0] = 0
    
    # Reconstruct the matrix
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def updated_cost_shadow(params, qubit, qnode, depth, beta, interaction_graph, ham_matrix, shadow_size=500):
    global iteration
    qubits = range(qubit)
    h = ham_matrix

    final_cost = 0

    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, qubit))
    outcomes = np.zeros((shadow_size, qubit))

    for num_shadows in range(shadow_size):
        obs = [unitary_ensemble[int(unitary_ids[num_shadows,i])](i) for i in range(qubit)]
        outcomes[num_shadows, :] = qnode(params, qubits, depth, None, qubit, interaction_graph, obs)
    result = 0

    rho_hat = shadow_state_reconstruction(qubit, outcomes, unitary_ids, shadow_size)
    eigs, _ = np.linalg.eig(rho_hat)
    print(eigs,'eigs')
    rho_hat = make_psd(rho_hat)
    eigs, _ = np.linalg.eig(rho_hat)
    print(eigs,'eigs2')

    

    exp_h = np.trace(np.matmul(h, rho_hat)) 
    print(exp_h,'exp')

    #entropy = sum(calculate_entropy(distribution))
    entropy = np.trace(rho_hat@scipy.linalg.logm(rho_hat))
    print(beta*exp_h,'beta * final cost')
    #print(sum(entropy), 'entropy')
    print(entropy,'entropy')
    final_final_cost = beta*exp_h - entropy# + np.trace(np.matmul(rho_hat, scipy.linalg.logm(rho_hat))))/2

    if (iteration%1 == 0):
      print("Cost at Step "+str(iteration)+": "+str(final_final_cost))
      '''
      if (iteration%50 == 0):
        if iteration == 0:
          with open('results/3_heisenberg_shadow/exp_h.txt','w') as file_h:
            print(final_cost,'GAYYYYYYYYYY')
            file_h.write(f'{final_cost}\n')
          with open('results/3_heisenberg_shadow/entropies.txt', 'w') as file_e:
            file_e.write(f'{sum(entropy)}\n')
        else:
          with open('results/3_heisenberg_shadow/exp_h.txt','a') as file_h:
            file_h.write(f'{final_cost}\n')
            print(final_cost,'GAYYYYYYYYYY')

          with open('results/3_heisenberg_shadow/entropies.txt', 'a') as file_e:
            file_e.write(f'{sum(entropy)}\n')
        '''
      

    iteration += 1

    return final_final_cost 