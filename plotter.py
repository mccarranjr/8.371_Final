import numpy as np
from matplotlib import pyplot as plt


#entropy_no_shadow = np.loadtxt('results/3_heisenberg_no_shadow/entropies.txt')
#exp_h_no_shadow = np.loadtxt('results/3_heisenberg_no_shadow/exp_h.txt')

#entropy_shadow = np.loadtxt('results/3_heisenberg_shadow/entropies.txt')
#exp_h_shadow = np.loadtxt('results/3_heisenberg_shadow/exp_h.txt',dtype=complex)

#print(exp_h_shadow,'\n')
#print(exp_h_no_shadow)

#for i in range(entropy_shadow.shape[0]):
    #entropy_shadow[i] += i*.05

tr_dist = [.21971940433053175, 
0.22949340783861283,
0.178371402237098522, 
 0.16091311716818812, 
 0.15337346286107478, 
 0.16857591874050376,
0.09312550163250594]

out = np.array(tr_dist)

plt.figure()
plt.plot(np.array([500,1000,2000,4000,6000,8000,10000]),out)
plt.xlabel('Number of Snapshots Used In Each Iteration')
plt.ylabel('Trace Distance ')
plt.title('Trace Distance vs Number of Snapshot States')
plt.savefig('Final_snapshots')



#-2.0792714902324483 H
#1.607831713321773 S
'''
plt.figure()
plt.plot([50 * i for i in range(6)], np.abs(1.607831713321773 - np.array(entropy_no_shadow[:6])), label='Regular VQT')
plt.plot([50 * i for i in range(6)], np.abs(1.607831713321773 - np.array(entropy_shadow[:6])), label='Classical Shadow VQT')
plt.legend()
plt.xlabel('Iteration #')
plt.ylabel('Error')
plt.title('Magnitude of Entropy Error')
plt.savefig('plots/3_entropy')

plt.figure()
plt.plot([50 * i for i in range(6)], np.abs( -2.0792714902324483 - np.array(exp_h_no_shadow[:6])), label='Regular VQT')
plt.plot([50 * i for i in range(6)], np.abs(-2.0792714902324483 - np.array(exp_h_shadow[:6])), label='Classical Shadow VQT')
plt.legend()
plt.xlabel('Iteration #')
plt.ylabel('Error')
plt.title('Magnitude of Energy Error')
plt.savefig('plots/3_energy')

plt.figure()
plt.plot([i for i in range(1,11)],[0.39315471604791286, 0.4526588858162639,0.1404559,0.29423068,0.44407917667310964,0.4196147347208828,0.2754993829,.345565677,.314960302959,.239402])
plt.xlabel('Trial #')
plt.ylabel('Trace distance')
plt.title('Shadow VQT Trace Distances')
plt.savefig('plots/tr_dist')
'''