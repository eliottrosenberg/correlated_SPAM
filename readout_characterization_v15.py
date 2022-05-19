import numpy as np
from scipy.stats import binom, chisquare, norm
from scipy.signal import find_peaks_cwt
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import math
from itertools import combinations
import time
import csv
import re
from tqdm import tqdm
rng = np.random.default_rng()

binosum_p_value = 0

def benchmark_experiment(backend,shots=88000,paulis_filename='SYK/paulis.csv',paulis_to_measure_filename='SYK/paulis_measured.csv',terms_measured_filename='SYK/terms_measured.csv',coeffs=[],job_SP=None,shots_SP=100000,delay=5e-4,skip_SP=True,params_from_job=None,use_measure_esp=False,reset_circuits=True):
    from qiskit import execute
    if not skip_SP:
        from SPAM_characterize_v2 import initialize, SP_circuits_submit, SPAM_parameters
        if job_SP == None:
            shots_SP = min(shots_SP,backend.configuration().max_shots)
            job_SP = SP_circuits_submit(backend,shots_SP)
    if params_from_job == None:
        paulis, coeffs = load_SYK(paulis_filename,coeffs)
        coeffs = coeffs.tolist()
        paulis = paulis.tolist()
        paulis = [ [int(q) for q in p] for p in paulis]
        paulis_to_measure = np.loadtxt(paulis_to_measure_filename,delimiter=',').tolist()
        paulis_to_measure = [ [int(q) for q in p] for p in paulis_to_measure]
        with open(terms_measured_filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|',quoting=csv.QUOTE_NONNUMERIC)
            terms_measured = [[int(_) for _ in r[:-1]] for r in reader]
        E = 0
        methods=['TNC','BFGS']
        for _ in range(100):
            E_i, theta_i, phi_i = product_state_H(paulis,coeffs,method=methods[_%2])
            print('E_i = '+str(E_i))
            if E_i < E:
                E = E_i
                theta = theta_i
                phi = phi_i
            if (not skip_SP and job_SP.done() and _ >= 10):
                break
            print('E = '+str(E))
        theta = theta.tolist()
        phi = phi.tolist()
    else:
        prev_tags = params_from_job.tags()
        paulis = read_from_tags('paulis',prev_tags)
        coeffs = read_from_tags('coeffs',prev_tags)
        paulis_to_measure = read_from_tags('paulis_to_measure',prev_tags)
        terms_measured = read_from_tags('terms_measured',prev_tags)
        E = read_from_tags('E',prev_tags)
        theta = read_from_tags('theta',prev_tags)
        phi = read_from_tags('phi',prev_tags)
        
        
    numMeasure, n = np.shape(paulis_to_measure)
    if skip_SP:
        num_circuits_for_SYK = (reset_circuits+1)*numMeasure
    else:
        num_circuits_for_SYK = 2*(reset_circuits+1)*numMeasure
    num_circuits_for_M = backend.configuration().max_experiments - num_circuits_for_SYK
    assert num_circuits_for_M > 20
    qc_M, bitstrs = readout_circuits_rand(n,num_circuits_for_M)
    
    if not skip_SP:
        p, dp2, r, dr2, ML_estimates = SPAM_parameters(job_SP,mc_trials=2)
        theta_SP = [m.x[3] for m in ML_estimates]
        phi_SP = [m.x[4] for m in ML_estimates]
        print('theta_SP = '+str(theta_SP))
        print('phi_SP = '+str(phi_SP))
        tags = ['SYK test','E = '+str(E),'N = '+str(2*n),'num_ham_terms = '+str(len(coeffs)),'num_measurements = '+str(numMeasure),'theta_SP = '+str(theta_SP),'phi_SP = '+str(phi_SP),'coeffs = '+str(coeffs),'paulis = '+str(paulis),'paulis_to_measure = '+str(paulis_to_measure),'terms_measured = '+str(terms_measured),'theta = '+str(theta),'phi = '+str(phi),'delay = '+str(delay),'use_measure_esp = '+str(use_measure_esp),'reset_circuits = '+str(reset_circuits)]+bitstrs
        qc_all = qc_M + qcs_for_H(paulis_to_measure,theta,phi,theta_SP,phi_SP,reset_circuits=reset_circuits) + qcs_for_H(paulis_to_measure,theta,phi,reset_circuits=reset_circuits)
    else:
        tags = ['SYK test','E = '+str(E),'N = '+str(2*n),'num_ham_terms = '+str(len(coeffs)),'num_measurements = '+str(numMeasure),'coeffs = '+str(coeffs),'paulis = '+str(paulis),'paulis_to_measure = '+str(paulis_to_measure),'terms_measured = '+str(terms_measured),'theta = '+str(theta),'phi = '+str(phi),'delay = '+str(delay),'use_measure_esp = '+str(use_measure_esp),'reset_circuits = '+str(reset_circuits)]+bitstrs
        qc_all = qc_M  + qcs_for_H(paulis_to_measure,theta,phi,reset_circuits=reset_circuits)
    
    job = execute(qc_all,backend,initial_layout=range(n),shots=shots,rep_delay = delay,job_tags=tags,optimization_level=0,use_measure_esp=use_measure_esp)
    return job


def benchmark_experiment_analyze(job):
    tags = job.tags()
    E = read_from_tags('E',tags)
    theta_SP = read_from_tags('theta_SP',tags)
    skip_SP = theta_SP == None
    
    part = readout_partition(job)
    
    
    print('E_exact = '+str(E))
    print('E_unmitigated = '+str(SYK_E_from_job(job,with_prerotations=False,readout_mitigate=False) ))
    if not skip_SP:
        print('E_prerotations = '+str(SYK_E_from_job(job,with_prerotations=True,readout_mitigate=False) ))
    print('E_uncorrelated_readout_no_prerotations = '+str(SYK_E_from_job(job,with_prerotations=False,readout_mitigate=True,part=part,uncorrelated=True) ))
    if not skip_SP:
        print('E_uncorrelated_readout_prerotations = '+str(SYK_E_from_job(job,with_prerotations=True,readout_mitigate=True,part=part,uncorrelated=True) ))
    print('E_correlated_readout_no_prerotations = '+str(SYK_E_from_job(job,with_prerotations=False,readout_mitigate=True,part=part) ))
    if not skip_SP:
        print('E_correlated_readout_and_prerotations = '+str(SYK_E_from_job(job,with_prerotations=True,readout_mitigate=True,part=part) ))


def measure_pauli_unmitigated(qubits_in_pauli, counts):
    shots = sum(counts.values())
    P = 0
    for bitstr in counts:
        Pi = 1 - 2*( sum(bitstr[-1-q]=='1' for q in qubits_in_pauli)%2 )
        prob = counts[bitstr]/shots
        P += Pi * prob
    return P
    
def measure_H_unmitigated(counts,which_paulis_in_each,paulis,coeffs):
    E = 0
    for i in range(len(counts)):
        for t in which_paulis_in_each[i]:
            E += coeffs[t] * measure_pauli_unmitigated( np.nonzero(paulis[t])[0], counts[i])
    return E


def measure_H_unmitigated_with_uncertainty(counts,which_paulis_in_each,paulis,coeffs):
    E = 0
    dE2 = 0
    shots = sum(counts[0].values())
    for i in range(len(counts)):
        for bitstr in counts[i]:
            prob = counts[i][bitstr]/shots
            d_prob2 = prob*(1-prob)/shots
            sum_commuting_terms = 0
            for t in which_paulis_in_each[i]:
                qubits_in_pauli = np.nonzero(paulis[t])[0]
                P_t = 1 - 2*( sum(bitstr[-1-q]=='1' for q in qubits_in_pauli)%2 )
                sum_commuting_terms += coeffs[t]*P_t
            E += sum_commuting_terms*prob
            dE2 += sum_commuting_terms**2 * d_prob2
    return E, np.sqrt(dE2)


def SYK_E_from_job(job,with_prerotations=False,readout_mitigate=False,part=None,uncorrelated=False,use_measure_H=False):
    tags = job.tags()
    reset_circuits = not 'reset_circuits = False' in tags
    num_measurements = read_from_tags('num_measurements',tags)
    terms_measured = read_from_tags('terms_measured',tags)
    coeffs = read_from_tags('coeffs',tags)
    paulis = read_from_tags('paulis',tags)
    if with_prerotations:
        starting_index = -2*(reset_circuits+1)*num_measurements+reset_circuits
    else:
        starting_index = -(reset_circuits+1)*num_measurements+reset_circuits
    counts_all = job.result().get_counts()
    counts = [counts_all[_] for _ in range(starting_index,starting_index+(reset_circuits+1)*num_measurements,(reset_circuits+1))]
    assert len(counts) == num_measurements
    if readout_mitigate or use_measure_H:
        E, dE = part.measure_H(counts,terms_measured,paulis,coeffs,uncorrelated,readout_mitigate=readout_mitigate)
    else:
        E, dE = measure_H_unmitigated_with_uncertainty(counts,terms_measured,paulis,coeffs) # dE is not exact
    return E, dE
    

def load_SYK(filename='SYK/paulis.csv',coeffs=[],method='L-BFGS-B',q=4,J=1):
    from math import comb, factorial
    paulis = np.loadtxt(filename,delimiter=',')
    if len(coeffs) > 0:
        return paulis, coeffs
    else:
        s = np.shape(paulis)
        n = s[1]
        numTerms = s[0]
        N = 2*n
        p = numTerms/comb(N,q)
        sigma = np.sqrt( factorial(q-1) * J**2 / (p * N**(q-1)))
        coeffs = rng.normal(0,sigma,numTerms)
        return paulis, coeffs
    

def product_state_H(paulis,coeffs,method='TNC',jac=True):
    # find the product state that approximates the ground state of the Hamiltonian described by (paulis, coeffs). paulis[i] has length n and paulis[i][j]
    # returns E, theta, phi
    n = len(paulis[0])    
        
    def energy(th,jac):
        theta = th[:n]
        phi = th[n:]
        E = 0
        dE = np.zeros(2*n)
        for i in range(len(coeffs)):
            factors = np.ones(2*n)
            d_factors = np.zeros(2*n)
            for q in range(n):
                if paulis[i][q] == 1:
                    factors[q] = np.sin(theta[q])
                    d_factors[q] = np.cos(theta[q])
                    factors[q+n] = np.cos(phi[q])
                    d_factors[q+n] = -np.sin(phi[q])
                elif paulis[i][q] == 2:
                    factors[q] = np.sin(theta[q])
                    d_factors[q] = np.cos(theta[q])
                    factors[q+n] = np.sin(phi[q])
                    d_factors[q+n] = np.cos(phi[q])
                elif paulis[i][q] == 3:
                    factors[q] = np.cos(theta[q])
                    d_factors[q] = -np.sin(theta[q])
            P = np.prod(factors)
            dP = np.zeros(2*n)
            for q in range(2*n):
                dP[q] = np.prod(factors[:q]) * np.prod(factors[(q+1):]) * d_factors[q]
            E += coeffs[i] * P
            dE += coeffs[i] * dP
        if jac:
            return E, dE
        else:
            return E
    
    
    from scipy.optimize import minimize
    th0 = np.random.rand(2*n)*2*np.pi
    sol = minimize( energy, th0, method=method,jac=jac,args=jac)
    #print(sol.message)
    E = sol.fun
    th = sol.x
    theta = th[:n]
    phi = th[n:]
    return E, theta, phi

def qcs_for_H(paulis_to_measure,theta,phi,theta_prerot=[],phi_prerot=[],option=0,reset_circuits=True):
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from SPAM_characterize_v2 import initialize
    n = len(paulis_to_measure[0])
    empty = QuantumCircuit(n,n)
    empty.measure(range(n),range(n))
    qcs = []
    for pauli in paulis_to_measure:
        if len(theta_prerot) == 0:
            qc = QuantumCircuit(n,n)
        else:
            qc = initialize(theta_prerot,phi_prerot,option)
        for q in range(n):
            qc.ry(theta[q],q)
            qc.rz(phi[q],q)
            if pauli[q] == 1:
                qc.h(q)
            elif pauli[q] == 2:
                qc.sdg(q)
                qc.h(q)
        qc.measure(range(n),range(n))
        qc = transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=3)
        qc = transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=3)
        if reset_circuits:
            qcs.append(empty)
        qcs.append(qc)
    
    return qcs


def read_from_tags(varName,tags):
    from numpy import array
    for tag in tags:
        sr = re.match(varName+' = ',tag)
        if sr != None:
            # fix issue with qubits_measured_all. Change the set to a list.
            if varName == 'qubits_measured_all':
                tag = tag[:22]+'['+tag[23:-1]+']'
            exec(tag)
            return eval(varName)
            break
            

def pool_bins_order(expected_counts,min_counts_in_bin=5,reverse_order = False):
    
    bins = []
    expected_counts_binned = []
    counts_bin = 0
    indices = set()
    if reverse_order:
        r = range(len(expected_counts)-1,-1,-1)
    else:
        r = range(len(expected_counts))
    for i in r:
        counts_bin += expected_counts[i]
        indices.add(i)
        if counts_bin >= 5:
            expected_counts_binned.append(counts_bin)
            bins.append(indices)
            counts_bin = 0
            indices = set()
            
    if counts_bin > 0:
        expected_counts_binned[-1] += counts_bin
        bins[-1] = bins[-1].union(indices)
            
    return bins, expected_counts_binned


def pool_bins(expected_counts,min_counts_in_bin=5):
    bins1, expected_counts_binned1 = pool_bins_order(expected_counts,min_counts_in_bin,False)
    bins2, expected_counts_binned2 = pool_bins_order(expected_counts,min_counts_in_bin,True)
    if len(expected_counts_binned1) > len(expected_counts_binned2):
        return bins1, expected_counts_binned1
    elif len(expected_counts_binned1) < len(expected_counts_binned2):
        return bins2, expected_counts_binned2
    else:
        if min(expected_counts_binned1) > min(expected_counts_binned2):
            return bins1, expected_counts_binned1
        else:
            return bins2, expected_counts_binned2



def load_qubit_map(machine,n):
  if machine == 'ibmq_montreal' or machine=='ibmq_toronto' or machine == 'ibmq_sydney' or machine == 'ibmq_mumbai':
    if n == 12:
        qubits = [1,4,7,10,12,13,14,11,8,5,3,2];
        #qubits = [12,15,18,21,23,24,25,22,19,16,14,13]
    elif n == 20:
        qubits = [1,4,7,10,12,15,18,21,23,24,25,22,19,16,14,11,8,5,3,2];
    else:
        qubits = np.arange(n)
  elif  machine=='ibmq_rochester':
      if n == 12:
          qubits = [21,22,23,24,25,29,36,35,34,33,32,28];
  elif machine=='ibmq_cambridge':
      if n == 12:
          qubits = [0,1,2,3,4,6,13,12,11,10,9,5];
          #qubits = [7,8,9,10,11,17,23,22,21,20,19,16];
          #qubits = [11,12,13,14,15,18,27,26,25,24,23,17];
      elif n == 20:
          qubits = [0,1,2,3,4,6,13,12,11,17,23,22,21,20,19,16,7,8,9,5];
          #qubits = [0,1,2,3,4,6,13,14,15,18,27,26,25,24,23,17,11,10,9,5];
          #qubits = [7,8,9,10,11,12,13,14,15,18,27,26,25,24,23,22,21,20,19,16];
      elif n == 24:
          qubits = [0,1,2,3,4,6,13,14,15,18,27,26,25,24,23,22,21,20,19,16,7,8,9,5];
  elif machine=='ibmq_16_melbourne':
      if n == 4:
          qubits = [2,3,11,12]; # 4 qubits
      elif n == 6:
          qubits = [0,1,2,12,13,14]; # 6 qubits
      elif n == 8:
          qubits = [0,1,2,3,11,12,13,14]; # 8 qubits
      elif n == 10:
          qubits = [0,1,2,3,4,10,11,12,13,14]; # 10 qubits
      elif n == 12:
          qubits = [0,1,2,3,4,5,9,10,11,12,13,14]; # 12 qubits
  elif machine == 'ibmq_manhattan' or machine == 'ibmq_brooklyn':
      if n == 12:
          qubits = [4,5,6,7,8,12,21,20,19,18,17,11]
      elif n == 20:
          qubits = [0,1,2,3,4,5,6,7,8,12,21,20,19,18,17,16,15,14,13,10]
      elif n == 44:
          qubits = [0,1,2,3,4,5,6,7,8,12,21,22,23,26,37,36,35,40,49,50,51,54,64,63,62,61,60,59,58,57,56,52,43,42,41,38,27,28,29,24,15,14,13,10]
      elif n == 52:
          qubits = [0,1,2,3,4,5,6,7,8,12,21,22,23,26,37,36,35,34,33,32,31,39,45,46,47,48,49,50,51,54,64,63,62,61,60,59,58,57,56,52,43,42,41,38,27,28,29,24,15,14,13,10]
      else:
          qubits = np.arange(n)
  elif machine == 'ibmq_guadalupe':
      if n == 12:
          qubits = [1,4,7,10,12,13,14,11,8,5,3,2]
      else:
          qubits = np.arange(n)
  else:
      qubits = np.arange(n)

  return qubits


def readout_circuits_rand(n,n_bitstr,add_id=False,add_barrier=True,add_reset_circuits=False,nx=1,nx0=0,measure_twice=False,barrier_after_measure=False):
    from qiskit import QuantumCircuit
    qc_all = []
    #rints = rng.integers(low=0,high=2**n,size=n_bitstr,dtype=np.uint64)
    if n <= 62:
        rints = rng.choice(2**n,size=n_bitstr,replace=False)
        bitstrs = [np.binary_repr(rint,width=n) for rint in rints]
    else:
        bitstrs_array = rng.integers(2,size=(n_bitstr,n),dtype=bool)
        bitstrs = [ ''.join([str(int( bitstrs_array[i,j])) for j in range(n)]) for i in range(n_bitstr)]
    
    qc_all = qc_from_bitstrs(bitstrs,add_id,add_barrier,add_reset_circuits,nx,nx0,measure_twice,barrier_after_measure)
    
    return qc_all, bitstrs


def qc_from_bitstrs(bitstrs,add_id=False,add_barrier=False,add_reset_circuits=True,nx=1,nx0=0,measure_twice=False,barrier_after_measure=True):
    from qiskit import QuantumCircuit
    qc_all = []
    n = len(bitstrs[0])
    for i in range(len(bitstrs)):
        if add_reset_circuits:
            qc = QuantumCircuit(n,n)
            qc.measure(range(n),range(n))
            qc_all.append(qc)
        if measure_twice:
            qc = QuantumCircuit(n,2*n)
        else:
            qc = QuantumCircuit(n,n)
        for q in range(n):
            if bitstrs[i][-1-q] == '1':
                qc.x([q]*nx)
            elif nx0 > 0:
                qc.x([q]*nx0)
            elif add_id:
                qc.id(q)
        if add_barrier:
            qc.barrier()
        if measure_twice:
            qc.measure(range(n),range(n))
            qc.barrier()
            qc.measure(range(n),range(n,2*n))
        else:
            qc.measure(range(n),range(n))
        if barrier_after_measure:
            qc.barrier()
        qc_all.append(qc)
    return qc_all
    
def qc_check_x_dependence(n,max_xs,add_reset_circuits=True):
    from qiskit import QuantumCircuit
    qc_all = []
    for nx in range(max_xs+1):
        if add_reset_circuits:
            qc = QuantumCircuit(n,n)
            qc.measure(range(n),range(n))
            qc_all.append(qc)
        qc = QuantumCircuit(n,n)
        if nx > 0:
            qc.x(list(range(n))*nx)
        qc.measure(range(n),range(n))
        qc_all.append(qc)
    return qc_all
    
def x_dependence_execute(n,max_xs,backend,add_reset_circuits=False,shots=8192):
    tags = ['x_dependence','max_xs = '+str(max_xs),'add_reset_circuits = '+str(add_reset_circuits)]
    from qiskit import execute
    qc_all = qc_check_x_dependence(n,max_xs,add_reset_circuits)
    if not backend.configuration().dynamic_reprate_enabled:
        print('dynamic rep-rate is not enabled')
        return execute(qc_all,backend,optimization_level=0,initial_layout=range(n),shots=shots,job_tags=tags)
    else:
        rep_delay = backend.configuration().rep_delay_range[1]
        return execute(qc_all,backend,optimization_level=0,initial_layout=range(n),shots=shots,job_tags=tags,rep_delay=rep_delay)

def check_correlation_type_circuits(n,max_xs,control_qubits,qubits_fixed_1,qubits_fixed_0,add_reset_circuits=True,num_trials=10):
    qc_all = []
    from qiskit import QuantumCircuit
    qc_all = []
    for _ in range(num_trials):
        for nx in range(max_xs+1):
            if add_reset_circuits:
                qc = QuantumCircuit(n,n)
                qc.measure(range(n),range(n))
                qc_all.append(qc)
            qc = QuantumCircuit(n,n)
            if len(qubits_fixed_1) > 0:
                qc.x(qubits_fixed_1)
            if nx > 0:
                qc.x(control_qubits*nx)
            other_qubits = set(range(n)) - set(control_qubits) - set(qubits_fixed_1) - set(qubits_fixed_0)
            other_qubits_x = [_ for _ in other_qubits if rng.integers(0,2,dtype=bool)]
            if len(other_qubits_x) > 0:
                qc.x(other_qubits_x)
            qc.measure(range(n),range(n))
            qc_all.append(qc)
            if len(control_qubits) == 2:
                for q in control_qubits:
                    if add_reset_circuits:
                        qc = QuantumCircuit(n,n)
                        qc.measure(range(n),range(n))
                        qc_all.append(qc)
                    qc = QuantumCircuit(n,n)
                    if nx > 0:
                        qc.x(control_qubits*nx)
                    qc.x(q)
                    if len(qubits_fixed_1) > 0:
                        qc.x(qubits_fixed_1)
                    other_qubits = set(range(n)) - set(control_qubits) - set(qubits_fixed_1) - set(qubits_fixed_0)
                    other_qubits_x = [_ for _ in other_qubits if rng.integers(0,2,dtype=bool)]
                    if len(other_qubits_x) > 0:
                        qc.x(other_qubits_x)
                    qc.measure(range(n),range(n))
                    qc_all.append(qc)
    return qc_all
    
def check_correlation_type_execute(n,max_xs,control_qubits,qubits_fixed_1,qubits_fixed_0,backend,add_reset_circuits=False,shots=8192,num_trials=10):
    tags = ['correlation_type_check','max_xs = '+str(max_xs),'add_reset_circuits = '+str(add_reset_circuits),'control_qubits = '+str(control_qubits),'qubits_fixed_1 = '+str(qubits_fixed_1),'num_trials = '+str(num_trials),'qubits_fixed_0 = '+str(qubits_fixed_0)]
    from qiskit import execute
    qc_all = check_correlation_type_circuits(n,max_xs,control_qubits,qubits_fixed_1,qubits_fixed_0,add_reset_circuits,num_trials)
    if not backend.configuration().dynamic_reprate_enabled:
        print('dynamic rep-rate is not enabled')
        return execute(qc_all,backend,optimization_level=0,initial_layout=range(n),shots=shots,job_tags=tags)
    else:
        rep_delay = backend.configuration().rep_delay_range[1]
        return execute(qc_all,backend,optimization_level=0,initial_layout=range(n),shots=shots,job_tags=tags,rep_delay=rep_delay)

def check_correlation_type_analyze(job,target_qubit,starting_register=0):
    counts = job.result().get_counts()
    if 'add_reset_circuits = True' in job.tags():
        counts = [counts[_] for _ in range(1,len(counts),2)]
    shots = sum(counts[0].values())
    n = len(list(counts[0].keys())[0])
    error_rate = []
    ideal = target_qubit in read_from_tags('qubits_fixed_1',job.tags())
    for counts_i in counts:
        e = 0
        for outcome_str in counts_i:
            if outcome_str[-1-target_qubit-starting_register] != str(int(ideal)):
                e += counts_i[outcome_str]
        error_rate.append(e)
    
    num_trials = read_from_tags('num_trials',job.tags())
    row_length = len(error_rate)//num_trials
    error_rate = np.reshape(error_rate,(num_trials,row_length))
    error_rate = np.mean(error_rate,0)
    
    return error_rate
    

def x_dependence_analyze(job):
    counts = job.result().get_counts()
    if 'add_reset_circuits = True' in job.tags():
        counts = [counts[_] for _ in range(1,len(counts),2)]
    shots = sum(counts[0].values())
    n = len(list(counts[0].keys())[0])
    error_rates = []
    ideal = False
    for counts_i in counts:
        error_rates_i = [0]*n
        for outcome_str in counts_i:
            for q in range(n):
                if outcome_str[-1-q] != str(int(ideal)):
                    error_rates_i[q] += counts_i[outcome_str]
        error_rates.append(error_rates_i)
        ideal = not ideal
    return error_rates


def optimize_Ising(J,hx,hz):
    # returns theta, E/n, corresponding to the product state that extremizes the mixed-field Ising Hamiltonian
    from numpy.polynomial.polynomial import Polynomial
    P = Polynomial((hz**2, 4*J*hz, 4*J**2 - hz**2 - hx**2, -4*J*hz, -4*J**2))
    c = P.roots()
    candidate_roots = np.array([ci for ci in c if abs(ci.imag) < 1e-5 and abs(ci) <= 1])
    theta = np.arccos(candidate_roots)
    E_n = -J*np.cos(theta)**2 - hx*np.sin(theta) - hz*np.cos(theta)
    sol_index = np.argmin(E_n)
    return theta[sol_index], E_n[sol_index]

def Ising_circuits(J,hx,hz,n,add_reset_circuits=True,barrier_after_measure=True):
    # return circuits for the product state that best approximates the ground state of the mixed-field Ising model
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    theta, E_n = optimize_Ising(J,hx,hz)
    qcz = QuantumCircuit(n,n)
    qcz.ry(theta,range(n))
    qcz.measure(range(n),range(n))
    qcz = transpile(qcz,basis_gates=['x','sx','rz','cx'],optimization_level=3)
    if barrier_after_measure:
        qcz.barrier()
    
    qcx = QuantumCircuit(n,n)
    qcx.ry(theta,range(n))
    qcx.h(range(n))
    qcx.measure(range(n),range(n))
    qcx = transpile(qcx,basis_gates=['x','sx','rz','cx'],optimization_level=3)
    if barrier_after_measure:
        qcx.barrier()
    
    if add_reset_circuits:
        id = QuantumCircuit(n,n)
        id.measure(range(n),range(n))
        qcs = [id, qcz, id, qcx]
    else:
        qcs = [qcz, qcx]
    return qcs

def Ising_energy_from_job_v0(job,mitigate=False,part=None,threshold=1e-7):
    tags = job.tags()
    hx = read_from_tags('hx',tags)
    hz = read_from_tags('hz',tags)
    J = read_from_tags('J',tags)
    counts = job.result().get_counts()
    if len(counts) > 2:
        counts = [counts[-3], counts[-1]]
    if mitigate:
        counts = [part.mitigate_counts(c,threshold) for c in counts]
    E = 0
    n = len(list(counts[0].keys())[0])
    shots = sum(counts[0].values())
    for outcome_str in counts[0]:
        for q in range(n):
            if outcome_str[-1-q] == '0':
                E -= hz * counts[0][outcome_str]/shots
            elif outcome_str[-1-q] == '1':
                E += hz * counts[0][outcome_str]/shots
            if outcome_str[-1-q] == outcome_str[-q]:
                E -= J*counts[0][outcome_str]/shots
            elif outcome_str[-1-q] != outcome_str[-q]:
                E += J*counts[0][outcome_str]/shots
    for outcome_str in counts[1]:
        for q in range(n):
            if outcome_str[-1-q] == '0':
                E -= hx * counts[1][outcome_str]/shots
            elif outcome_str[-1-q] == '1':
                E += hx * counts[1][outcome_str]/shots
    return E
    
    
def Ising_energy_from_job(job):
    tags = job.tags()
    hx = read_from_tags('hx',tags)
    hz = read_from_tags('hz',tags)
    J = read_from_tags('J',tags)
    counts = job.result().get_counts()
    if len(counts) > 2:
        counts = [counts[-3], counts[-1]]
    n = len(list(counts[0].keys())[0])
    shots = sum(counts[0].values())
    Z = np.zeros(n)
    X = np.zeros(n)
    ZZ = np.zeros(n)
    for outcome_str in counts[0]:
        for q in range(n):
            if outcome_str[-1-q] == '1':
                Z[q] += counts[0][outcome_str]/shots
            if outcome_str[-1-q] != outcome_str[-q]:
                ZZ[q] += counts[0][outcome_str]/shots
    for outcome_str in counts[1]:
        for q in range(n):
            if outcome_str[-1-q] == '1':
                X[q] +=  counts[1][outcome_str]/shots
    
    dZ = 2*np.sqrt(Z*(1-Z)/shots)
    dX = 2*np.sqrt(X*(1-X)/shots)
    dZZ = 2*np.sqrt(ZZ*(1-ZZ)/shots)
    
    Z = 1-2*Z
    X = 1-2*X
    ZZ = 1-2*ZZ
    
    E = -hz*sum(Z) - hx*sum(X) - J*sum(ZZ)
    dE = np.sqrt( hz**2 *np.sum(dZ**2) + hx**2 * np.sum(dX**2) + J**2 * np.sum(dZZ**2)   )
    
    return E, dE

def readout_circuits_rand_execute(n,n_bitstr,backend_device,shots=8192,add_id=False,add_barrier=False,add_reset_circuits=False,nx=1,nx0=0,bitstrs=[],measure_twice=False,barrier_after_measure=False,use_measure_esp=False):
    tags = ['readout_test','add_id = '+str(add_id),'add_barrier = '+str(add_barrier),'add_reset_circuits = '+str(add_reset_circuits),'nx = '+str(nx),'nx0 = '+str(nx0),'measure_twice = '+str(measure_twice),'barrier_after_measure = '+str(barrier_after_measure),'use_measure_esp = '+str(use_measure_esp)]
    from qiskit import execute
    if len(bitstrs) == 0:
        qc_all, bitstrs = readout_circuits_rand(n,n_bitstr,add_id,add_barrier,add_reset_circuits,nx,nx0,measure_twice,barrier_after_measure)
    else:
        qc_all = qc_from_bitstrs(bitstrs,add_id,add_barrier,add_reset_circuits,nx,nx0,measure_twice,barrier_after_measure)
    qubits = load_qubit_map(backend_device.name(),n)
    max_shots = backend_device.configuration().max_shots
    max_experiments = backend_device.configuration().max_experiments
    if shots > max_shots:
        print('Decreasing shots to max allowed = '+str(max_shots))
        shots = max_shots
    if n_bitstr <= max_experiments:
        if not backend_device.configuration().dynamic_reprate_enabled:
            print('dynamic rep-rate is not enabled')
            return execute(qc_all,backend_device,initial_layout=qubits,optimization_level=0,shots=shots,use_measure_esp=use_measure_esp,job_tags=tags+bitstrs)
        else:
            print('setting rep delay to max')
            return execute(qc_all,backend_device,initial_layout=qubits,optimization_level=0,shots=shots,use_measure_esp=use_measure_esp,job_tags=tags+bitstrs,rep_delay = backend_device.configuration().rep_delay_range[1])
    else:
        num_jobs = int(np.ceil(len(qc_all)/max_experiments))
        jobs_all = []
        for which_job in range(num_jobs):
            qc_slice = qc_all[max_experiments*which_job:min( max_experiments*(which_job+1), len(qc_all) )]
            bitstr_slice = bitstrs[max_experiments*which_job:min( max_experiments*(which_job+1), len(qc_all) )]
            if not backend_device.configuration().dynamic_reprate_enabled:
                print('dynamic rep-rate is not enabled')
                job = execute(qc_slice,backend_device,initial_layout=qubits,optimization_level=0,shots=shots,job_tags=tags+bitstr_slice)
            else:
                print('setting rep delay to max')
                job = execute(qc_slice,backend_device,initial_layout=qubits,optimization_level=0,shots=shots,job_tags=tags+bitstr_slice, rep_delay = backend_device.configuration().rep_delay_range[1])
            jobs_all.append(job)
        return jobs_all


def ising_experiment(n,n_bitstr,backend_device,shots,J=1,hx=1.5,hz=0.1,barrier_after_measure=True):
    from qiskit import execute
    qc_calib, bitstrs = readout_circuits_rand(n,n_bitstr,False,False,False,barrier_after_measure=barrier_after_measure)
    qc_ising = Ising_circuits(J,hx,hz,n,barrier_after_measure=barrier_after_measure)
    qc_all = qc_calib + qc_ising
    tags = ['readout_ising_experiment', 'J = '+str(J), 'hx = '+str(hx), 'hz = '+str(hz),'barrier_after_measure = '+str(barrier_after_measure)] + bitstrs
    job = execute(qc_all,backend_device,initial_layout=range(n),optimization_level=0,shots=shots,job_tags=tags, rep_delay = backend_device.configuration().rep_delay_range[1])
    return job
  
def expected_overlap(nA, nB, N, trials=10000,plot=False):
    # sample A contains nA choices from N, sample B contains nB choices from N. What is the expected intersection of set(sample A) and set(sample B)?
    
    sA = rng.integers(N,size=(nA,trials))
    sB = rng.integers(N,size=(nB,trials))
    h = np.zeros(N+1,dtype=np.uint)
    for i in range(trials):
        I = len(set(sA[:,i]).intersection(set(sB[:,i])))
        h[I] += 1
    if plot:
        plt.bar(np.arange(N+1),h)
        plt.xlabel('overlap')
        plt.show()
    return h
    
def min_overlap(nA, nB, N, trials):
    # sample A contains nA choices from N, sample B contains nB choices from N. What is the expected intersection of set(sample A) and set(sample B)?
    sA = rng.integers(N,size=(nA,trials))
    sB = rng.integers(N,size=(nB,trials))
    I_min = min(nA,nB)
    for i in range(trials):
        I = len(set(sA[:,i]).intersection(set(sB[:,i])))
        if I < I_min:
            I_min = I
            if I_min == 0:
                break
    return I_min


def min_overlap_parallel(nA, nB, N, trials,n_parallel=2):
    from multiprocessing import Process, Value
    # sample A contains nA choices from N, sample B contains nB choices from N. What is the expected intersection of set(sample A) and set(sample B)?
    sA = rng.integers(N,size=(nA,trials))
    sB = rng.integers(N,size=(nB,trials))
    I_min = Value('i',min(nA,nB))
    def I(i,I_min):
        I_i = len(set(sA[:,i]).intersection(set(sB[:,i])))
        if I_i < I_min.value:
            I_min.value = I_i
    
    threads = []
    for i in range(trials):
        p = Process(target=I, args=(i,I_min))
        p.start()
        threads.append(p)
        if i%n_parallel == n_parallel-1:
            for p in threads[-n_parallel:]:
                p.join()
                p.close()
        if I_min.value == 0:
            break
        
    return I_min.value
    
    
def check_whether_disjoint(bitstrs_upper,bitstrs_lower,which_qubits):
    bitstrs_upper = bitstrs_upper[:,which_qubits]
    bitstrs_lower = bitstrs_lower[:,which_qubits]
    bitstrs_upper = set(tuple(_) for _ in bitstrs_upper.tolist())
    bitstrs_lower = set(tuple(_) for _ in bitstrs_lower.tolist())
    I = bitstrs_upper.intersection(bitstrs_lower)
    return len(I)
    

def check_all_groupings_from_bitstrs(bitstrs,which_qubits):
    
    which_qubits = list(which_qubits)
    n = len(which_qubits)
    
    bitstrs = bitstrs[:,which_qubits]
    upper_bitstrs_strings = [ ''.join([ str(int(bi)) for bi in b]) for b in bitstrs]
    upper_bitstrs_numbers = [int(b,2) for b in upper_bitstrs_strings]
    frequencies = [ sum(b == i for b in upper_bitstrs_numbers) for i in range(2**len(which_qubits))]
    expected_frequencies = [ len(upper_bitstrs_numbers)/2**len(which_qubits) for _ in range(2**len(which_qubits))]
    chisq, p_value = chisquare(frequencies,expected_frequencies)
            
    return p_value, frequencies
        

def check_all_operators_from_bitstrs(bitstrs,min_size=1,max_size=float('inf'),parity_qubits_to_exclude=[],p_crit = 0.05):
    
    qubits_to_exclude = { list(q)[0] for q in parity_qubits_to_exclude if len(q) == 1}
    print('qubits_to_exclude = '+str(qubits_to_exclude))
    n = np.shape(bitstrs)[1]
    rn = [q for q in range(n) if q not in qubits_to_exclude]
    shots = len(bitstrs)
    for nOne in range(min_size,max_size+1):
        print('checking all operators on '+str(nOne)+' qubits')
        z_crit = 1-2*binom.ppf( 0.5*(1- (1-p_crit)**(1/(n* math.comb(n,nOne)* math.comb(2**nOne,2**(nOne-1))))), shots, 0.5)/shots
        print('z_crit = '+str(z_crit))
        if z_crit == 1:
            break
        basis = 2**np.arange(nOne)
        for qubits_i in combinations(rn,nOne):
            if qubits_i in parity_qubits_to_exclude:
                continue
            #bitstrs_numbers = [ int(''.join([str(int(bitstrs[i,j])) for j in qubits_i]),2) for i in range(shots)] # order of bits is arbitrary
            bitstrs_numbers = bitstrs[:,np.array(qubits_i)] @ basis
            #unique_bitstr_numbers, num_occurances = np.unique(bitstrs_numbers, return_counts=True)
            #v1 = coo_matrix((num_occurances, (np.zeros(len(unique_bitstr_numbers)),unique_bitstr_numbers)),shape=(1,2**nOne))
            for op in combinations(range(2**nOne),2**(nOne-1)):
                #v2 = coo_matrix((np.ones(2**(nOne-1)), (np.array(op),np.zeros(2**(nOne-1)))),shape=(2**nOne,1))
                Z = -2*sum( (i in op) for i in bitstrs_numbers)/shots + 1
                #Z = -2*(v1@v2).toarray()[0,0]/shots + 1
                if abs(Z) > z_crit:
                    print('found z = '+str(Z))
                    print('qubits = '+str(qubits_i))
                    
                    # need to compute upper and lower.
                    if Z < 0:
                        # states in op match those in bitstrs
                        upper = {tuple(int(_) for _ in np.binary_repr(op_i,nOne)[::-1]) for op_i in op}
                        lower = {tuple( int(_) for _ in np.binary_repr(i,nOne)[::-1]) for i in range(2**nOne) if i not in op}
                    elif Z > 0:
                        lower = {tuple(int(_) for _ in np.binary_repr(op_i,nOne)[::-1]) for op_i in op}
                        upper = {tuple( int(_) for _ in np.binary_repr(i,nOne)[::-1]) for i in range(2**nOne) if i not in op}
                    return [qubits_i], lower, upper
    return False, False, False

def expected_variance_of_variance(n,p,s):
    if s == 1:
        # if s is 1, variance is 0
        return 0
    mu2 = n*p*(1-p)
    mu4 = n*p*(1-p)*(1 + (3*n-6)*p*(1-p))
    
    return mu4/s - mu2**2 * (s-3)/(s*(s-1))

def check_all_z_from_bitstrs(bitstrs,reverse=False,min_size=1,max_size=float('inf'),max_memory_gb=200,parity_qubits_to_exclude=[],p_crit=0.05,reduce_zcrit=False):
    
    qubits_to_exclude = { list(q)[0] for q in parity_qubits_to_exclude if len(q) == 1}
    
    n = np.shape(bitstrs)[1]
    rn = [q for q in range(n) if q not in qubits_to_exclude]
    if reverse:
        a1 = min(n-len(qubits_to_exclude),max_size)
        a2 = min_size-1
        a3 = -1
    else:
        a1 = min_size
        a2 = min(n-len(qubits_to_exclude)+1,max_size+1)
        a3 = 1
    for nOne in range(a1,a2,a3):
        print('starting nOne = '+str(nOne))
        
        # identify z_crit
        shots = len(bitstrs)
        z_crit = 1-2*binom.ppf( 0.5*(1- (1-p_crit)**(1/(2*n* math.comb(n,nOne)))), shots, 0.5)/shots
        print('z_crit = '+str(z_crit))
        if reduce_zcrit and z_crit >= 1:
            print('reducing z_crit')
            z_crit = 0.9
        
        
        which_comb = 0
        n_comb = math.comb(n-1,nOne)
        
        num_parallel_choices = int(max_memory_gb * 25/30.55 * 1024**3 / (8*len(bitstrs)))
        
        for qubits_i in combinations(rn,nOne):
                   
            if which_comb%num_parallel_choices == 0:
                # initialize new matrix
                Z = None
                qubits = np.zeros((n,min(num_parallel_choices,n_comb-which_comb)))
            qubits[list(qubits_i),(which_comb%num_parallel_choices)] = 1
            if (which_comb%num_parallel_choices == num_parallel_choices-1) or which_comb == n_comb-1:
                # matrix is full. compute Z
                Z = 1-2*np.mean( (bitstrs @ qubits)%2, 0)
                if max(abs(Z)) > z_crit:
                    print('found z = '+str(max(abs(Z))))
                    qubits = qubits[:,abs(Z)>z_crit]
                    Z = Z[abs(Z)>z_crit]
                    qubits = [{ _ for _ in range(n) if q[_] > 0} for q in qubits.T]
                    good_indices = [_ for _ in range(len(qubits)) if qubits[_] not in parity_qubits_to_exclude]
                    Z = Z[good_indices]
                    qubits = [qubits[g] for g in good_indices]
                    if len(qubits) > 0:
                        print('qubits = '+str(qubits))
                        return qubits
                    
            which_comb += 1
            
    return False


def integers_between(n1,n2,include_endpoint=False):
    if int(n1) == n1:
        lower = int(n1)
    else:
        lower = int(np.floor(n1))+1
        
    if (int(n2) == n2 and include_endpoint) or int(n2) != n2:
        upper = int(n2)+1
    else:
        upper = int(n2)
    
    return np.arange(lower,upper)

    
def str_list_to_array(str_list):
    n = len(str_list[0])
    L = np.empty((len(str_list),n),dtype=bool)
    for i in range(len(str_list)):
        for j in range(n):
            L[i,j] = int(str_list[i][-1-j])
    return L
    

def binosum_chi2(p_vec,n_flip,shots,weights='equal',return_p_value=False):
    # Computes the chi2 value, assuming that n_flip is distributed according to a sum of binomial distributions, each with their corresponding weights and p
    
    if weights == 'equal':
        weights = [1/len(p_vec) for _ in p_vec]
    
    n_flip = np.array(n_flip)
    
    lower = int(min(n_flip)) ## possibly change this.
    upper = int(max(n_flip))
    
    actual_counts = np.array([ np.sum(n_flip == _) for _ in range(lower,upper+1)])
    x = np.arange(lower,upper+1)
    expected_counts = sum( binom.pmf(x,shots,p_vec[i])*len(n_flip)*weights[i] for i in range(len(p_vec)) )
    expected_counts[0] = sum( binom.cdf(lower,shots,p_vec[i])*len(n_flip)*weights[i] for i in range(len(p_vec)) )
    expected_counts[-1] = sum( binom.sf(upper-1,shots,p_vec[i])*len(n_flip)*weights[i] for i in range(len(p_vec)) )
    
    # pool to make sure there are at least 5 counts in each bin:
    bins, expected_counts = pool_bins(expected_counts)
    actual_counts = [sum( actual_counts[_] for _ in indices) for indices in bins]
    
    
    chisq, p_value = chisquare(actual_counts,expected_counts)
    
    global binosum_p_value
    binosum_p_value = p_value
    
    if return_p_value:
        return p_value
    else:
        return chisq


def binosum_chi2_for_optimization(p_vec,n_flip,shots,weights='equal',return_p_value=False):
    # Computes the chi2 value, assuming that n_flip is distributed according to a sum of binomial distributions, each with their corresponding weights and p
    
    if weights == 'equal':
        weights = [1/len(p_vec) for _ in p_vec]
    
    n_flip = np.array(n_flip)
    
    actual_counts = np.array([ np.sum(n_flip == _) for _ in range(shots+1)])
    x = np.arange(shots+1)
    expected_counts = sum( binom.pmf(x,shots,p_vec[i])*len(n_flip)*weights[i] for i in range(len(p_vec)) )
    #expected_counts = np.array([ sum(binom.pmf(_,shots,p_vec[i])*len(n_flip)*weights[i] for i in range(len(p_vec))) for _ in range(shots+1)])
    indices_to_keep = np.where(expected_counts)
    actual_counts = actual_counts[indices_to_keep]
    expected_counts = expected_counts[indices_to_keep]
    
    chisq, p_value = chisquare(actual_counts,expected_counts)
    
    global binosum_p_value
    binosum_p_value = p_value
    
    if return_p_value:
        return p_value
    else:
        return chisq


def binosum_callback(p_vec):
    return binosum_p_value > 0.99

def guess_binosum(n_flip,shots,n_binos):
    p_guess = np.mean(n_flip)/shots
    if n_binos > 1:
        actual_counts = np.array([ np.sum(n_flip == _) for _ in range(shots+1)])
        widths = np.sqrt(p_guess*(1-p_guess)/shots) * np.ones(n_binos)
        try:
            p_guess = find_peaks_cwt(actual_counts,widths=widths)/shots
            print('p_guess = '+str(p_guess))
        except:
            p_guess = rng.normal(p_guess, np.sqrt(p_guess*(1-p_guess)/shots), (n_binos,))
        if len(p_guess) != n_binos:
            p_guess = rng.normal(p_guess, np.sqrt(p_guess*(1-p_guess)/shots), (n_binos,))
        p_guess = np.sort(np.abs(p_guess))
    else:
        p_guess = [p_guess]
    return p_guess
    
def fit_n_binosum(n_flip,shots,n_binos,method='Nelder-Mead'):
    p_guess = guess_binosum(n_flip,shots,n_binos)
    if method == 'Nelder-Mead':
        return minimize(binosum_chi2_for_optimization, p_guess,args=(n_flip,shots), method=method, callback = binosum_callback )
    else:
        return minimize(binosum_chi2_for_optimization, p_guess,args=(n_flip,shots), bounds = [ (0,1) for _ in range(n_binos)] , method=method, callback = binosum_callback )


def bitstrs_from_tags(tags):
    for i in range(len(tags)):
        if tags[i][0] == '0' or tags[i][0] == '1':
            break
    return tags[i:]


def load_from_file(filename):
    # loads a readout partition from a file but uses the updated version of the class.
    import pickle
    part = pickle.load(open(filename,"rb"))
    part_new = readout_partition(part.counts,part.ideal_strs,False,part.backend_name)
    part_new.shots = part.shots
    part_new.n = part.n
    part_new.partitions = part.partitions
    return part_new


class noise_model_simulator_aer:
    # **** not working ****
    # for now, assume just two qubits
    def __init__(self,e):
        # e[target_qubit,target_qubit_state,control_qubit_state]
        R = np.empty((4,4))
        for init_state in range(4):
            init_bitstr = np.binary_repr(init_state,2)
            for final_state in range(4):
                fin_bitstr = np.binary_repr(final_state,2)
                p = 1
                for q in range(2):
                    error =  fin_bitstr[1-q] != init_bitstr[1-q]
                    target_qubit_state = int(init_bitstr[-1-q])
                    control_qubit_state = int(init_bitstr[-1-(1-q)])
                    if error:
                        p = p*e[q,target_qubit_state,control_qubit_state]
                    else:
                        p = p* (1-e[q,target_qubit_state,control_qubit_state])
                R[init_state,final_state] = p
                
        from qiskit.providers.aer.noise import NoiseModel, ReadoutError
        readout_model = ReadoutError(R)
        N = NoiseModel()
        N.add_readout_error(R,[0,1])
        self.noise_model = N
    
    def run(self,circuits,shots=10000):
        from qiskit import Aer, execute
        sim = Aer.get_backend('qasm_simulator')
        job = execute(circuits, sim,shots=shots,basis_gates=self.noise_model.basis_gates ,noise_model=self.noise_model)
        return job


class noise_model_simulator:
    # for now, assume just two qubits
    def __init__(self,e):
        # e[target_qubit,target_qubit_state,control_qubit_state]
        R = np.empty((4,4))
        for init_state in range(4):
            init_bitstr = np.binary_repr(init_state,2)
            for final_state in range(4):
                fin_bitstr = np.binary_repr(final_state,2)
                p = 1
                for q in range(2):
                    error =  fin_bitstr[1-q] != init_bitstr[1-q]
                    target_qubit_state = int(init_bitstr[-1-q])
                    control_qubit_state = int(init_bitstr[-1-(1-q)])
                    if error:
                        p = p*e[q,target_qubit_state,control_qubit_state]
                    else:
                        p = p* (1-e[q,target_qubit_state,control_qubit_state])
                R[final_state,init_state] = p
        self.response_matrix = R
    
    def run(self,circuits,shots):
        from qiskit import Aer, execute
        sim = Aer.get_backend('statevector_simulator')
        job = execute(circuits,sim)
        P_noiseless = [job.result().get_statevector(_).probabilities() for _ in range(len(circuits))]
        P = [self.response_matrix @ job.result().get_statevector(_).probabilities() for _ in range(len(circuits))]
        sample = [ rng.multinomial(shots,Pi) for Pi in P]
        result = []
        for s in sample:
            d = {}
            for _ in range(len(s)):
                d[np.binary_repr(_,2)] = s[_]
            result.append(d)
        return result


class readout_partition:
    def __init__(self,counts_or_job,ideal_strs='from_tags',run_partition=True,backend_name='ibmq_manhattan',condition_on_previous_state=True,min_n_control=0,max_n_control=3,min_error_rates=1,starting_register=0):
        # for repeated measurements, starting_register = 0 gives the second measurement, starting_register = n gives the first measurement
        self.backend_name = backend_name
        if str(type(counts_or_job)) == "<class 'qiskit.providers.ibmq.job.ibmqjob.IBMQJob'>":
            self.counts = counts_or_job.result().get_counts()
            ideal_strs = bitstrs_from_tags(counts_or_job.tags())
            self.backend_name = counts_or_job.backend().name()
        elif str(type(counts_or_job)) == "<class 'list'>":
            self.counts = counts_or_job
        if str(type(ideal_strs)) == "<class 'list'>":
            ideal_strs = str_list_to_array(ideal_strs)
        self.ideal_strs = ideal_strs
        self.shots = sum(self.counts[0].values())
        self.n = np.shape(ideal_strs)[1]
        self.partitions = [ [readout_partition_qubit(self.counts, q, g, ideal_strs,run_partition,min_n_control,max_n_control,min_error_rates,condition_on_previous_state,starting_register) for g in [0,1]] for q in range(self.n)]
        #self.minimize_islands()
        
    def minimize_islands(self):
        # repartition to minimize islands
        from itertools import product
        undetermined = []
        possible_controls = []
        possible_keys = []
        for q in range(self.n):
            for given in range(2):
                if self.partitions[q][given].multiple_options:
                    undetermined.append( (q,given) )
                    possible_controls.append( self.partitions[q][given].possible_control_qubits )
                    possible_keys.append( self.partitions[q][given].possible_keys )
        
        print('The following (qubit, given) had multiple possible control qubits:')
        print(undetermined)
        
        smallest_island = 4**self.n
        degenerate = False
        for controls in product(*possible_controls):
            for _ in range(len(undetermined)):
                self.partitions[undetermined[_][0]][undetermined[_][1]].control_qubits = controls[_]
            islands_i = self.get_islands()
            sizes = np.array([len(I) for I in islands_i])
            island_size_i = np.sum( 4**sizes )
            if island_size_i < smallest_island:
                smallest_island = island_size_i
                degenerate = False
                best_controls = controls
            elif island_size_i == smallest_island:
                degenerate = True
        print('Partitions selected to minimize island size.')
        if degenerate:
            print('Warning: multiple partitions gave this island size.')
            
        # need to set n_flip, indices, and key. But let's test this first.
        ##### Not finished
        
    def error_rate(self,qubit,given):
        n_flip = self.partitions[qubit][given].n_flip
        return [np.mean(n_flip_i)/self.shots for n_flip_i in n_flip]
    
    def connected_qubits(self,qubit,given):
        c = set()
        for s in self.partitions[qubit][given].control_qubits:
            c.add(s)
        return c
    
    def plot_qubit(self,qubit,given,condition_on_previous_state=False):
        self.partitions[qubit][given].plot(condition_on_previous_state=condition_on_previous_state)
        
    def write_table(self,filename):
        qubits = load_qubit_map(self.backend_name,self.n)
        with open(filename,'w',newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            writer.writerow(['Qubit', 'Given', 'Correlated with qubits', 'Loop Qubit', 'Correlated with loop qubits', 'Successfully separated?', 'Error rates'])
            for q in range(self.n):
                for given in [0,1]:
                    part = self.partitions[q][given]
                    separated = part.separated
                    error_rates = [np.mean(n_flip_i)/part.shots for n_flip_i in part.n_flip]
                    correlated_qubits = [{qubits[ci] for ci in c} for c in part.control_qubits]
                    writer.writerow([qubits[q],given,correlated_qubits,q,part.control_qubits,separated]+error_rates)
    
    def plot(self,include_secondary_xs=True):
        from device_plotter import plot_device
        qubits = load_qubit_map(self.backend_name,self.n)
        avg_error_rate = np.array([sum( np.mean(self.error_rate(q,given)) for given in [0,1])/2 for q in range(self.n)])
        affected_by = [self.connected_qubits(q,0).union(self.connected_qubits(q,1)) for q in range(self.n)]
        arrows_between = []
        for i in range(self.n):
            for j in affected_by[i]:
                arrows_between.append((j,i))
        separated = [self.partitions[q][0].separated == True and self.partitions[q][1].separated == True for q in range(self.n)]
        xs_over = [i for i in range(self.n) if not separated[i]]
        plt.close()
        plot_device(self.backend_name,avg_error_rate,arrows_between, xs_over,qubits,ylabel='average SPAM error rate',include_secondary_xs=include_secondary_xs)
    def compare_to_IBM(self,job_or_properties):
        if str(type(job_or_properties)) == "<class 'qiskit.providers.ibmq.job.ibmqjob.IBMQJob'>":
            properties = job_or_properties.properties()
        else:
            properties = job_or_properties
            
        for given in [0,1]:
            if given == 0:
                prop_str = 'prob_meas1_prep0'
            elif given == 1:
                prop_str = 'prob_meas0_prep1'
            e = np.array([properties.qubit_property(q,prop_str)[0] for q in range(self.n)])
            de = np.sqrt(e*(1-e)/5000)
            plt.errorbar(np.arange(self.n), e, de, capsize=10,fmt='none',label='IBM reported, given = '+str(given))
            e_obs = [self.error_rate(q,given) for q in range(self.n)]
            max_groups = max( [len(_) for _ in e_obs])
            for g in range(max_groups):
                q_incl = []
                e_obs_group = []
                de_obs_group = []
                for q in range(self.n):
                    if len(e_obs[q]) > g:
                        q_incl.append(q)
                        e_obs_group.append(e_obs[q][g])
                        de_obs_group.append( np.sqrt( e_obs[q][g]*(1- e_obs[q][g])/(self.shots * len(self.partitions[q][given].n_flip[g]))))
                plt.errorbar(q_incl, e_obs_group, de_obs_group, capsize=10,fmt='none',label='observed, given = '+str(given))
        plt.legend(loc='best')
        plt.xlabel('qubit')
        plt.ylabel('error rate')
        plt.show()
    def response_matrix_element(self,meas_state,given_state,qubits=None,uncorrelated=False,output_dP=True):
        # returns the probability of obtaining meas_state if the qubits are prepared in given_state.
        # inputs can be either binary strings or ints.
        
        if qubits == None:
            qubits = list(range(self.n))
        n = len(qubits)
        
        if type(meas_state) is int:
            meas_state = np.binary_repr(meas_state,n)
        if type(given_state) is int:
            given_state = np.binary_repr(given_state,n)
        if len(meas_state) != n or len(given_state) != n:
            print('Warning: make sure that input states have '+str(n)+' binary digits')
        
        if output_dP:
            factors = np.empty(n)
            d_factors2 = np.empty(n)
        else:
            P = 1
        for q in range(n):
            meas = int(meas_state[-1-q])
            given = int(given_state[-1-q])
            if self.partitions[qubits[q]][given].n_error_rates > 1:
                control_qubits = self.partitions[qubits[q]][given].control_qubits
                control_qubit_state = ''.join([given_state[-1-qubits.index(c)] for c in np.flip(control_qubits)]) # check that this is correct
                control_qubit_state_int = int(control_qubit_state,2)
                which_group = self.partitions[qubits[q]][given].key[control_qubit_state_int]
            else:
                which_group = 0
            if not uncorrelated:
                n_flip = self.partitions[qubits[q]][given].n_flip[which_group]
            elif uncorrelated:
                n_flip = []
                for n_flip_i in self.partitions[qubits[q]][given].n_flip:
                    n_flip += n_flip_i
            tot_shots = self.shots * len(n_flip)
            e = np.mean(n_flip)/self.shots
            if output_dP:
                de2 = e*(1-e)/tot_shots
                if meas == given:
                    factors[q] = 1-e
                else:
                    factors[q] = e
                d_factors2[q] = de2
            else:
                if meas == given:
                    P = P*(1-e)
                else:
                    P = P*e
                

        if output_dP:
            P = np.prod(factors)
            factors2 = factors**2
            dP2 = sum( np.prod(factors2[:q])*d_factors2[q]*np.prod(factors2[q+1:]) for q in range(n) )
            dP = np.sqrt(dP2)
            
            return P, dP
        else:
            return P
    
    def get_islands(self):
        islands = []
        for q in range(self.n):
            already_included = len(islands) > 0 and max([q in i for i in islands])
            if already_included:
                continue
            else:
                island_i = self.grow_island(q)
                islands.append(island_i)
        return islands
    
    def all_connected_qubits(self,qubit):
        s = self.connected_qubits(qubit,0).union(self.connected_qubits(qubit,1))
        for q in range(self.n):
            if qubit in self.connected_qubits(q,0) or qubit in self.connected_qubits(q,1):
                s.add(q)
        return s
    
    def grow_island(self,q):
        island = {q}
        to_add = self.all_connected_qubits(q)
        finished = len(to_add) == 0
        while not finished:
            island = island.union(to_add)
            to_add_new = set()
            for qi in to_add:
                to_add_new = to_add_new.union( self.all_connected_qubits(qi) )
            finished = len(island.union(to_add_new)) == len(island)
            to_add = to_add_new
        return list(island)
    
    def island_response_matrix_element(self,which_island,meas_state,given_state,uncorrelated=False):
        islands = self.get_islands()
        qubits = islands[which_island]
        P, dP = self.response_matrix_element(meas_state,given_state,qubits,uncorrelated=uncorrelated)
        return P, dP
    
    def island_response_matrix(self,which_island,output_dR = True,uncorrelated=False):
        islands = self.get_islands()
        qubits = islands[which_island]
        n = len(qubits)
        R = np.empty((2**n,2**n))
        if output_dR:
            dR = np.empty((2**n,2**n))
        for i in tqdm(range(2**n)):
            for j in range(2**n):
                if output_dR:
                    P, dP = self.response_matrix_element(i,j,qubits,uncorrelated=uncorrelated)
                    dR[i,j] = dP
                else:
                    P = self.response_matrix_element(i,j,qubits,output_dP=False,uncorrelated=uncorrelated)
                R[i,j] = P     
        if output_dR:
            return R, dR
        else:
            return R
    
    def response_matrix(self):
        R = np.empty((2**self.n, 2**self.n))
        dR = np.empty((2**self.n, 2**self.n))
        for i in range(2**self.n):
            for j in range(2**self.n):
                P, dP = self.response_matrix_element(i,j)
                R[i,j] = P
                dR[i,j] = dP
        return R, dR
    
    def apply_inverse_response_matrix(self,obs_states,obs_probs,which_island=None,try_exact=True):
        
        # if which_island == None, then use the full response matrix
        if which_island == None:
            qubits = None
            n = self.n
        else:
            islands = self.get_islands()
            qubits = islands[which_island]
            n = len(qubits)
        v0 = np.zeros(2**n)
        for _ in range(len(obs_states)):
            v0[obs_states[_]] = obs_probs[_]    
        
        # first, try exact inversion:
        if try_exact and which_island != None:
            from numpy.linalg import inv
            R = self.island_response_matrix(which_island,output_dR=False)
            sol_exact = inv(R)@v0
            
            if min(sol_exact) >= 0 and max(sol_exact) <= 1:
                return sol_exact
            else:
                print('Matrix inversion gives a non-physical solution. Try optimization routine:')
    

        
        
        # next, find the vector satisfying the constraints that has the largest inner product with ref_vec.
        from scipy.optimize import Bounds, NonlinearConstraint, minimize, LinearConstraint
        bound = Bounds(0,1)
        #constr = NonlinearConstraint(np.sum, 1, 1)
        #constr = LinearConstraint(np.ones((2**n,2**n)),1,1)
        constr = {'type': 'eq', 'fun': (lambda x : np.sum(x) - 1), 'jac': (lambda x : np.ones(len(x)))}
        def cost(v):
            
            C = 0
            dC = np.zeros(2**n)
            for i in range(2**n):
                Rv_i = sum( self.response_matrix_element(i,j,qubits)[0] * v[j] for j in range(2**n) )
                if i in obs_states:
                    Rv_minus_p_i =  Rv_i - obs_probs[obs_states.index(i)]
                else:
                    Rv_minus_p_i = Rv_i
                C += Rv_minus_p_i**2
                for j in range(2**n):
                    dC[j] += 2*self.response_matrix_element(i,j,qubits)[0] * v[j] * Rv_minus_p_i
            
            print(C, sum(v))
            return C, dC
        
        #v0 = np.random.rand(2**n)
        #v0 = v0/sum(v0)
        sol = minimize(cost,v0, method='SLSQP',jac=True,bounds=bound,constraints=constr)
        return sol
                
    
    def mitigate_ising(self,job,hx=None,hz=None,J=None):
        
        
        tags = job.tags()
        if hx == None:
            hx = read_from_tags('hx',tags)
        if hz == None:
            hz = read_from_tags('hz',tags)
        if J == None:
            J = read_from_tags('J',tags)
        counts = job.result().get_counts()
        if len(counts) > 2:
            counts = [counts[-3], counts[-1]]
        
        shots = sum(counts[0].values())
        from numpy.linalg import inv
        islands = self.get_islands()
        print('constructing inverse response matrix')
        R_all = [ self.island_response_matrix(i,output_dR = False) for i in range(len(islands))]
        Rinv_all = [ inv(R) for R in R_all]
        
        # next, compute the bit parities needed for the Ising Hamiltonian
        def Z_mitigated(measured_bitstring):
            # computes the error-mitigated \sum_i Z_i for the measured bitstring
            Z = 0
            substrs = [ ''.join([ measured_bitstring[-1-i] for i in np.flip(island)]) for island in islands]
            in_state = [ int(substr,2) for substr in substrs]
            for which_island in range(len(islands)):
                # probabilities of out_states are Rinv_all[which_island][:,in_state[which_island]]
                n_island = len(islands[which_island])
                for out_state in range(2**n_island):
                    n1 = np.binary_repr(out_state).count('1')
                    Z += (n_island - 2*n1)*Rinv_all[which_island][out_state,in_state[which_island]]
            return Z
            
        def ZZ_mitigated(measured_bitstring):
            ZZ = 0
            substrs = [ ''.join([ measured_bitstring[-1-i] for i in np.flip(island)]) for island in islands]
            in_state = [ int(substr,2) for substr in substrs]
            for q in range(self.n):
                island_q = [q in i for i in islands].index(True)
                island_qpp = [(q+1)%self.n in i for i in islands].index(True)
                island_qubit_q = islands[island_q].index(q)
                island_qubit_qpp = islands[island_qpp].index((q+1)%self.n)
                
                if island_q == island_qpp:
                    n_island = len(islands[island_q])
                    for out_state in range(2**n_island):
                        out_state_str = np.binary_repr(out_state,n_island) 
                        if out_state_str[-1-island_qubit_q] == out_state_str[-1-island_qubit_qpp]:
                            ZZ += Rinv_all[island_q][out_state,in_state[island_q]]
                        else:
                            ZZ -= Rinv_all[island_q][out_state,in_state[island_q]]
                else:
                    n_island_q = len(islands[island_q])
                    n_island_qpp = len(islands[island_qpp])
                    for out_state_island_q in range(2**n_island_q):
                        out_state_q = np.binary_repr(out_state_island_q,n_island_q)[-1-island_qubit_q] # could do this more efficiently
                        for out_state_island_qpp in range(2**n_island_qpp):
                            out_state_qpp = np.binary_repr(out_state_island_qpp,n_island_qpp)[-1-island_qubit_qpp]
                            if out_state_q == out_state_qpp:
                                ZZ += Rinv_all[island_q][out_state_island_q,in_state[island_q]] * Rinv_all[island_qpp][out_state_island_qpp,in_state[island_qpp]]
                            else:
                                ZZ -= Rinv_all[island_q][out_state_island_q,in_state[island_q]] * Rinv_all[island_qpp][out_state_island_qpp,in_state[island_qpp]]
            return ZZ
            
        E = 0
        print('applying inverse response matrix')
        for outcome_str in tqdm(counts[0]):
            E -= (J*ZZ_mitigated(outcome_str) + hz*Z_mitigated(outcome_str)) * counts[0][outcome_str]/shots
        for outcome_str in tqdm(counts[1]):
            E -= hx*Z_mitigated(outcome_str) * counts[1][outcome_str]/shots
            
        return E
    
    def mitigate_ising_uncorrelated(self,job,hx=None,hz=None,J=None):
        tags = job.tags()
        if hx == None:
            hx = read_from_tags('hx',tags)
        if hz == None:
            hz = read_from_tags('hz',tags)
        if J == None:
            J = read_from_tags('J',tags)
        counts = job.result().get_counts()
        if len(counts) > 2:
            counts = [counts[-3], counts[-1]]
        shots = sum(counts[0].values())
        
        
        def Z_mitigated(q,counts):
            e0 = self.error_rate(q,0)[0]
            e1 = self.error_rate(q,1)[0]
            p1 = sum( counts[outcome_str]/shots for outcome_str in counts if outcome_str[-1-q] == '1')
            Z = 1 -2*p1
            Z = (Z + e0 - e1)/(1-e0-e1)
            return Z
            
        def ZZ_mitigated(qa,qb,counts):
            e0a = self.error_rate(qa,0)[0]
            e1a = self.error_rate(qa,1)[0]
            e0b = self.error_rate(qb,0)[0]
            e1b = self.error_rate(qb,1)[0]
            R = np.kron( np.array([[1-e0b, e1b],[e0b, 1-e1b]]), np.array([[1-e0a, e1a],[e0a, 1-e1a]]) )
            v = np.zeros(4)
            for outcome_str in counts:
                outcome = int(outcome_str[-1-qb] + outcome_str[-1-qa],2)
                v[outcome] += counts[outcome_str]/shots
            from numpy.linalg import inv
            v_mitigated = inv(R) @ v
            p1 = v_mitigated[1] + v_mitigated[2]
            ZZ = 1 - 2*p1
            return ZZ
            
            
        E = 0
        for q in range(self.n):
            E -= hz*Z_mitigated(q,counts[0]) + J*ZZ_mitigated(q,(q+1)%self.n,counts[0]) + hx*Z_mitigated(q,counts[1])
        
        return E
            
            
            
        
    
    def test_approximation(self,plot=False):
        empty_circuits = not (len(self.ideal_strs) == len(self.counts))
        print('empty_circuits = '+str(empty_circuits))
        total_error_rate = np.empty(len(self.ideal_strs))
        d_total_error_rate = np.empty(len(self.ideal_strs))
        approx_total_error_rate = np.empty(len(self.ideal_strs))
        d_approx_total_error_rate = np.empty(len(self.ideal_strs))
        uncorrelated_error_rate = np.empty(len(self.ideal_strs))
        d_uncorrelated_error_rate = np.empty(len(self.ideal_strs))
        for i in range(len(self.ideal_strs)):
            if empty_circuits:
                counts = self.counts[2*i+1]
            else:
                counts = self.counts[i]
            prepared_state = self.ideal_strs[i]
            prepared_str = ''.join([str(int(_)) for _ in np.flip(prepared_state)])
            nflip = self.shots - counts.get(prepared_str,0)
            p = nflip/self.shots
            dp = np.sqrt(p*(1-p)/self.shots)
            total_error_rate[i] = p
            d_total_error_rate[i] = dp
            
            p, dp = self.response_matrix_element(prepared_str,prepared_str)
            approx_total_error_rate[i] = 1-p
            d_approx_total_error_rate[i] = dp
            
            p, dp = self.response_matrix_element(prepared_str,prepared_str,uncorrelated=True)
            uncorrelated_error_rate[i] = 1-p
            d_uncorrelated_error_rate[i] = dp
        
        if plot:
            import matplotlib.pyplot as plt
            plt.errorbar(range(len(self.ideal_strs)),total_error_rate,d_total_error_rate,label='actual error rates',capsize=4)
            plt.errorbar(range(len(self.ideal_strs)),approx_total_error_rate,d_approx_total_error_rate,label='correlated model',capsize=4)
            plt.errorbar(range(len(self.ideal_strs)),uncorrelated_error_rate,d_uncorrelated_error_rate,label='uncorrelated model',capsize=4)
            plt.xlabel('prepared state')
            plt.ylabel('overall error rate')
            plt.legend(loc='best')
            
            
            
            correlated_residuals = (approx_total_error_rate - total_error_rate)/np.sqrt(d_total_error_rate**2 + d_approx_total_error_rate**2)

            uncorrelated_residuals = (uncorrelated_error_rate - total_error_rate)/np.sqrt(d_total_error_rate**2 + d_uncorrelated_error_rate**2)
            
            plt.figure()
            ax = plt.gca()
            plt.hist(correlated_residuals)
            plt.xlabel('number of standard deviations')
            plt.ylabel('frequency out of '+str(len(approx_total_error_rate)))
            plt.title('residuals from correlated error model')
            

            plt.text(0.99,0.9,'mean = %1.2f $\pm$ %1.2f \n var = %1.2f $\pm$ %1.2f' % (np.mean(correlated_residuals), 1/np.sqrt(len(approx_total_error_rate)),np.var(correlated_residuals,ddof=1), 2/(len(approx_total_error_rate)-1)),ha='right',transform = ax.transAxes)
            
            
            plt.figure()
            ax = plt.gca()
            plt.hist(uncorrelated_residuals)
            plt.xlabel('number of standard deviations')
            plt.ylabel('frequency out of '+str(len(approx_total_error_rate)))
            plt.title('residuals from uncorrelated error model')
            plt.text(0.99,0.9,'mean = %1.2f $\pm$ %1.2f \n var = %1.2f $\pm$ %1.2f' % (np.mean(uncorrelated_residuals), 1/np.sqrt(len(approx_total_error_rate)),np.var(uncorrelated_residuals,ddof=1), 2/(len(approx_total_error_rate)-1)),ha='right',transform = ax.transAxes)
            
            
            plt.show()
        
        return total_error_rate, d_total_error_rate, approx_total_error_rate, d_approx_total_error_rate, uncorrelated_error_rate, d_uncorrelated_error_rate
    
    def measure_paulis(self,qubits_in_paulis,counts,R_inv_all=[],coeffs=[1]):
        # Counts contains the results of measurements in an eigenbasis of the pauli operator. qubits_in_pauli contains the qubits on which the Pauli operator acts
        # optionally, pass the inverse response matrix (useful if this is called from another function that measures a Hamiltonian)
        # error analysis treats island probabilities as independent, even though they aren't really. Also ignores uncertainty in R_inv
        from numpy.linalg import inv
        if not hasattr(qubits_in_paulis[0],'__iter__'):
            qubits_in_paulis = [qubits_in_paulis]
        islands = self.get_islands()
        which_islands = []
        for qubits_in_pauli in qubits_in_paulis:
            which_islands += [[q in island for island in islands].index(True) for q in qubits_in_pauli]
        # which_islands is a list of which island each relevant qubit is in
        # qubit_number_in_island = [ islands[which_islands[q]].index(qubits_in_pauli[q]) for q in range(len(qubits_in_pauli))]
        which_islands_unique = list(set(which_islands))
        
        if len(R_inv_all) == 0:
            print('constructing inverse response matrix')
            R_all = [ self.island_response_matrix(i,output_dR = False) for i in which_islands_unique]
            R_inv_all = [ inv(R) for R in R_all]
        else:
            R_inv_all = [R_inv_all[i] for i in which_islands_unique]
        
        islands = [ islands[i] for i in which_islands_unique] # keep only islands that are used
        which_islands = range(len(which_islands_unique))
        
        #print('islands = '+str(islands))  #   looks correct...
        #print('shape(Rinv) = '+str([np.shape(Rinv) for Rinv in R_inv_all]))
        
        
        E = 0
        dE2 = 0
        shots = sum(counts.values())
        for outcome_str in counts:
            p_i = counts[outcome_str]/shots
            dp_i2 = p_i*(1-p_i)/shots
            S = 0
            for t in range(len(coeffs)):
                prod = 1
                for I in which_islands:
                    #print('I = '+str(I))
                    i_I = int( ''.join([ outcome_str[-1-q] for q in np.flip(islands[I]) ] )  ,2)
                    n_I = len(islands[I])
                    PR_I = 0
                    #print('I = '+str(I))
                    #print('island = '+str(islands[I]))
                    #print('shape(Rinv) = '+str(np.shape(R_inv_all[I])))
                    for j in range(2**n_I):
                        output_bitstr_I = np.binary_repr(j,n_I)
                        #print('output_bitstr = '+str(output_bitstr_I))
                        P_J = 1 - 2*(sum( 1 for q in range(n_I) if output_bitstr_I[-1-q] == '1' and islands[I][q] in qubits_in_paulis[t])%2)
                        #print('Z = '+str(P_J))
                        PR_I += P_J * R_inv_all[I][j,i_I]
                    prod = prod * PR_I
                S += coeffs[t] * prod
            E += p_i * S
            dE2 += dp_i2 * S**2
        
        return E, dE2
    
    def measure_H(self,counts,which_paulis_in_each,paulis,coeffs,uncorrelated=False,readout_mitigate=True):
        # uncertainty estimate is not exact; treats as independent things which are not actually independent.
        shots = sum(counts[0].values())
        from numpy.linalg import inv
        islands = self.get_islands()
        if readout_mitigate:
            print('constructing inverse response matrix')
            R_all = [ self.island_response_matrix(i,output_dR = False,uncorrelated=uncorrelated) for i in range(len(islands))]
            Rinv_all = [ inv(R) for R in R_all]
        else:
            Rinv_all = [np.eye(2**len(_)) for _ in islands]
        E = 0
        dE2 = 0
        for i in range(len(counts)):
            qubits_in_paulis = [ np.nonzero(paulis[t])[0] for t in which_paulis_in_each[i] ]
            coeffs_i = [coeffs[t] for t in which_paulis_in_each[i]]
            E_i, dE_i2 = self.measure_paulis(qubits_in_paulis, counts[i], Rinv_all, coeffs_i)
            E += E_i
            dE2 += dE_i2
        return E, np.sqrt(dE2)
    
    def ising_mitigate_v2(self,job,hx=None,hz=None,J=None):
        tags = job.tags()
        if hx == None:
            hx = read_from_tags('hx',tags)
        if hz == None:
            hz = read_from_tags('hz',tags)
        if J == None:
            J = read_from_tags('J',tags)
        counts = job.result().get_counts()
        if len(counts) > 2:
            counts = [counts[-3], counts[-1]]
        
        shots = sum(counts[0].values())
        from numpy.linalg import inv
        islands = self.get_islands()
        print('constructing inverse response matrix')
        R_all = [ self.island_response_matrix(i,output_dR = False) for i in range(len(islands))]
        Rinv_all = [ inv(R) for R in R_all]
        
        E = 0
        for q in range(self.n):
            E -= J*self.measure_pauli([q,(q+1)%self.n],counts[0],Rinv_all)
            E -= hz*self.measure_pauli([q],counts[0],Rinv_all)
            E -= hx*self.measure_pauli([q],counts[1],Rinv_all)

        return E
        
        
    
class readout_partition_qubit:
    def __init__(self,counts_or_job,qubit,given,ideal_strs='from_tags',run_partition=True,min_n_control=0,max_n_control=3,min_error_rates=1,condition_on_previous_state=False,starting_register=0):
        # for repeated measurements, starting_register = 0 gives the second measurement, starting_register = n gives the first measurement
        print('-------starting qubit '+str(qubit)+', given '+str(given)+'-------')
        if str(type(counts_or_job)) == "<class 'qiskit.providers.ibmq.job.ibmqjob.IBMQJob'>":
            self.counts = counts_or_job.result().get_counts()
            ideal_strs = bitstrs_from_tags(counts_or_job.tags())
        elif str(type(counts_or_job)) == "<class 'list'>":
            self.counts = counts_or_job
        if str(type(ideal_strs)) == "<class 'list'>":
            self.str_list = ideal_strs
            ideal_strs = str_list_to_array(ideal_strs)
        if len(self.counts) == 2*len(ideal_strs):
            self.counts = [self.counts[_] for _ in range(1,2*len(ideal_strs),2)]
        #if len(self.counts) != len(ideal_strs):
            #print('Note: lengths of counts and ideal_strs do not match. (This may be intended behavior.)')
        self.ideal_strs = ideal_strs
        self.qubit = qubit
        self.given = given
        self.shots = sum(self.counts[0].values())
        self.n = np.shape(ideal_strs)[1]
        self.max_n_control = max_n_control
        self.min_n_control = min_n_control
        self.min_error_rates = min(min_error_rates,2**min_n_control)
        if not condition_on_previous_state:
            self.indices = [[i for i in range(len(self.ideal_strs)) if self.ideal_strs[i,qubit] == given]]
        elif condition_on_previous_state:
            self.indices = [[i for i in range(len(self.ideal_strs)) if (self.ideal_strs[i,qubit] == given and self.ideal_strs[i-1,qubit] == 0)]]
        self.n_flip = [[ self.get_n_flip(i,starting_register) for i in self.indices[0]]]
        
        self.initial_n_flip = self.n_flip
        self.initial_indices = self.indices
        self.p_quick = 0
        
        self.separated = False
        self.n_control = 0
        self.n_error_rates = 1
        self.control_qubits = []
        self.multiple_options = False
        if run_partition:
            while not self.separated:
                self.partition()
    def get_n_flip(self,trial,starting_register=0):
        return sum(self.counts[trial][outcome_str] for outcome_str in self.counts[trial] if int(outcome_str[-1-self.qubit-starting_register]) != self.given)
    def p_value(self,p_crit = 0.05,use_MC=True):
        
        p_value = np.array([self.variance_test(n_flip) for n_flip in self.n_flip])
        which_partition = np.argmin(p_value)
        p_value = p_value[which_partition]
        
        # variance_test gives the probability of obtaining a variance larger than observed assuming a binomial distribution. Need to scale this to find the probability of finding any variance this large or larger among any of the n qubits * 2 givens * len(n_flip) partitions.
        p_value = 1 - (1-p_value)**(2*self.n*self.n_error_rates)
            
        # if failed variance test, try MC variance test (since the gaussian approximation underestimates the upper tail).
        if p_value < p_crit and use_MC:
            p_value = self.variance_test_MC(self.n_flip[which_partition])
            p_value = 1 - (1-p_value)**(2*self.n*len(self.n_flip)) # scale MC p-value too
            
        return p_value  
    def variance_test(self,n_flip):
        if len(n_flip) <= 1:
            return 1
        else:
            p_meas = np.mean(n_flip)/self.shots
            observed_variance = np.var(n_flip,ddof=1)
            expected_variance = p_meas*self.shots*(1-p_meas)
            d_var_exp = np.sqrt( expected_variance_of_variance(self.shots,p_meas,len(n_flip) ))
            p = norm.sf(observed_variance,expected_variance,d_var_exp)
            return p
    def variance_test_MC(self,n_flip,p_crit=0.05):
        
        p_scaled_crit = 1 - (1-p_crit)**(1/(2*self.n*self.n_error_rates))
        trials = int(1000/p_scaled_crit)
        
        if len(n_flip) <= 1:
            observed_variance = 0
            var_MC = 0
            p = 1
        else:
            p_meas = np.mean(n_flip)/self.shots
            observed_variance = np.var(n_flip,ddof=1)
            n_flip_MC = rng.binomial(self.shots,p_meas,size=(trials,len(n_flip)))
            var_MC = np.var(n_flip_MC,1,ddof=1)
            p = np.sum(var_MC >= observed_variance)/trials
        if p == 0:
            #print('Trials not large enough for accurate p-value. Returning 1/trials.')
            p = 1/trials
        return p
    def partition(self,p_crit = 0.05):
        
        if self.n_control < self.min_n_control or self.n_error_rates < self.min_error_rates:
            # initialize:
            self.n_control = self.min_n_control
            self.n_error_rates = self.min_error_rates
        else:
            #check and iterate
            p = self.p_value(p_crit)
            self.separated = p > p_crit
            if self.separated:
                return
            elif self.n_error_rates < 2**self.n_control:
                self.n_error_rates += 1
            elif self.n_control < self.max_n_control:
                self.n_control += 1
                self.n_error_rates = max(self.min_error_rates,2)
            else:
                self.separated = 2
                return
        
        print('starting n_control = '+str(self.n_control)+', n_error_rates = '+str(self.n_error_rates))
        
        from itertools import combinations, product
        self.p_quick = 0
        self.possible_control_qubits = []
        self.possible_keys = []
        self.possible_p_values = []
        # iterate through control qubits:
        for control_qubits in combinations(range(self.n),self.n_control):
            # next iterate through all possible keys. To avoid redundancies, fix keys[0] = 0, the first nonzero element of keys to be 1, etc.
            for key in product(range(self.n_error_rates),repeat=2**self.n_control):
                # filter out redundant keys:
                valid_key = True
                if key[0] != 0:
                    valid_key = False
                    continue
                i_previous = 0
                for g in range(1,self.n_error_rates):
                    if g not in key:
                        valid_key = False
                        break
                    i = key.index(g)
                    if i < i_previous:
                        valid_key = False
                        break
                    i_previous = i
                # end filtering
                if not valid_key:
                    continue
                
                # partition n_flip based on key
                self.n_flip = [ [] for _ in range(self.n_error_rates) ] # set to parent object so that pvalue() can use it
                self.indices = [ [] for _ in range(self.n_error_rates) ]
                for i in range(len(self.initial_n_flip[0])):
                    bitstr = self.ideal_strs[self.initial_indices[0][i]]
                    control_state = bitstr[np.array(control_qubits)]
                    control_state_str = ''.join( [ str(int(_)) for _ in np.flip(control_state) ] )
                    control_state_int = int(control_state_str,2)
                    group = key[control_state_int]
                    self.n_flip[group].append(self.initial_n_flip[0][i])
                    self.indices[group].append(self.initial_indices[0][i])

                # look for the one with the highest p-value
                p_quick = self.p_value(use_MC=False)
                if p_quick > p_crit:
                    self.possible_control_qubits.append(control_qubits)
                    self.possible_keys.append(key)
                    self.possible_p_values.append(p_quick)
                if p_quick > self.p_quick:
                    self.p_quick = p_quick
                    best_n_flip = self.n_flip
                    best_indices = self.indices
                    best_key = key
                    best_control_qubits = control_qubits
        if self.p_quick > 0:
            self.n_flip = best_n_flip
            self.indices = best_indices
            self.key = best_key
            self.control_qubits = best_control_qubits
        if len(set(self.possible_control_qubits)) > 1:
            print('Warning: multiple choices of control qubits were identified: '+str(set(self.possible_control_qubits)))
            self.multiple_options = True
        elif len(self.possible_keys) > 1:
            print('Warning: multiple possible keys were identified.')
        return
    def plot(self,p_vec=[],trials='all',condition_on_previous_state=False,starting_register=0):
        if trials == 'all':
            n_flip = self.n_flip
            trials = np.array([_ for _ in range(len(self.n_flip)) if len(self.n_flip[_]) > 0])
        else:
            if not hasattr(trials,'__iter__'):
                trials = [trials]
            n_flip = np.array(self.n_flip,dtype=object)[trials]
        
        plt.clf()
        bins = np.arange( min(min(self.n_flip[_]) for _ in trials), max(max(self.n_flip[_]) for _ in trials) + 2)
        if condition_on_previous_state:
            label = ['Previous state = 0','Previous state = 1']
        else:
            label = ['Actual counts, group '+str(g) for g in range(self.n_error_rates)]
        if condition_on_previous_state:
            previous_0_indices = [i for i in range(len(self.ideal_strs)) if (self.ideal_strs[i,self.qubit] == self.given and self.ideal_strs[i-1,self.qubit] == 0)]
            previous_1_indices = [i for i in range(len(self.ideal_strs)) if (self.ideal_strs[i,self.qubit] == self.given and self.ideal_strs[i-1,self.qubit] == 1)]
            n_flip_0 = [ self.get_n_flip(i,starting_register) for i in previous_0_indices]
            n_flip_1 = [ self.get_n_flip(i,starting_register) for i in previous_1_indices]
            h0 = plt.hist(n_flip_0, bins=bins-.5,  histtype='step', label=label[0],stacked=True,linestyle='solid')
            h1 = plt.hist(n_flip_1, bins=bins-.5,  histtype='step', label=label[1],stacked=True,linestyle='--')
        elif self.n_error_rates == 2:
            h0 = plt.hist(n_flip[0], bins=bins-.5,  histtype='step', label=label[0],stacked=True,linestyle='solid')
            h1 = plt.hist(n_flip[1], bins=bins-.5,  histtype='step', label=label[1],stacked=True,linestyle='--')
        else:
            h = plt.hist(n_flip, bins=bins-.5,  histtype='step', label=label,stacked=True)
        
        if not condition_on_previous_state:
            expected_counts = [ binom.pmf(bins[:-1],self.shots, np.mean(self.n_flip[i])/self.shots)*len(self.indices[i]) for i in trials]
            
            style = ['-','--']
            for i in range(len(trials)-1,-1,-1):
                #p_meas = np.mean(self.n_flip[i])/self.shots
                #expected_counts = [binom.pmf(bin,self.shots,p_meas)*len(self.indices[i]) for bin in bins]
                label = ['Expected counts, group '+str(g) for g in range(self.n_error_rates)]
                plt.plot(bins[:-1],expected_counts[i],style[i%2],label=label[i])
                
            plt.plot(bins[:-1],np.sum(expected_counts,0),':',label= 'Expected counts')
            
            if len(p_vec) > 0:
                counts_binosum = sum( binom.pmf(bins[:-1], self.shots, p_i)* sum(len(ni) for ni in n_flip)/len(p_vec) for p_i in p_vec)
                plt.plot(bins[:-1],counts_binosum,label='sum of binomials with p = '+str(p_vec))
            
        plt.xlabel('# of times qubit '+str(self.qubit)+' flips from '+str(self.given)+' to '+str(1-self.given)+' out of '+str(int(self.shots))+' shots',fontsize=12)
        plt.ylabel('frequency out of '+str( sum(len(self.indices[t]) for t in trials ))+' ideal states of the other qubits',fontsize=12)
        plt.legend(loc='best',prop={'size':12})
        if len(self.control_qubits) > 0:
            plt.title('control qubits: '+str(self.control_qubits)+'\nkey: '+str(self.key))
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.show()
    
