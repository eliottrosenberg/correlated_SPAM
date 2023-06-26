from qiskit import QuantumCircuit, execute
import numpy as np
rng = np.random.default_rng()
#vol_pi = 0.5
#vol_sphere = 4/3*np.pi
vol = 1/3*np.pi # northern hemisphere of sphere

def SP_circuits(qubits,insert_empty_circuit=True,theta=[],phi=[],option=0):
    n = len(qubits)
    if len(theta) == 0:
        theta = [0 for _ in range(n)]
    if len(phi) == 0:
        phi = [0 for _ in range(n)]
    if len(theta) != n or len(phi) != n:
        print('warning: theta or phi is not the expected length')
    
    id = initialize(theta,phi,option)
    id.measure(qubits,qubits)
    
    I = QuantumCircuit(n,n)
    I.measure(qubits,qubits)
    
    x = initialize(theta,phi,option)
    x.x(qubits)
    x.measure(qubits,qubits)
    
    xm90 = initialize(theta,phi,option)
    xm90.rx(-np.pi/2,qubits)
    xm90.measure(qubits,qubits)
    
    x90 = initialize(theta,phi,option)
    x90.rx(np.pi/2,qubits)
    x90.measure(qubits,qubits)
    
    y90 = initialize(theta,phi,option)
    y90.ry(np.pi/2,qubits)
    y90.measure(qubits,qubits)
    
    ym90 = initialize(theta,phi,option)
    ym90.ry(-np.pi/2,qubits)
    ym90.measure(qubits,qubits)
    
    if insert_empty_circuit == 'both':
        return [id,x,xm90,x90,y90,ym90, I,id, x, I,xm90, I,x90, I,y90, I,ym90]
    elif not insert_empty_circuit:
        return [id,x,xm90,x90,y90,ym90]
    elif insert_empty_circuit:
        return [I,id, x, I,xm90, I,x90, I,y90, I,ym90]


def SP_circuits_submit(backend,shots,insert_empty_circuit=True):
    n = backend.configuration().n_qubits
    qc_all = SP_circuits(range(n),insert_empty_circuit)
    job = execute(qc_all,backend,initial_layout=range(n),shots=shots,rep_delay = backend.configuration().rep_delay_range[1],job_tags=['SPAM characterization','insert_empty_circuit = '+str(insert_empty_circuit)],optimization_level=3,use_measure_esp=False)
    return job


def SPAM_parameters(counts_or_job,mc_trials = int(1e6),r_radius=[0.002,0.002,0.002,0.01,np.pi/10],r_to_1=True):
    if str(type(counts_or_job)) == "<class 'qiskit.providers.ibmq.job.ibmqjob.IBMQJob'>":
        counts = counts_or_job.result().get_counts()
    else:
        counts = counts_or_job
    shots = sum( counts[0].values() )
    n = len(list(counts[0].keys())[0])
    
    if len(counts) == 17:
        insert_empty_circuit = 'both'
        successes_0 = [ np.array( [ get_successes(c,q) for c in counts[:6] ]) for q in range(n) ]
        successes_1 = [ np.array( [ get_successes(counts[i],q) for i in [7,8,10,12,14,16] ]) for q in range(n) ]
        successes = successes_0 + successes_1
    elif len(counts) == 6:
        insert_empty_circuit = False
        successes = [ np.array( [ get_successes(c,q) for c in counts[:6] ]) for q in range(n) ]
    elif len(counts) == 11 or len(counts) == 15:
        insert_empty_circuit = True
        successes = [ np.array( [ get_successes(counts[i],q) for i in [1,2,4,6,8,10] ]) for q in range(n) ]
    # elif len(counts) == 22:
        # insert_empty_circuit = True
        # successes_0 = [ np.array( [ get_successes(counts[i],q) for i in [1,2,4,6,8,10] ]) for q in range(n) ]
        # successes_1 = [ np.array( [ get_successes(counts[i],q) for i in np.array([1,2,4,6,8,10] ) + 11]) for q in range(n) ]
        #successes = successes_0 + successes_1
    elif len(counts)%11 == 0:
        nrep = len(counts)//11
        successes = []
        for k in range(nrep):
            successes += [ np.array( [ get_successes(counts[i],q) for i in np.array([1,2,4,6,8,10] ) + 11*k]) for q in range(n) ]
    
    p_estimates = []
    dp2 = []
    r_estimates = []
    dr2 = []
    ML_estimates = []
    
    for s in successes:
        
        est_ML = parameter_estimate_ML_combined(s,shots,r_to_1=r_to_1)
        ML_estimates.append(est_ML)
        r_center = est_ML.x
        max_L = -1 * est_ML.fun
        
        p_est, L, norm, p, r, r_est = parameter_estimate(s,shots,mc_trials,max_L,r_center,r_radius)
        p_estimates.append(p_est)
        r_estimates.append(r_est)
        
        var_p, var_r = parameter_variance(s,shots,mc_trials,max_L,p_est, L, norm, p,r_center,r_radius,r,r_est)
        dp2.append(var_p)
        dr2.append(var_r)
        
        
    return p_estimates, dp2, r_estimates, dr2, ML_estimates

def get_successes(counts,qubit):
    s = 0
    for bitstr in counts:
        if bitstr[-1 - qubit] == '0':
            s += counts[bitstr]
    return s

def likelihood_of_observed(p,successes,shots,max_L):
    #p[0] = \pi_0, p[1] = \pi_z, p[2] = x_0, p[3] = y_0, p[4] = z_0
    # shape(p) = (5,ntrials)
    print('p = '+str(p[:,0]))
    sh = np.shape(p)
    if len(sh) > 1:
        ntrials = sh[1]
        f = np.empty((6,ntrials))
    else:
        f = np.empty(6)
    f[0] = p[0] + p[1]*p[4]
    f[1] = p[0] - p[1]*p[4]
    
    f[2] = p[0] - p[1]*p[3]
    f[3] = p[0] + p[1]*p[3]
    
    f[4] = p[0] - p[1]*p[2]
    f[5] = p[0] + p[1]*p[2]
    
    L = successes@np.log(f) +(shots - successes) @ np.log(1-f)
    print('before subtraction, L = '+str(L))
    print('max_L = '+str(max_L))
    L -= max_L
    
    print('L = '+str(L))
    print('max MC L = '+str(max(L)))
    
    likelihood = np.exp(L)
    
    print('likelihood = '+str(likelihood))
    # import matplotlib.pyplot as plt
    # plt.hist(L)
    # plt.show()
    
    print('----')
    
    # if bias_towards_NP:
        # sintheta = np.sqrt( 1 - p[-1]**2 )
        # likelihood = likelihood * sintheta
    
    norm = vol * np.mean(likelihood)
    
    return likelihood, norm


def cost_fn(r,successes,shots,jac):
    # returns the log likelihood and its gradient
    # r[0] = pi0 + piz, r[1] = pi0 - pi[z], r[2] = r, r[3] = theta, r[4] = phi
    
    p = [ (r[0]+r[1])/2, (r[0] - r[1])/2, r[2]*np.sin(r[3])*np.cos(r[4]), r[2]*np.sin(r[3])*np.sin(r[4]), r[2]*np.cos(r[3]) ]
    
    # grad_p[i][j] = dp[i]/dr[j]
    if jac:
        grad_p = np.array( [ [1/2,1/2,0,0,0], [1/2,-1/2,0,0,0], [0,0,np.sin(r[3])*np.cos(r[4]),r[2]*np.cos(r[3])*np.cos(r[4]),-r[2]*np.sin(r[3])*np.sin(r[4])], [0,0,np.sin(r[3])*np.sin(r[4]), r[2]*np.cos(r[3])*np.sin(r[4]),r[2]*np.sin(r[3])*np.cos(r[4])], [0,0,np.cos(r[3]),-r[2]*np.sin(r[3]),0] ] )
    
    #p[0] = \pi_0, p[1] = \pi_z, p[2] = x_0, p[3] = y_0, p[4] = z_0
    f = np.empty(6)
    f[0] = p[0] + p[1]*p[4]
    f[1] = p[0] - p[1]*p[4]
    
    f[2] = p[0] - p[1]*p[3]
    f[3] = p[0] + p[1]*p[3]
    
    f[4] = p[0] - p[1]*p[2]
    f[5] = p[0] + p[1]*p[2]
    
    grad_f = np.array( [ [1, p[4], 0, 0, p[1]], [1, -p[4], 0, 0, -p[1]], [1, -p[3], 0, -p[1], 0], [1, p[3], 0, p[1], 0], [1, -p[2], -p[1], 0, 0], [1, p[2], p[1], 0, 0] ] )
    
    L = successes@np.log(f) +(shots - successes) @ np.log(1-f)
    if np.isnan(L) or np.isinf(L):
        if jac:
            r0 = [1,0,1,0,0]
            return (np.inf, r - r0)
        else:
            return np.inf
    
    if jac:
        grad_L = ( (successes/f) - (shots-successes)/(1-f) ) @ grad_f @ grad_p
        return (-L, -grad_L)
    else:
        return -L
        
def check_gradient(r0,successes,shots):
	def func(r):
	    return cost_fn(r,successes,shots,jac=False)
	def grad(r):
		(a,b) = cost_fn(r,successes,shots,jac=True)
		return b
	from scipy.optimize import check_grad
	return check_grad(func,grad,r0)


def rand_insphere(mc_trials,r_center=[0.5,np.pi/4,np.pi],r_radius=[0.5,np.pi/4,np.pi]):
    
    r3_min = max( (r_center[0] - r_radius[0])**3, 0)
    r3_max = min( (r_center[0] + r_radius[0])**3, 1)
    
    costheta_min = max( np.cos(r_center[1] + r_radius[1]), 0) # force northern hemisphere
    costheta_max = min(np.cos(r_center[1] - r_radius[1]), 1)
    
    phi_min = r_center[2] - r_radius[2]
    phi_max = r_center[2] + r_radius[2]
    
    phi = rng.random(mc_trials) * (phi_max - phi_min) + phi_min
    costheta = rng.random(mc_trials)* (costheta_max - costheta_min) + costheta_min
    th = np.arccos(costheta)
    r3 = rng.random(mc_trials) * (r3_max - r3_min) + r3_min
    r = np.cbrt(r3)
    
    x = r*np.sin(th)*np.cos(phi)
    y = r*np.sin(th)*np.sin(phi)
    z = r*np.cos(th)
    
    return x,y,z, r, th, phi
    

def rand_Pi(mc_trials,r_center=[0.5,0.5],r_radius=[0.5,0.5]):
    
    pi_plus_min = max(r_center[0] - r_radius[0], 0)
    pi_plus_max = min(r_center[0] + r_radius[0], 1)
    pi_minus_min = max(r_center[1] - r_radius[1], 0)
    pi_minus_max = min(r_center[1] + r_radius[1], 1)
    
    
    pi_plus = rng.random(mc_trials) * (pi_plus_max - pi_plus_min) + pi_plus_min
    pi_minus = rng.random(mc_trials) * (pi_minus_max - pi_minus_min) + pi_minus_min
    
    
    pi0 = (pi_plus + pi_minus)/2
    piz = (pi_plus - pi_minus)/2
    
    
    return pi0, piz, pi_plus, pi_minus
    
def rand_params(mc_trials,r_center = [0.5,0.5,0.5,np.pi/4,np.pi],r_radius = [0.5,0.5,0.5,np.pi/4,np.pi]):
    x,y,z, r, th, phi = rand_insphere(mc_trials,r_center[2:],r_radius[2:])
    pi0, piz, pi_plus, pi_minus = rand_Pi(mc_trials,r_center[:2],r_radius[:2])
    p = np.concatenate(( (pi0,),(piz,),(x,),(y,),(z,) ))
    r = np.concatenate(( (pi_plus,),(pi_minus,),(r,),(th,),(phi,) ))
    return p, r
    
#def normalization(successes,shots,mc_trials):
#    # sample over domain of parameters
#    
#    p = rand_params(mc_trials)
#    return vol * np.mean( likelihood_of_observed(p, successes, shots) )
    

# def prob_of_parameters(p,successes,shots,mc_trials):
    # return likelihood_of_observed(p, successes, shots)/ normalization(successes,shots,mc_trials)


def parameter_estimate(successes,shots,mc_trials,max_L,r_center = [0.5,0.5,0.5,np.pi/4,np.pi],r_radius = [0.5,0.5,0.5,np.pi/4,np.pi]):
    p, r = rand_params(mc_trials,r_center,r_radius)
    L , norm = likelihood_of_observed(p, successes, shots,max_L)
    p_est = vol *  p @ L / (norm * mc_trials)
    r_est = vol *  r @ L / (norm * mc_trials)
    return p_est, L, norm, p, r, r_est
    

def parameter_estimate_ML(successes,shots,method='Powell',jac=False,r0='NP',tol=None,r_to_1=False):
    from scipy.optimize import minimize
    if type(r0) == type('') and r0 == 'NP':
        epsilon = 1e-4
        r0 = [1-epsilon,0+epsilon,1-epsilon,0+epsilon,0]
    elif type(r0) == type('') and r0 == 'rand':
        phi = rng.random()*np.pi*2
        costheta = rng.random() # start in the northern hemisphere
        th = np.arccos(costheta)
        r3 = rng.random()
        r = np.cbrt(r3)
        pplus = rng.random()
        pminus = rng.random()
        r0 = [pplus, pminus, r, th, phi]
    
    # r[0] = pi0 + piz, r[1] = pi0 - piz, r[2] = r, r[3] = theta, r[4] = phi
    
    if r_to_1:
        r0[2] = 1
    return minimize(cost_fn,r0,args=(successes,shots,jac),method=method,jac=jac,bounds=[(0,1),(0,1),(int(r_to_1),1),(0,np.pi/2),(-np.pi,np.pi)],tol=tol)
    
def parameter_estimate_ML_combined(successes,shots,num_rand=10,second_method='TNC',r_to_1=False):
    sol_NP = parameter_estimate_ML(successes,shots,method='Powell',jac=False,r0='NP',r_to_1=r_to_1)
    sols_rand = [parameter_estimate_ML(successes,shots,method='Powell',jac=False,r0='rand',r_to_1=r_to_1) for _ in range(num_rand)]
    sols = [sol_NP] + sols_rand
    cf = [s.fun for s in sols]
    indx = np.argmin(cf)
    sol = sols[indx]
    
    #print('initial min: '+str(sol.fun))
    
    sol2 = parameter_estimate_ML(successes,shots,second_method,True,sol.x,tol=0.0001/sol.fun,r_to_1=r_to_1)
    
    #print('second min: '+str(sol2.fun))
    
    return sol2
    

def parameter_variance(successes,shots,mc_trials,max_L,p_est = [], L = [], norm = None, p = [],r_center = [0.5,0.5,0.5,np.pi/4,np.pi],r_radius = [0.5,0.5,0.5,np.pi/4,np.pi],r=[],r_est=[]):
    if len(p_est) == 0 or len(L) == 0 or norm == None or len(p) == 0:
        p_est, L, norm, p, r, r_est = parameter_estimate(successes,shots,mc_trials,max_L,r_center,r_radius)
    var_p = vol *  (p - np.outer(p_est, np.ones(mc_trials) )  )**2 @ L / (norm * mc_trials)
    var_r = vol *  (r - np.outer(r_est, np.ones(mc_trials) )  )**2 @ L / (norm * mc_trials)
    return var_p, var_r

def plot(r,dr2,n,ml_r=[],labels=[]):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5,1,sharex=True)
    plt.xlabel('qubit number')
    def plot_i(r,dr2,label=None):
        axs[0].errorbar(range(n),[1-ri[0] for ri in r], [np.sqrt(dr2i[0]) for dr2i in dr2],label=label,capsize=5,linestyle=None)
        if len(ml_r) > 0:
            axs[0].plot(range(n),[1-ml_ri[0] for ml_ri in ml_r],'.',label='ML')
        for i in range(1,5):
            axs[i].errorbar(range(n),[ri[i] for ri in r], [np.sqrt(dr2i[i]) for dr2i in dr2],label=label,capsize=5,fmt='')
            if len(ml_r) > 1:
                axs[i].plot(range(n),[ri[i] for ri in ml_r],'.',label='ML')
        axs[0].set(ylabel='$\epsilon_0$')
        axs[1].set(ylabel='$\epsilon_1$')
        axs[2].set(ylabel='r')
        axs[3].set(ylabel='$θ$')
        axs[4].set(ylabel='$\phi$')
    
    
    
    if len(r) == n:
        plot_i(r,dr2)
    else:
        nrep = len(r)//n
        for k in range(nrep):
            plot_i(r[(k*nrep):((k+1)*nrep)],dr2[(k*nrep):((k+1)*nrep)],labels[k])
            
    plt.show()   
    

def plot_theta(r,dr2,n,show=False):
    trials = len(r)//n
    theta = np.zeros((trials,n))
    dtheta = np.zeros((trials,n))
    for trial in range(trials):
        for q in range(n):
            theta[trial,q] = r[trial*n+q][3]
            dtheta[trial,q] = np.sqrt(dr2[trial*n+q][3])
    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        for trial in range(trials):
            if trial == 0:
                label = 'no prerotation'
            else:
                label = 'prerotation option '+str(trial)
            plt.errorbar(range(n),theta[trial,:],dtheta[trial,:],capsize=5,label=label)
        plt.xlabel('qubit number')
        plt.ylabel(r'$\theta$')
        plt.legend(loc='best')
        plt.show()
    return theta, dtheta

def check_previous_state_dependence(job,circuit_1=2,circuit_2=10,label_1='after X',label_2='after I'):
    counts = job.result().get_counts()
    n = len(list(counts[0].keys())[0])
    
    def bias(counts,qubit):
        shots = sum(counts.values())
        nOne = 0
        for bitstr in counts:
            if bitstr[-1-qubit] == '1':
                nOne += counts[bitstr]
        fOne = nOne/shots
        dfOne = np.sqrt( fOne* (1-fOne)/shots )
        Z = 1 - 2*fOne
        dZ = 2*dfOne
        return Z, dZ
    
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    Z_after_X = []
    dZ_after_X = []

    Z_after_I = []
    dZ_after_I = []
    for q in range(7):
        Z_after_X_i, dZ_after_X_i = bias(counts[circuit_1],q)
        Z_after_I_i, dZ_after_I_i = bias(counts[circuit_2],q)
        Z_after_X.append(Z_after_X_i)
        dZ_after_X.append(dZ_after_X_i)
        Z_after_I.append(Z_after_I_i)
        dZ_after_I.append(dZ_after_I_i)
    plt.errorbar(range(7),Z_after_X,dZ_after_X,label=label_1,capsize=5,linestyle=None)
    plt.errorbar(range(7),Z_after_I,dZ_after_I,label=label_2,capsize=5,linestyle=None)
    plt.xlabel('qubit number')
    plt.ylabel('<Z>')
    plt.legend(loc='best')
    plt.show()


def initialize(theta,phi,option=0):
    # initialize a QuantumCircuit with prerotations applied
    # option sets the axis to rotate to first in the xy plane. 0 = +x axis, 1 = +y, 2 = -x, 3 = -y
    n = len(theta)
    qc = QuantumCircuit(n,n)
    if max(theta) == 0:
        return qc
    else:
        for q in range(n):
            if option == 0:
                qc.rz(-phi[q],q)
                qc.ry(-theta[q],q)
            elif option == 1:
                qc.rz(np.pi/2 - phi[q],q)
                qc.rx(theta[q],q)
            elif option == 2:
                qc.rz(np.pi - phi[q],q)
                qc.ry(theta[q],q)
            elif option == 3:
                qc.rz(3*np.pi/2 - phi[q],q)
                qc.rx(-theta[q],q)
    return qc

def validate_circuits(theta,phi):
    n = len(theta)
    qubits = range(n)
    qc_all = SP_circuits(qubits) + SP_circuits(qubits,theta=theta,phi=phi,option=0) + SP_circuits(qubits,theta=theta,phi=phi,option=1) + SP_circuits(qubits,theta=theta,phi=phi,option=2) + SP_circuits(qubits,theta=theta,phi=phi,option=3)
    
    return qc_all

def validate(backend,shots):
    job_calib = SP_circuits_submit(backend,shots)
    p_estimates, dp2, r_estimates, dr2, ML_estimates = SPAM_parameters(job_calib,mc_trials = 100) # use small mc_trials because we'll just use ML_estimates here
    theta = [m.x[3] for m in ML_estimates]
    phi = [m.x[4] for m in ML_estimates]
    n = len(ML_estimates)
    qc = validate_circuits(theta,phi)
    job_validate = execute(qc,backend,initial_layout=range(n),shots=shots,rep_delay = backend.configuration().rep_delay_range[1],job_tags=['SPAM validation','theta = '+str(theta),'phi = '+str(phi)],use_measure_esp=False)
    return job_validate

def error_rate(counts,qubit,prep=0,mitigate=False,e0=0,de0=0,e1=0,de1=0):
    prep = str(prep)
    nflip = 0
    shots = sum(counts.values())
    for bitstr in counts:
        if bitstr[-1-qubit] != prep:
            nflip += counts[bitstr]
            
    e = nflip/shots
    de = np.sqrt(e*(1-e)/shots)
    
    if hasattr(e0,'__iter__'):
        e0 = e0[qubit]
        e1 = e1[qubit]
        de0 = de0[qubit]
        de1 = de1[qubit]
    
    if mitigate:
        if prep == '1':
            eA = e1
            deA = de1
            eB = e0
            deB = de0
        elif prep == '0':
            eA = e0
            deA = de0
            eB = e1
            deB = de1
        
        phat = e
        dphat = de
        
        e = (phat - eA)/(1-eA-eB)
        
        de = np.sqrt( (dphat/(1-eA-eB))**2 + ( -1/(1-eA-eB) + (phat - eA)/(1-eA-eB)**2)**2 * deA**2 + ((phat - eA)/(1-eA-eB)**2)**2 * deB**2  )
        
    
    return e, de

def plot_validation_results(validation_job,basis='x',mitigate=False,ep0=0,dep0=0,ep1=0,dep1=0,r=1,dr = 0):
    
    if mitigate == False:
        r = 1
        dr = 0
    if basis == 'y':
        ind = 4
    elif basis == 'x':
        ind = 8
    
    counts = validation_job.result().get_counts()
    n = len(list(counts[0].keys())[0])

    
    num_options = len(counts)//11 - 1
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,1,sharex=True)
    plt.xlabel('qubit number')
    
    for option in range(num_options):
        e0 = []
        de0 = []
        e1 = []
        de1 = []
        e0_prerot = []
        de0_prerot = []
        e1_prerot = []
        de1_prerot = []
        for q in range(n):
            if hasattr(r,'__iter__'):
                rq = r[q]
                drq = dr[q]
            else:
                rq = r
                drq = dr
            e0_i, de0_i = error_rate(counts[ind],q,1,mitigate,ep0,dep0,ep1,dep1)
            e0_i = (1 - 2*e0_i)/rq
            de0_i = np.sqrt( (2*de0_i/rq)**2 + (e0_i*drq/rq)**2)
            e1_i, de1_i = error_rate(counts[ind+2],q,0,mitigate,ep0,dep0,ep1,dep1)
            e1_i = (1 - 2*e1_i)/rq
            de1_i = np.sqrt( (2*de1_i/rq)**2 + (e1_i*drq/rq)**2 )
            
            e0_prerot_i, de0_prerot_i = error_rate(counts[11*(option+1)+ind],q,1,mitigate,ep0,dep0,ep1,dep1)
            e1_prerot_i, de1_prerot_i = error_rate(counts[11*(option+1)+2+ind],q,0,mitigate,ep0,dep0,ep1,dep1)
            
            e0_prerot_i = (1 - 2*e0_prerot_i)/rq
            de0_prerot_i = np.sqrt( (2*de0_prerot_i/rq)**2 + (e0_prerot_i*drq/rq)**2 )
            
            e1_prerot_i = (1 - 2*e1_prerot_i)/rq
            de1_prerot_i = np.sqrt(   (2*de1_prerot_i/rq)**2 + (e1_prerot_i*drq/rq)**2 )
            
            
            
            e0.append(e0_i)
            de0.append(de0_i)
            e1.append(e1_i)
            de1.append(de1_i)
            
            e0_prerot.append(e0_prerot_i)
            de0_prerot.append(de0_prerot_i)
            e1_prerot.append(e1_prerot_i)
            de1_prerot.append(de1_prerot_i)
        
        
        
        if basis == 'x':
            fig.suptitle('<X>')
            axs[0].set(ylabel='-<Z> after Ry(pi/2)')
            axs[1].set(ylabel='<Z> after Ry(-pi/2)')
        elif basis == 'y':
            fig.suptitle('<Y>')
            axs[0].set(ylabel='-<Z> after Rx(-pi/2)')
            axs[1].set(ylabel='<Z> after Rx(pi/2)')
        
        if option == 0:
            axs[0].errorbar(range(n),e0,de0,label='without prerotation',capsize=5)
        axs[0].errorbar(range(n),e0_prerot,de0_prerot,label='with prerotation '+str(option),capsize=5)
        
        if option == 0:
            axs[1].errorbar(range(n),e1,de1,label='without prerotation',capsize=5)
        axs[1].errorbar(range(n),e1_prerot,de1_prerot,label='with prerotation '+str(option),capsize=5)
        
    axs[0].axhline(y=0, color='k')
    axs[1].axhline(y=0, color='k')
    plt.legend(loc='best')
    plt.show()

def overall_SPAM(job):
    counts = job.result().get_counts()
    n = len(list(counts[0].keys())[0])
    e0 = []
    de0 = []
    e1 = []
    de1 = []
    for q in range(n):
        e0i, de0i = error_rate(counts[1],q,prep=0)
        e1i, de1i = error_rate(counts[2],q,prep=1)
        e0.append(e0i)
        de0.append(de0i)
        e1.append(e1i)
        de1.append(de1i)
    
    return e0, de0, e1, de1

def plot_overall_SPAM(jobs,labels):
    counts = jobs[0].result().get_counts()
    n = len(list(counts[0].keys())[0])
    def plot_i(job,label):
        e0, de0, e1, de1 = overall_SPAM(job)
        axs[0].errorbar(range(n),e0,de0,label=label,capsize=5)
        axs[1].errorbar(range(n),e1,de1,label=label,capsize=5)
        
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,1,sharex=True)
    axs[0].set(ylabel='$\epsilon_0$')
    axs[1].set(ylabel='$\epsilon_1$')
    plt.xlabel('qubit number')
    for i in range(len(jobs)):
        plot_i(jobs[i],labels[i])
    plt.legend(loc='best')
    plt.show()
    
def expected_improvement(theta,phi,theta_est,phi_est,dtheta=0,dphi=0,plot=True):
    # returns the expected state preparation error with and without pre-rotations
    # theta = np.array(theta)
    # phi = np.array(phi)
    # theta_est = np.array(theta_est)
    # phi_est = np.array(phi_est)
    
    Delta = 0.5*( -np.cos(theta) + np.cos(theta)*np.cos(theta_est) + np.sin(theta)*np.sin(theta_est)*np.cos( (phi-phi_est)/2 ) )
    
    dDelta2 = 0.25*( (1-np.cos(theta_est))*np.sin(theta) + np.cos(theta)*np.sin(theta_est)*np.cos ((phi - phi_est)/2))**2 * dtheta**2 + 1/16 * (np.sin(theta)*np.sin(theta_est)*np.sin( (phi-phi_est)/2) )**2 * dphi**2
    
    P0 = np.sin(theta/2)**2
    dP0 = np.cos(theta/2)*np.sin(theta/2)*dtheta
    
    dDelta = np.sqrt(dDelta2)
    
    
    if plot:
        import matplotlib.pyplot as plt
        n = len(theta)
        plt.errorbar(range(n),Delta, dDelta,capsize=5,label='expected improvement')
        plt.errorbar(range(n),P0,dP0,capsize=5,label='max improvement')
        plt.xlabel('qubit number')
        plt.ylabel('improvement in fidelity from pre-rotations')
        plt.legend(loc='best')
        plt.axhline(y=0, color='k')
        plt.tight_layout()
        plt.show()
    
    return Delta, dDelta


def expected_paulis(theta,phi,theta_est,phi_est,dtheta=0,dphi=0,plot=True):
    
    X0 = np.sin(theta)*np.cos(phi)
    dX0 = np.sqrt((np.cos(theta)*np.cos(phi)*dtheta)**2 + (np.sin(theta)*np.sin(phi)*dphi)**2)
    
    Y0 = np.sin(theta)*np.sin(phi)
    dY0 = np.sqrt((np.cos(theta)*np.sin(phi)*dtheta)**2 + (np.sin(theta)*np.cos(phi)*dphi)**2)
    
    Z0 = np.cos(theta)
    dZ0 = np.sin(theta)*dtheta
    
    
    X = np.sin(theta)*np.cos(theta_est)*np.cos(phi-phi_est) - np.cos(theta)*np.sin(theta_est)
    dX = np.sqrt(  (np.cos(theta)*np.cos(theta_est)*np.cos(phi-phi_est) + np.sin(theta)*np.sin(theta_est))**2 * dtheta**2 + (np.sin(theta)*np.cos(theta_est)*np.sin(phi-phi_est)*dphi)**2  )
    
    Y = np.sin(theta)*np.sin(phi-phi_est)
    dY = np.sqrt( (np.cos(theta)*np.sin(phi-phi_est)*dtheta)**2 + (np.sin(theta)*np.cos(phi-phi_est)*dphi)**2    )
    
    Z = np.cos(theta)*np.cos(theta_est) + np.sin(theta)*np.sin(theta_est)*np.cos(phi-phi_est)
    dZ = np.sqrt( (-np.sin(theta)*np.cos(theta_est) + np.cos(theta)*np.sin(theta_est)*np.cos(phi-phi_est))**2 * dtheta**2  + (np.sin(theta)*np.sin(theta_est)*np.sin(phi-phi_est)*dphi)**2   )
    
    
    if plot:
        import matplotlib.pyplot as plt
        n = len(theta)
        fig, axs = plt.subplots(3,1,sharex=True)
        axs[0].errorbar(range(n),X0, dX0,capsize=5,label='without prerotations')
        axs[0].errorbar(range(n),X, dX,capsize=5,label='with prerotations')
        axs[1].errorbar(range(n),Y0, dY0,capsize=5,label='without prerotations')
        axs[1].errorbar(range(n),Y, dY,capsize=5,label='with prerotations')
        axs[2].errorbar(range(n),Z0, dZ0,capsize=5,label='without prerotations')
        axs[2].errorbar(range(n),Z, dZ,capsize=5,label='with prerotations')
        plt.xlabel('qubit number')
        axs[0].set(ylabel='expected $<X>$')
        axs[1].set(ylabel='expected $<Y>$')
        axs[2].set(ylabel='expected $<Z>$')
        plt.legend(loc='best')
        axs[0].axhline(y=0, color='k')
        axs[1].axhline(y=0, color='k')
        axs[2].axhline(y=1, color='k')
        plt.tight_layout()
        plt.show()
    
    return X0, dX0, Y0, dY0, Z0, dZ0, X, dX, Y, dY, Z, dZ


# def entropy_r(r):
    # S = np.sum( 


# def overall_damping(r):
    # # finds the overall depolarizing probability p with the same von Neumann entropy as the initial state.
        
