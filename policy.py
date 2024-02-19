import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.linalg import inv
from math import log, floor
from sklearn.linear_model import RidgeCV

def HSDP(demand, J, L, beta, pmin, dmax, T=20000,parameterCI=8*1e-2, parametergamma=1e-5, initial_history=None):
    """
    Main Algorithm Function
    Parameters:
        J: Number of intervals
        T: Time horizon
        L: parameter
        beta: smoothness
        pmin: minimum price
        dmax: maximum demand
        parameterCI, parametergamma: Tuning parameters
        initial_history: history
    """
    k = floor(beta)
    a = np.zeros(J)
    b = np.zeros(J)
    length = 1 - pmin
    for j in range(J):
        a[j] = pmin + j * length / J
        b[j] = a[j] + length / J
    Delta = L * ((length / J) ** k)

    # Initialization
    p = np.zeros(T)
    d = np.zeros(T)
    revenue = np.zeros(T)
    tau = np.zeros(J)
    n = np.zeros(J)
    CI = np.zeros(J)
    select = np.zeros(J)
    hattheta = np.ones((J, k))
    hatp = (a + b) / 2
    Lambda = np.array([np.eye(k) for _ in range(J)])
    parameterexplore = 125
    lambda_ = np.zeros((k, J)) 
    phistory = np.zeros((J, T))
    dhistory = np.zeros((J, T))

    if initial_history is None:
        count = parameterexplore*J
        for j in range(J):  
            explore_price = np.linspace(a[j],b[j],parameterexplore)
            for idx, pt in enumerate(explore_price): 
                t = j * parameterexplore + idx 
                p[t] = pt
                d[t] = min(max(np.random.normal(demand(pt), 0.05), 0), 1) 
                revenue[t] = p[t] * demand(p[t])
                n[j] += 1
                phistory[j, t] = p[t]
                dhistory[j, t] = d[t]
                tau[j] += p[t] * d[t]
                CI[j] = parameterCI * (Delta + ((3 + L) * np.sqrt(2)) / np.sqrt(n[j])) * (k + 1) * np.log(2 * (k + 1) * T)
                select[j] = tau[j] / n[j] + CI[j]

    else:
        count = len(initial_history[0])
        p[:count] = initial_history[0][:count]
        d[:count] = initial_history[1][:count]
        revenue[:count] = initial_history[2][:count]
        for t in range(count):
            interval_index = (a > p[t]).argmax() - 1
            tau[j] += p[t] * d[t]
            n[interval_index] += 1
            phistory[interval_index, t] = p[t]
            dhistory[interval_index, t] = d[t]
        for j in range(J):
            CI[j] = parameterCI * (Delta + ((3 + L) * np.sqrt(2)) / np.sqrt(n[j])) * (k + 1) * np.log(2 * (k + 1) * T)
            select[j] = tau[j] / n[j] + CI[j]

    # Then start learning
    for t in range(count, T):
        # Select an interval, then do rigde regression
        optj = np.argmax(select)
        gamma = parametergamma * (L * np.sqrt(k) + Delta * np.sqrt(n[optj]) + np.sqrt(
            2 * k * np.log(max(1, 4 * k * (n[optj])) / (1 / T ** 2))) + 2)
        lambda_[:, optj] += dhistory[optj, int(n[optj]) - 1] * phi_local(phistory[optj, int(n[optj]) - 1], a[optj],
                                                                         b[optj], k).flatten()
        Lambda[optj, :, :] += phi_local(phistory[optj, int(n[optj]) - 1], a[optj], b[optj], k) * phi_local(
            phistory[optj, int(n[optj]) - 1], a[optj], b[optj], k).T
        hattheta[optj, :] = (inv(Lambda[optj, :, :]) @ lambda_[:, optj]).T

        # Search the best price
        stepsize = 0.01
        numberpexplored = int(np.floor(
            (b[optj] - a[optj]) / stepsize))  
        evenspacep = np.zeros(numberpexplored)
        averageevenspacerevenue = np.zeros(numberpexplored)

        for i in range(numberpexplored):
            evenspacep[i] = a[optj] + (i - 1) * stepsize
            averageevenspacerevenue[i] = -negaLBrevenue(evenspacep[i], hattheta[optj, :], a[optj], b[optj], k, gamma,
                                                        Lambda[optj, :, :], Delta)

        hatp[optj] = evenspacep[np.argmax(averageevenspacerevenue)]

        # Update
        p[t] = hatp[optj]
        d[t] = min(max(np.random.normal(demand(p[t]), 0.05),0),1)
        revenue[t] = p[t] * demand(p[t])
        n[optj] += 1
        phistory[optj, int(n[optj]) - 1] = p[t]
        dhistory[optj, int(n[optj]) - 1] = d[t]
        tau[optj] += p[t] * d[t]
        CI[optj] = parameterCI * (Delta + ((3 + L) * np.sqrt(2)) / np.sqrt(n[j])) * (k + 1) * np.log(2 * (k + 1) * T)
        select[optj] = tau[optj] / n[optj] + CI[optj]

    # Problem with complete info
    truefunc = lambda p: negatruerevenue(p,demand)
    p0 = (pmin + 1) / 2
    res = minimize(truefunc, p0, method='Nelder-Mead', bounds=[(pmin, 1)])
    trueoptp = res.x[0]
    trueoptrevenue = trueoptp * demand(trueoptp)
    regret = (trueoptrevenue - np.mean(revenue))  / trueoptrevenue

    return regret

def SADP(demand,betamax,J,L,pmin,dmax,T=20000,parameterCI=8 * 1e-2,parametergamma=1e-5,parameterbetahat=0.5):
    # Step 1: Set local polynomial regression degree
    l = floor(betamax)

    # Step 2: Initialize k1, k2, K1, and K2
    k1 = 1 / (2 * betamax + 2)
    k2 = 1 / (4 * betamax + 2)
    K1 = int(2 ** floor(k1 * log(T, 2)))
    K2 = int(2 ** floor(k2 * log(T, 2)))
    K = [K1, K2]
    k = [k1, k2]

    # Initialize regression parameters
    parameter1 = np.zeros((K1, l + 1))
    parameter2 = np.zeros((K2, l + 1))
    parameter = [parameter1, parameter2]

    # Initialize RidgeCV
    ridge = RidgeCV(alphas=[1e-8])

    # Initialize end_points_set
    end_points_set = []

    p_history = np.array([])
    d_history = np.array([])
    revenue_history = np.array([])

    for i in range(2):
        Ti = int(floor(T ** (0.5 + k[i])))
        trials = np.random.uniform(pmin, 1, Ti)
        outputs = np.array([min(max(np.random.normal(demand(p), 0.05),0),1) for p in trials])
        revenues = np.array([p * demand(p) for p in trials])
        end_points_set.append(np.linspace(pmin, 1, K[i] + 1))

        p_history = np.append(p_history, trials)
        d_history = np.append(d_history, outputs)
        revenue_history = np.append(revenue_history, revenues)

        point_sets = []
        output_sets = []

        # Separate the trials into bins
        for j in range(K[i]):
            choice = (trials >= end_points_set[i][j]) & (trials < end_points_set[i][j + 1])
            point_sets.append(trials[choice])
            output_sets.append(outputs[choice])

        # Perform local polynomial regression for each bin
        for interval_index in range(K[i]):
            X = np.zeros((len(point_sets[interval_index]), l + 1))
            for m, point in enumerate(point_sets[interval_index]):
                X[m, :] = phi_local(point, end_points_set[i][interval_index], end_points_set[i][interval_index + 1],
                                    l + 1).flatten()
            y = output_sets[interval_index]
            ridge.fit(X, y)
            parameter[i][interval_index, :] = ridge.coef_

    # Initialize maximum difference
    max_difference = 0
    points = np.linspace(pmin, 1, 10000)

    for point in points:
        interval_index1 = np.searchsorted(end_points_set[0], point) - 1
        interval_index2 = np.searchsorted(end_points_set[1], point) - 1

        # Calculate local polynomial approximations
        poly1 = np.dot(parameter[0][interval_index1, :], np.array([point ** i for i in range(l + 1)]))
        poly2 = np.dot(parameter[1][interval_index2, :], np.array([point ** i for i in range(l + 1)]))

        # Update maximum difference
        max_difference = max(max_difference, abs(poly1 - poly2))

    beta_estimation = -log(max_difference) / log(T) - log(log(T)) / log(T) + parameterbetahat * 4 * (betamax + 1) * log( log(T)) / log(T)
    initial_history = [p_history, d_history, revenue_history]
    regret = HSDP(demand=demand,J=J,T=T,L=L,beta=beta_estimation, pmin=pmin,dmax=dmax,parameterCI=parameterCI, parametergamma=parametergamma, initial_history=initial_history)
    return regret


def phi_local(p,a,b,k):
    return np.array([(1/2 + (p-(a+b)/2)/(b-a)) ** i for i in range(k)]).reshape(k,1)

def negaLBrevenue(p,hattheta,a,b,k,gamma,Lambda,Delta):
    phival = phi_local(p,a,b,k)
    return -p * min(1,np.dot(hattheta,phival)+ gamma*np.sqrt(phival.T @ (inv(Lambda) @ phival)) + Delta)

def negatruerevenue(p,demand):
    return -p * demand(p)