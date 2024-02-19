from policy import HSDP,SADP
from utils import f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Initialization
    T=20000
    betamax=3.0
    pmin=0.1
    dmax=1
    L=5
    N=6

    # Repeat
    round = 30

    # Experiment setting
    def demand_gur(p):
        return f(p,"from_gur",T=T) # The smoothness parameter is 0.8
    beta_list_gur = np.linspace(0.5,3,6)
    df_record_gur = pd.DataFrame(columns=["mean"])


    # Simulation
    regret_gur_list = []
    for i in range(round):
        regret_gur = SADP(demand=demand_gur,T=T,J=N,betamax=betamax,L=L,pmin=pmin,dmax=dmax)
        regret_gur_list.append(regret_gur)
    mean = np.mean(regret_gur_list)
    df_record_gur = pd.concat([df_record_gur,pd.DataFrame({"mean":[mean]},index=["Adaptive"])],axis=0)

    for beta in beta_list_gur:
        regret_gur_list = []
        for i in range(round):
            regret_gur = HSDP(demand=demand_gur,T=T,J=N,beta=beta,L=L,pmin=pmin,dmax=dmax)
            regret_gur_list.append(regret_gur)
        mean = np.mean(regret_gur_list)
        df_record_gur = pd.concat([df_record_gur,pd.DataFrame({"mean":[mean]},index=[f"beta:{beta}"])],axis=0)
        
    x_values = beta_list_gur
    y_values = df_record_gur['mean'][1:] * 100
    adaptive_value = df_record_gur['mean'][0] * 100

    # Plot the regret
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, color='red', marker='s', label='$HSDP(\\tilde{\\beta})$')
    plt.plot(x_values, [adaptive_value] * len(x_values), color='blue', marker='^', linestyle='--', label='$SADP$')

    plt.xlabel('$\\beta$', fontsize=14)
    plt.ylabel('Mean Relative Regret (%)', fontsize=14)
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()

    plt.savefig("gur_setting.pdf",format="pdf")

