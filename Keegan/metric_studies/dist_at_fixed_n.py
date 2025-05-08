import numpy as np
import matplotlib.pyplot as plt
import joblib as jb

from scipy.optimize import curve_fit
from ScalingLaws1D_from_yoni import get_mode_of_dist

distdict = jb.load("losses_dict_exptrunc_mu_is_5.jb")

losses = distdict["losses"]

PLOT = False

mean_mode_ratio = []

for i in range(len(losses)):
    ni = distdict["Ts"][i]

    mean = np.log10(losses[i].mean())
    mode = np.log10(get_mode_of_dist(losses[i]))
    
    mean_mode_ratio.append(10**mean/10**mode)

    if PLOT:
      plt.hist(np.log10(losses[i]),bins=20)
      plt.axvline(mean,color="r",label=f"mean {mean:.5e}")
      plt.axvline(mode,color="g",label=f"mode {mode:.5e}")
      plt.xlabel('log10(L)')
      plt.title(f'L histogram - T={ni}')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(f"loss_dist_{i}.pdf")
      plt.clf()

#trunc = 14
trunc = 19

x_data = np.log10(distdict["Ts"][trunc:])
y_data = np.log10(mean_mode_ratio[trunc:])

def linear_model(x, m, b, n):
    return np.log10(abs(m) / 10**(x * n) + abs(b))
        
popt, pcov = curve_fit(linear_model, x_data, y_data)#, p0=[0.40,0.034,0.71])
#popt, pcov = curve_fit(linear_model, x_data, y_data, p0=[0.40,0.034,0.71]) 
m_fit, b_fit, n_fit = popt
m_err, b_err, n_err = np.sqrt(np.diag(pcov))
    
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = linear_model(x_fit, m_fit, b_fit,n_fit)


plt.plot(x_fit, y_fit, 'r-', label=f'ratio = {m_fit:.1f} / Ts^({n_fit:.2f}+-{n_err:.2f}) + ({b_fit:.2e}+-{b_err:.2e})')
plt.scatter(x_data,y_data)
plt.xlabel('log10(T)')
plt.title('log10(mean / mode)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_mode_ratios.pdf")
plt.clf()


plt.plot(x_fit, np.log10(10**y_fit-b_fit), 'r-', label=f'ratio = {m_fit:.1f} / Ts^({n_fit:.2f}+-{n_err:.2f}) + ({b_fit:.2e}+-{b_err:.2e})')
#plt.scatter(x_data,y_data)
plt.scatter(x_data,np.log10(10**y_data-b_fit))
plt.xlabel('log10(T)')
plt.title('log10(mean / mode) - no floor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_mode_ratios_approach.pdf")
plt.clf()









