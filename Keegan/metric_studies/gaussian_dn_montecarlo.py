import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import joblib as jb

from scipy.optimize import curve_fit
from tqdm import tqdm



def GenerateDataSeperated(T,delmu=5,dist="gauss"):
    """ 
    generates 1D data (pilfered from Yoni's notebook)
        gauss - gaussians separated by delmu
        exp - exponential distribution (0 for x<delmu and 1 at x=delmu)
        exptrunc - exponential dsitribution (truncated to [0,1] one peaked at 1 and the other at 0)
    """

    if dist=="gauss":
        zsA = scp.stats.norm.rvs(loc=0,scale=1,size=int(T/2.0)) #first Gaussian with mean 0 and variance 1
        zsB = scp.stats.norm.rvs(loc=delmu,scale=1,size=int(T/2.0)) # second Gaussian with mean delmu and variance 1

    elif dist=="exp":
        zsA = scp.stats.expon.rvs(loc=0,scale=1,size=int(T/2.0)) #first Gaussian with mean 0 and variance 1
        zsB = scp.stats.expon.rvs(loc=delmu,scale=1,size=int(T/2.0)) # second Gaussian with mean delmu and variance 1
        
    elif dist=="exptrunc": # in this case delmu is interpretted as \lambda (the scaling) rather than the difference in the means
        zsA = np.log((np.exp(delmu)-1) * np.random.random(T) + 1) / delmu
        zsB = np.log((np.exp(-delmu)-1) * np.random.random(T) + 1) / (-delmu)

    return zsA, zsB
    

def unit_test_new_dist(dist="gauss"):

    zsA, zsB = GenerateDataSeperated(1000,5,dist)
    plt.hist(zsA)
    plt.hist(zsB)
    plt.savefig(f"generated_data_test_{dist}.pdf")
    plt.clf()


def SEMD_metric(x,xp):

    z = np.array(x * (1-x))
    zp = np.array(xp * (1-xp))

    #print(z/zp)

    d = 2 * np.sqrt(1 - np.min(np.min([z/zp,zp/z])))

    return d


def getmind(n,delmu=5,dist="gauss"):
    """compute the minimum distance between the sampled gaussians"""
    
    #gaussA, gaussB = GenerateDataSeperated(n,delmu,dist)

    #mins = []

    #for i in range(len(gaussA)):
    #    mins.append(min(abs(gaussB-gaussA[i])))

    zA, zB = GenerateDataSeperated(n,delmu,dist)

    mins = []

    for i in range(len(zA)):
        mins.append(min(abs(zB-zA[i])))
        #mins.append(np.min(SEMD_metric(zB,zA[i])))

    #print(mins)

    return min(mins)
    
    
    
def fit_scaling_line_and_plot(dndict,mukey="5",dist="gauss"):
    """courtesy of our friend chatgpt"""

    def linear_model(x, m, b):
        return m * x + b

    x_data = np.log10(np.array(dndict["ns"]))
    y_data = np.log10(np.array(dndict["dnbar"]))
    y_errors = np.array(dndict["dnstd"])/(np.array(dndict["dnbar"]) * np.log(10))# d Log10(y) = dy / (ln(10) * y)

    # Fit a line
    popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_errors, absolute_sigma=True)
    m_fit, b_fit = popt
    m_err, b_err = np.sqrt(np.diag(pcov))
    
    print(f"stds for linear fit: dm = {m_err} , db = {b_err}")
    
    # Fit a line with a scaling prior (m=-1)
    poptp, pcovp = curve_fit(lambda x,b: linear_model(x,-1.0,b), x_data, y_data, sigma=y_errors, absolute_sigma=True)
    b_fitp = poptp[0]
    b_errp = np.sqrt(np.diag(pcovp))[0]
    
    print(f"stds for -1 scaling fit: db = {b_errp}")

    # Fit a line with a scaling prior (m=-2)
    poptp2, pcovp2 = curve_fit(lambda x,b: linear_model(x,-2.0,b), x_data, y_data, sigma=y_errors, absolute_sigma=True)
    b_fitp2 = poptp2[0]
    b_errp2 = np.sqrt(np.diag(pcovp2))[0]
    
    print(f"stds for -2 scaling fit: db = {b_errp2}")
    
    # Create line for plotting
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = linear_model(x_fit, m_fit, b_fit)
    y_fitp = linear_model(x_fit, -1.0, b_fitp)
    y_fitp2 = linear_model(x_fit, -2.0, b_fitp2)
    
    # Plot
    plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', label='data')
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: Log10(dn) = {m_fit:.2f}Log10(n) + {b_fit:.2f}')
    plt.plot(x_fit, y_fitp, 'g--', label=f'Linear: Log10(dn) = -Log10(n) + {b_fitp:.2f}')
    plt.plot(x_fit, y_fitp2, 'y--', label=f'Quadratic: Log10(dn) = -2Log10(n) + {b_fitp2:.2f}')
    plt.xlabel('Log10(n)')
    plt.ylabel('Log10(dn)')
    plt.title(f'dn scaling with n - separated {dist} (delmu = {mukey})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"log_dns_{dist}_mu_is_{mukey}.png")
    plt.clf()

    
    
def get_dn_scaling_w_n(delmu=5,mukey="5",dist="gauss"):

    RUN = True
    
    nmin, nmax = [2,4]
    
    if dist=="gauss" and 10**nmin < np.exp(delmu**2 / 8):
        print(f"be aware that we are starting outside the small dn regime for mu={delmu}")
    
    if dist=="exp" and 10**nmin < np.exp(delmu / 2):
        print(f"be aware that we are starting outside the small dn regime for mu={delmu}")
    
    
    if RUN:
        ns = np.round(10**np.linspace(nmin,nmax,num=8)).astype(int)
        
        dns = np.array([[getmind(n,delmu,dist) for j in tqdm(range(100),desc="sample at fixed n",leave=False)] \
                       for n in tqdm(ns,desc="n",leave=False)])
        
        dndict = {"dns":dns,
                  "dnbar":[dn.mean() for dn in dns],
                  "dnstd":[dn.std() for dn in dns],
                  "ns":ns}
      
        jb.dump(dndict,f"dn_{dist}_mu_is_{mukey}.jb")
        #jb.dump(dndict,f"dn_{dist}_mu_is_{mukey}_SEMD.jb")
      
    else:
        dndict = jb.load(f"dn_{dist}_mu_is_{mukey}.jb")
      
    
    fit_scaling_line_and_plot(dndict,mukey,dist)
    
    
    
def main():

    distribution = ["gauss","exp","exptrunc"][2]

    '''
    zsA, zsB = GenerateDataSeperated(100,5,distribution)
    print(zsA)
    print(zsB)
    '''
    mus = [0,1,2.5,5][1:-1]
    #mus = []

    if distribution=="exptrunc" and 0 in mus:
        print("This value of mu will throw nans for the truncated exponential.")

    for mu in mus:
        print(mu)
        
        get_dn_scaling_w_n(mu,str(mu).replace(".","p"),distribution)

    
    
if __name__=="__main__":

    main()
    
