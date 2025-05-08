import numpy as np

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy.stats import norm, expon, gaussian_kde
from scipy.stats.sampling import TransformedDensityRejection
from scipy.optimize import curve_fit
from scipy import integrate

import matplotlib.pyplot as plt
import joblib as jb



def GenerateData(T,dist="gauss",delmu=5):
    """ generates 1D Gaussian data and labels with equal probability for both"""
    zs=np.zeros(T)
    
    #choose label first
    Y=np.random.default_rng().integers(2, size=T)

    #choose z conditional on label
    A_idx=np.where(Y==1)[0]
    B_idx=np.where(Y==0)[0]

    if dist=="gauss":
        zsA = norm.rvs(loc=0,scale=1,size=A_idx.shape[0]) # first Gaussian with mean 0 and variance 1
        zsB = norm.rvs(loc=delmu,scale=1,size=B_idx.shape[0]) # second Gaussian with mean delmu and variance 1

    elif dist=="exp":
        zsA = expon.rvs(loc=0,scale=1,size=A_idx.shape[0]) # first exp with mean 0 and variance 1
        zsB = expon.rvs(loc=delmu,scale=1,size=B_idx.shape[0]) # second exp with mean delmu and variance 1

    elif dist=="exptrunc": # in this case delmu is interpretted as \lambda (the scaling) rather than the difference in the means
        zsA = np.log((np.exp(delmu)-1) * np.random.random(A_idx.shape[0]) + 1) / delmu
        zsB = np.log((np.exp(-delmu)-1) * np.random.random(B_idx.shape[0]) + 1) / (-delmu)

    zs[A_idx] = zsA
    zs[B_idx] = zsB
    
    return zs.reshape(-1,1), Y


def FitLR(X_train, Y_train):
    LR = LogisticRegression()
    LR.fit(X_train, Y_train)
    return LR


def BCELoss(model, x, y):
    reg=1e-20
    logprob = np.log(model.predict_proba(x)+reg)
    loss = -(logprob[:,1] * y + logprob[:,0] * (1 - y)).mean()
    return loss
    

#def TrainLR(X_train, Y_train, X_test, Y_test, Ts, Navg = 10, dist="gauss", delmu=5):
def TrainLR(X_test, Y_test, Ts, Navg = 10, dist="gauss", delmu=5):

    lossarr = []

    for T in Ts:
        print(T)
        losses=[]
        for i in range(Navg):
            #idx = np.random.choice(np.arange(len(X_train)), T, replace=False) # this takes way too long if X_train is large (ie. 10^3 * 10^5 entries long). So just stick to sampling at each iteration. Wasn't an issue when we weren't averaging over the large n points
            #X_sel = X_train[idx]
            #Y_sel = Y_train[idx]
            X_sel, Y_sel = GenerateData(T,dist)
            cls = FitLR(X_sel, Y_sel)
            losses.append(BCELoss(cls,X_test,Y_test))
        lossarr.append(np.array(losses))
    
    return lossarr


def unit_test_GenerateData(dist="gauss"):
    testdat, labels = GenerateData(1000,dist)
    plt.hist(testdat)
    plt.savefig(f"generated_data_test_{dist}.pdf")
    plt.clf()
    

def get_loss_scaling(dist="gauss", delmu=5):
    
    Ts = [20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000, 5000, 6000, 8000, 10000, 20000, 30000, 50000, 100000]
    #Ts = np.linspace(3,6,10)
    
    _Navg = 1000
    #_Navg = 10
    
    X_test, Y_test = GenerateData(int(max(Ts)/2),dist, delmu)
    #X_train, Y_train = GenerateData(int(_Navg*max(Ts)),dist)
    
    #losses_LR = TrainLR(X_train, Y_train, X_test, Y_test, Ts, _Navg, dist, delmu)
    losses_LR = TrainLR(X_test, Y_test, Ts, _Navg, dist, delmu)
    
    return Ts, losses_LR


def plot_loss_scaling(Ts,losses_LR):
    
    plt.plot(Ts,losses_LR)
    plt.axhline(0.0172046,color='red',linestyle='dashed')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("absolute_loss.pdf")
    plt.clf()
    
    #approach to loss floor
    plt.plot(Ts,losses_LR-0.0172046,label='L - L0')
    plt.plot(Ts,3/np.array(Ts),linestyle='dashed',label='1/n')
    plt.plot(Ts,.1/np.array(Ts)**(1/2),linestyle='dashed',label=r'$1/\sqrt{n}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig("approach_to_floor.pdf")
    plt.clf()


def analyze_loss_scaling(losses_dict, dist="gauss", delmu=5, data_type="lossbar"):

    def linear_model(x, m, b, n):
        return np.log10(abs(m) / 10**(x * n) + abs(b))
        
    if dist=="gauss":
        #headtrunc = 14
        headtrunc = 0
        
    elif dist=="exptrunc":
        #headtrunc = 14
        #headtrunc = 13
        headtrunc = 16

    '''
    if not usemin:
        y_data = np.log10(np.array(losses_dict["lossbar"]))[headtrunc:]
        y_errors = (np.array(losses_dict["lossstd"])/(np.array(losses_dict["lossbar"]) * np.log(10)))[headtrunc:]

    else:
        y_data = np.log10(np.array(losses_dict["lossmin"]))[headtrunc:]
        y_errors = (np.array(losses_dict["lossstd"])/(np.array(losses_dict["lossmin"]) * np.log(10)))[headtrunc:]
    '''
        
    y_data = np.log10(np.array(losses_dict[data_type]))[headtrunc:]
    y_errors = (np.array(losses_dict["lossstd"])/(np.array(losses_dict[data_type]) * np.log(10)))[headtrunc:]
            
    x_data = np.log10(np.array(losses_dict["Ts"]))[headtrunc:]

    '''
    print("xdata = {",end="")
    [print(10**xd ,", ",end="") for xd in x_data]
    print("}")
    print("ydata = {",end="")
    [print(10**yd ,", ",end="") for yd in y_data]
    print("}")
    print(y_errors)
    '''
   
    #print(x_data)
    #print(y_data)
    #print(y_errors)
    
    
    # Fit to the modeal
    #popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_errors, absolute_sigma=True, p0=[0.40,0.034,0.71])
    popt, pcov = curve_fit(linear_model, x_data, y_data, p0=[0.40,0.034,0.71]) 
    m_fit, b_fit, n_fit = popt
    m_err, b_err, n_err = np.sqrt(np.diag(pcov))
    
    #print(f"stds for linear fit: dm = {m_err} , db = {b_err}, dn = {n_err}")
    
    '''
    # Fit a line with a scaling prior (n=-1)
    poptp, pcovp = curve_fit(lambda x,m,b: linear_model(x,m,b,-1.0), x_data, y_data, sigma=y_errors, absolute_sigma=True,p0=[1e4,0.034])
    m_fitp, b_fitp = poptp
    b_errp = np.sqrt(pcovp[0,0])

    # Fit a line with a scaling prior (n=-2)
    poptp2, pcovp2 = curve_fit(lambda x,m,b: linear_model(x,m,b,-2.0), x_data, y_data, sigma=y_errors, absolute_sigma=True,p0=[5,0.034])
    m_fitp2, b_fitp2 = poptp2
    b_errp2 = np.sqrt(pcovp2[0,0])
    '''
    
    # Create line for plotting
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = linear_model(x_fit, m_fit, b_fit,n_fit)
    #y_fitp = linear_model(x_fit, m_fitp, b_fitp,-1.0)
    #y_fitp2 = linear_model(x_fit, m_fitp2, b_fitp2,-2.0)
    
    
    # Plot
    #plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', label='data')
    plt.errorbar(x_data, y_data, fmt='o', label='data')
    plt.axhline(np.log10(abs(b_fit)),color='red',linestyle='dashed')
    #plt.plot(x_fit, y_fit, 'r-', label=f'Fit: L = {m_fit:.2f} / Ts^({n_fit:.2f}+-{n_err:.2f}) + ({b_fit:.4f}+-{b_err:.2f})')
    #plt.plot(x_fit, y_fit, 'r-', label=f'Fit: L = {m_fit:.2f} / Ts^({n_fit:.2f}+-{n_err:.2f}) + ({b_fit:.4f}+-{b_err:.2f})')
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: L = {m_fit:.2f} / Ts^({n_fit:.2f}+-{n_err:.2f}) + ({b_fit:.2e}+-{b_err:.2e})')
    #plt.plot(x_fit, y_fitp, 'g--', label=f'Linear: L = {m_fitp:.1f} / Ts^1 + ({b_fitp:.1e}+-{b_errp:.1e})')
    #plt.plot(x_fit, y_fitp2, 'y--', label=f'Quadratic: L = {m_fitp2:.1f} / Ts^2 + ({b_fitp2:.1e}+-{b_errp2:.1e})')
    plt.xlabel('log10(Ts)')
    plt.ylabel('log10(L)')
    plt.title(f'L scaling with T - {dist}, mu = {delmu}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Loss_fit_{data_type}_{dist}_mu_is_{delmu}_trunc_is_{headtrunc}.pdf")
    plt.clf()

    #print(f"Loss_fit_{dist}_mu_is_{delmu}_trunc_is_{headtrunc}.pdf")

    # Plot (without the loss floor)
    #approach_uncertainty = np.sqrt(10**(2 * y_data) * y_errors**2 + b_err**2 / np.log(10)**2)/(10**(y_data)-b_fit )
    approach_uncertainty = 0
    plt.errorbar(x_data, np.log10(10**y_data-b_fit), yerr=approach_uncertainty, fmt='o', label='data')
    #plt.plot(x_fit, np.log10(10**y_fit-b_fit), 'r-', label=f'Fit: L = {m_fit:.2f} / Ts^{n_fit:.2f} + {b_fit:.4f}')
    plt.plot(x_fit, np.log10(10**y_fit-b_fit), 'r-', label=f'Fit: L = {m_fit:.2f} / Ts^({n_fit:.2f}+-{n_err:.2f}) + ({b_fit:.2e}+-{b_err:.2e})')
    #plt.plot(x_fit, np.log10(10**y_fitp-b_fitp), 'g--', label=f'Linear: L = {m_fitp:.1f} / Ts^1 + ({b_fitp:.1e}+-{b_errp:.1e})')
    #plt.plot(x_fit, np.log10(10**y_fitp2-b_fitp2), 'y--', label=f'Quadratic: L = {m_fitp2:.1f} / Ts^2 + ({b_fitp2:.1e}+-{b_errp2:.1e})')
    plt.xlabel('log10(Ts)')
    plt.ylabel('log10(L)')
    plt.title(f'L scaling with T (no floor) - {dist}, mu = {delmu}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Loss_fit_approach_{data_type}_{dist}_mu_is_{delmu}_trunc_is_{headtrunc}.pdf")
    plt.clf()


def get_mode_of_dist(data):
    
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)
    mode = x_vals[np.argmax(kde(x_vals))]
    return mode

    
def Run_or_analyze_loss(dist="gauss", delmu=5, RUN=True, ut_GD=True, data_type="lossbar"):
    
    if ut_GD:
        unit_test_GenerateData(dist)

    if RUN:
        print(f"Classifying {dist} separated by {delmu}")
    
        Ts,losses_LR = get_loss_scaling(dist, delmu)
        
        losses_dict = {"Ts":Ts,
                       "losses":losses_LR,
                       "lossmin":np.array([np.min(ls) for ls in losses_LR]),
                       "lossbar":np.array([ls.mean() for ls in losses_LR]),
                       "lossstd":np.array([ls.std() for ls in losses_LR])}
    
        jb.dump(losses_dict,f"losses_dict_{dist}_mu_is_{delmu}.jb")
        
        plot_loss_scaling(Ts,losses_dict["lossbar"])

        analyze_loss_scaling(losses_dict, dist, delmu)

    else:
        print(f"Analyzing loss for {dist} separated by {delmu}")
        
        losses_dict = jb.load(f"losses_dict_{dist}_mu_is_{delmu}.jb")
        
        losses_dict["lossmin"] = np.array([np.min(ls) for ls in losses_dict["losses"]])
        
        losses_dict["lossmode"] = np.array([get_mode_of_dist(ls) for ls in losses_dict["losses"]])
        
        analyze_loss_scaling(losses_dict, dist, delmu, data_type)
    
        
def main():
    
    dists = np.array(["gauss","exp","exptrunc"])[[2]]
    
    delmus = np.array([0, 1, 5, 100])[[2]]
    
    data_type = ["lossbar","lossmin","lossmode"][0]
    
    RUN = False
    ut_GD = False
    
    for dist in dists:
        for mu in delmus:
            Run_or_analyze_loss(dist, mu, RUN, ut_GD, data_type)
        
        

if __name__=="__main__":
    
    main()    
    
    '''
    COMMENTS / NOTES / TODOS:
    
    - try fixing the division in number between the two dists (probably won't do much but to be sure) - MAKE SURE TO SAVE CURRENT RUNS
        -- a good way to do this would just be to look at number of samples
        -- no, there's no reason to do this save my uninformed thought that this could throw off the scaling, and Yoni already chose to do this and is obviously more informed.
        
    - Whatever outcome is, make sure that mathematica and python agree on the fit scaling. 
        -- done, there was just a mismatch between function argument and passed data (function wasn't expecting log data)
        
    - Ask Yoni (or at least call his attention to) the different predicted scaling for the truncated exponential Loss scaling
    
    - Separate these into classes and merge it with the dn scaling script
    
    - it may also be interesting to look at how the L scaling and the floor change with various truncations of the head of the distribution.
        -- I think maybe the reason the correct scaling is only appearing at high n is that the points at low n (head) throw off the fit of the floor. So in reality we might get this scaling at lower n but it is skewed by the floor not properly being subtracted off. This can be seen in the approach plots where without sufficient truncation we see non-linear behaviour at the tail (indicating some residual floor)
        -- this can be confirmed by saving the value of the floor from the truncated distribution and using it in the fits / approach plots of the un (or at least less) truncated plots   
    
    '''
    
    
    
