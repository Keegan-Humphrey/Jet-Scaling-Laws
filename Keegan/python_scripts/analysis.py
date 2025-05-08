import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import energyflow as ef

# ------------------ SETTINGS ------------------

dmax = 6
#dmax = 3
measure = 'hadr'
beta = 0.5
max_jets = 10000
epsilon = 1e-12  # for log stability

# ------------------ LOAD CSVs ------------------

qcd_df = pd.read_csv('../cpp_scripts/qcd_jets_filtered_particles.csv')
w_df = pd.read_csv('../cpp_scripts/w_jets_filtered_particles.csv')

# ------------------ GROUPING ------------------



def add_fallback_jet_id(df):
    
    #print(df.columns)

    if 'jet_id' in df.columns:
        #print("I'm in here!")
        return df
        
    df['jet_key'] = df['jet_pT'].round(3).astype(str) + '_' + df['jet_mass'].round(3).astype(str)
    
    return df
    

def group_jets(df, use_jet_id=False):

    df = add_fallback_jet_id(df)
    
    key = 'jet_id' if use_jet_id and 'jet_id' in df.columns else 'jet_key'
    
    jets = []
    for _, group in df.groupby(key):
        jets.append(group[['pT', 'eta', 'phi']].to_numpy())
        
    return jets

qcd_jets = group_jets(qcd_df,True) # group final state particles into their clustered jets
w_jets = group_jets(w_df,True)


# ------------------ SUBSAMPLE ------------------

n_samples = min(len(qcd_jets), len(w_jets), max_jets) # ensure the number of jets is equal in both data sets (restrict to max_jets in case data set is large)
qcd_jets = qcd_jets[:n_samples]
w_jets = w_jets[:n_samples]

print("analysing {} jets in each process".format(n_samples))

# ------------------ COMPUTE EFPs ------------------

print("Computing EFPs...")
efpset = ef.EFPSet(('d<=', dmax), measure=measure, beta=beta) # object which is used to: generate graphs and compute the EFPs for each jet

"""
EFPs are a basis for all IR safe observables. 

They are constructed out of the energy fraction (of the total jet energy) of a given particle as well as pairwise angles between particles. Products of these for a given set of particles form monomials which can be linearly combined to form all IR safe observables. Each monomial can be interpretted as a graph: each pairwise angle, an edge, and each particle / energy fraction, a vertex. 

There are a finite number of monomials / graphs at a given number of edges d (ie. degree of the angular monomial (angles are small in a jet so this is a good expansion parameter)), this is specified by dmax above. We can compute all graphs for a given jet at a given order using the EFPSet instance. These turn out to be great input data on which to train a NN on collider info.   

**** how does metric fit in?
**** how does measure fit in?
\beta is the angular weighting exponent used in the EMD metric 

see: 
**** cite Jesse Thaler's paper
**** cite documentation for energyflow  

"""

X_qcd = efpset.batch_compute(qcd_jets)
X_w = efpset.batch_compute(w_jets)

#print(X_qcd)

print(np.shape(X_qcd))

# ------------------ FIND K4 GRAPH ------------------

# Fully connected 4-node graph
k4_graph = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]

def canonical(g): return sorted(sorted(pair) for pair in g)

all_graphs = efpset.graphs() # generate a list of graphs to the given order (edges connecting the labeled vertices)

#print(all_graphs)

print(len(all_graphs))

#k4_index = next(i for i, g in enumerate(all_graphs) if canonical(g) == canonical(k4_graph))

try:
  k4_index = next(i for i, g in enumerate(all_graphs) if canonical(g) == canonical(k4_graph))
  
  print(f"K4 EFP index: {k4_index}")

except Exception:
  k4_index = -1
  
  print("Couldn't find k4 index, maybe your dmax isn't high enough?")



# ------------------ PLOT K4 HISTOGRAM ------------------

X_qcd_k4 = X_qcd[:, k4_index]
X_w_k4 = X_w[:, k4_index]

plt.figure(figsize=(10,6))
plt.hist(X_qcd_k4, bins=50, alpha=0.6, label='QCD jets', density=True, histtype='step')
plt.hist(X_w_k4, bins=50, alpha=0.6, label='W jets', density=True, histtype='step')
plt.title('K4 EFP Distribution')
plt.xlabel('EFP Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("k4_histogram.png")
plt.show()

# ------------------ SAVE LOG(EFPs) + LABELS ------------------

X_qcd_log = np.log(np.clip(X_qcd, epsilon, None))
X_w_log = np.log(np.clip(X_w, epsilon, None))

# Ensure same number of events
min_n = min(len(X_qcd_log), len(X_w_log))
X_all = np.vstack([X_qcd_log[:min_n], X_w_log[:min_n]])
y_all = np.array([0]*min_n + [1]*min_n)

df = pd.DataFrame(X_all)
df['label'] = y_all
df.to_csv('log_efps_labeled.csv', index=False)
print("Saved: log_efps_labeled.csv")
