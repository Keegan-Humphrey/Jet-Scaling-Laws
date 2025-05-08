import pythia8

# 1) Setup Pythia
pythia = pythia8.Pythia()

# Disable all of Pythia's own printouts
pythia.readString("Init:showAllSettings = off")
pythia.readString("Init:showChangedSettings = off")
pythia.readString("Init:showAllParticleData = off")
pythia.readString("Next:numberCount = 0")       # never auto-print an event
pythia.readString("Next:numberShowEvent = 0")   # suppress cross-section summaries

# Example: W production + QCD jets (tune to taste)
pythia.readString("Beams:eCM = 13000.")
pythia.readString("WeakSingleBoson:ffbar2W = on")
pythia.readString("HardQCD:all = on")

pythia.init()

# 2) Event loop
nEvents = 10
for iEv in range(nEvents):
    if not pythia.next():
        continue
    # 3) Print only final-state four‑momenta
    #    Columns: index, pdgID, px, py, pz, E
    for p in pythia.event:
        if not p.isFinal():
            continue
        print(f"{p.index():3d}  "
              f"{p.id():6d}  "
              f"{p.px():8.3f}  "
              f"{p.py():8.3f}  "
              f"{p.pz():8.3f}  "
              f"{p.e():8.3f}")

