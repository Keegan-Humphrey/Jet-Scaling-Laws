#!/usr/bin/env python3
"""
A simple test script using Pythia for:
  - QCD jet events
  - W boson (and weak single boson) events

Make sure that Pythia8 has been compiled with Python support and
that the module is accessible as pythia8.
"""

import pythia8

def run_qcd_events(nEvents=5):
    print("=== Running QCD jet events ===")
    # Create a new Pythia instance for QCD processes.
    pythia = pythia8.Pythia()
    # Set proton-proton beams at 13 TeV.
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString("Beams:eCM = 13000.")
    # Enable all hard QCD processes to generate jet events.
    pythia.readString("HardQCD:all = on")
    # Optionally, you can tune additional parameters (like pT cuts).
    pythia.init()

    # Loop over events.
    for iEvent in range(nEvents):
        if not pythia.next():
            continue  # Skip events that fail generation.
        # Retrieve the event record size (number of particles in the event).
        nParticles = pythia.event.size()
        print(f"Event {iEvent}: {nParticles} final-state particles.")
    # Print statistics.
    pythia.stat()


def run_weak_single_boson_events(nEvents=5):
    print("\n=== Running Weak Single Boson events (W, Z) ===")
    # Create a new Pythia instance for W/Z production.
    pythia = pythia8.Pythia()
    # Set proton-proton beams at 13 TeV.
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString("Beams:eCM = 13000.")
    # Enable weak single boson production.
    pythia.readString("WeakSingleBoson:all = on")
    # Optionally, you can force a specific decay mode or further restrict processes.
    pythia.init()

    # Loop over events.
    for iEvent in range(nEvents):
        if not pythia.next():
            continue
        # Scan the event record for W bosons (PDG id = ±24).
        foundW = any(abs(p.id()) == 24 for p in pythia.event)
        if foundW:
            print(f"Event {iEvent}: W boson produced.")
        else:
            print(f"Event {iEvent}: No W boson in this event.")
    # Print statistics.
    pythia.stat()


if __name__ == '__main__':
    run_qcd_events(5)
    run_weak_single_boson_events(5)
