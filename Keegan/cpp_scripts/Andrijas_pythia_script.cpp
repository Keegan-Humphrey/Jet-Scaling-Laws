#include "Pythia8/Pythia.h"

#include "fastjet/ClusterSequence.hh"

#include <iostream>

#include <fstream>

using namespace Pythia8;
using namespace fastjet;

int main() {
    // Settings
    const double pTHatMin = 550.0;
    const double pTHatMax = 650.0;
    const double jetR = 0.4;
    const int nEvents = 1000000;
    const double minJetMass = 70.0;
    const double maxJetMass = 90.0;



// Output
bool verbose = false;
const int nProg = 1000;
std::ofstream out_qcd("qcd_jets_filtered_particles.csv");
std::ofstream out_w("w_jets_filtered_particles.csv");

//out_qcd << "pT,eta,phi\n";
//out_w << "pT,eta,phi\n";
out_qcd << "pT,eta,phi,jet_id\n";
out_w << "pT,eta,phi,jet_id\n";


for (std::string process : {"QCD", "W"}) {
    Pythia pythia;

    if (process == "QCD") {
        pythia.readString("HardQCD:all = on"); // .readstring() updates the parameters of the Pythia object using a string (can read the string or a par file name which it will read)
    } else if (process == "W") {
        pythia.readString("WeakBosonAndParton:qg2Wq = on");
        pythia.readString("WeakBosonAndParton:qqbar2Wg = on");
        pythia.readString("24:onMode = off");
        pythia.readString("24:onIfAny = 1 2 3 4 5");
    }

    if (!verbose) {
    	pythia.readString("Init:showAllSettings = off");   // Only if you want to print inputs
    	pythia.readString("Init:showChangedSettings = off");
    	pythia.readString("Init:showAllParticleData = off");
    	pythia.readString("Next:numberCount = 0");     // Show progress every N events
    	pythia.readString("Next:numberShowInfo = 0");
    	pythia.readString("Next:numberShowProcess = 0");
    	pythia.readString("Next:numberShowEvent = 0");
    }

    // *** ASK ANDRIJA I'm pretty sure that 500 -> the pTHatMin variable above
    // also it seems like this should get rid of the string fragmentation warnings, but fails to, so is it not getting read by pythia?
    //pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(500));
    pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(500)); // (for pythia) looser cut than for fastjet
    pythia.init();


    int jet_id = 0; // set a global label for each jet in the run for this process

    // Loop over events
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue; // returns next() returns False if event was unsuccessful (so this skips bad events)

	if (iEvent%nProg == 0) {std::cout << "Event: " << iEvent << std::endl;};

      /*
      PsuedoJets are extremely flexible data structures
    	they are initialized as in the loop below with the components of 4 momenta of the particle, jet, etc. that they represent
      they have built in methods for basic kinematics etc, also a set_user_index() method to introduce a lable to the PsuedoJet (accessed with .user_index())
      */
        std::vector<PseudoJet> particles;
        for (int i = 0; i < pythia.event.size(); ++i) { // Loop over particles in the event (event is an object with all event info, the method .size() returns the number of particles in the event)
            if (!pythia.event[i].isFinal()) continue; // .isFinal() return True if (ith) particle in the event is in the final state (weaker than .isVisible)
            if (!pythia.event[i].isVisible()) continue; // .isVisible() returns true if (ith) particle is in principle detectable (eg. not a neutrino)

            double px = pythia.event[i].px();
            double py = pythia.event[i].py();
            double pz = pythia.event[i].pz();
            double E = pythia.event[i].e();

            particles.emplace_back(px, py, pz, E); // Create a PsuedoJet with the particle 4-momentum (emplace_back() is more efficient than pushback() here)
        }

        JetDefinition jet_def(antikt_algorithm, jetR);
        ClusterSequence cs(particles, jet_def); // Cluster into jets using the Anti kt algorithm
        std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets()); // get ALL jets, regardless of pT (and impose a cut later), then sort output by decreasing pT

        for (const auto& jet : jets) {// Loop over the clustered jets in the event
            double jet_pT = jet.pt();
            double jet_mass = jet.m();
            if (jet_pT < pTHatMin || jet_pT > pTHatMax) continue; // restrict the jet pT to the range of interest (for fastjet)
            if (jet_mass < minJetMass || jet_mass > maxJetMass) continue; // restrict the jet mass to the range of interest

            //if (iEvent%100 == 0) {std::cout << "jet_ids in " << iEvent << " are: " << jet_id << std::endl;};

            for (const auto& constituent : jet.constituents()) {// Loop over the particles in the jet
                double pT = constituent.pt();
                double eta = constituent.eta();
                double phi = constituent.phi();
                if (process == "QCD")
                    out_qcd << pT << "," << eta << "," << phi << "," << jet_id << std::endl;
                else
                    out_w << pT << "," << eta << "," << phi << "," << jet_id << std::endl;
            }
            jet_id++;
        }
    }

    //pythia.stat(); // prints status etc.
    std::cout << "I just finished: " << process << std::endl;
    std::cout << "Great Success! I like!" << std::endl;
}

out_qcd.close();
out_w.close();

return 0;

}
