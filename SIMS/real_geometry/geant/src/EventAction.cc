#include "EventAction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"

EventAction::EventAction()
: G4UserEventAction(),
  fEdep(0.0)
{
    // Output will be written in the directory where you run the executable,
    // usually /data1/flerner/hpge_sims/geant/build
    fOut.open("geant_total_edep.csv");

    fOut << "event_id,total_edep_keV" << std::endl;
}

EventAction::~EventAction()
{
    if (fOut.is_open()) {
        fOut.close();
    }
}

void EventAction::BeginOfEventAction(const G4Event*)
{
    fEdep = 0.0;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    G4int eventID = event->GetEventID();

    // Write every event, even if edep is zero.
    fOut << eventID << "," << fEdep / keV << std::endl;

    // Also print only events with nonzero deposition.
    if (fEdep > 0.0) {
        G4cout
            << "Event " << eventID
            << " total Ge edep = "
            << fEdep / keV << " keV"
            << G4endl;
    }
}

void EventAction::AddEdep(G4double edep)
{
    fEdep += edep;
}