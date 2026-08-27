#include "EventAction.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
std::ofstream EventAction::fEventOut;
bool EventAction::fHeaderWritten = false;

EventAction::EventAction() {
    if (!fHeaderWritten) {
        fEventOut.open("geant_event_totals.csv");
        fEventOut << "event_id,total_edep_keV\n";
        fHeaderWritten = true;
    }
}
EventAction::~EventAction() { if (fEventOut.is_open()) fEventOut.flush(); }
void EventAction::BeginOfEventAction(const G4Event*) { fTotalEdep = 0.0; }
void EventAction::EndOfEventAction(const G4Event* event) {
    if (fTotalEdep > 0.0)
        fEventOut << event->GetEventID() << ',' << fTotalEdep/keV << '\n';
}
