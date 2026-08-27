#include "SteppingAction.hh"
#include "EventAction.hh"
#include "G4Step.hh"
#include "G4TouchableHandle.hh"
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolume.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"

std::ofstream SteppingAction::fHitsOut;
std::ofstream SteppingAction::fTracksOut;
bool SteppingAction::fHeaderWritten = false;

SteppingAction::SteppingAction(EventAction* eventAction)
: G4UserSteppingAction(), fEventAction(eventAction) {
    if (!fHeaderWritten) {
        fHitsOut.open("geant_hits.csv");
        fTracksOut.open("geant_gamma_tracks.csv");
        fHitsOut << "event_id,track_id,x_mm,y_mm,z_g4_mm,z_ssd_mm,edep_keV\n";
        fTracksOut << "event_id,track_id,parent_id,particle,step_id,"
                   << "x_pre_mm,y_pre_mm,z_pre_g4_mm,z_pre_ssd_mm,"
                   << "x_post_mm,y_post_mm,z_post_g4_mm,z_post_ssd_mm,edep_keV\n";
        fHeaderWritten = true;
    }
}

SteppingAction::~SteppingAction() {
    if (fHitsOut.is_open()) fHitsOut.flush();
    if (fTracksOut.is_open()) fTracksOut.flush();
}

void SteppingAction::UserSteppingAction(const G4Step* step) {
    auto* track = step->GetTrack();
    const auto* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
    const int eventID = currentEvent ? currentEvent->GetEventID() : -1;
    const int trackID = track->GetTrackID();
    const int parentID = track->GetParentID();
    const auto particleName = track->GetParticleDefinition()->GetParticleName();
    const auto pre = step->GetPreStepPoint()->GetPosition();
    const auto post = step->GetPostStepPoint()->GetPosition();
    const G4double edep = step->GetTotalEnergyDeposit();

    if (particleName == "gamma") {
        fTracksOut << eventID << ',' << trackID << ',' << parentID << ','
                   << particleName << ',' << track->GetCurrentStepNumber() << ','
                   << pre.x()/mm << ',' << pre.y()/mm << ',' << pre.z()/mm << ','
                   << pre.z()/mm + 45.0 << ',' << post.x()/mm << ','
                   << post.y()/mm << ',' << post.z()/mm << ','
                   << post.z()/mm + 45.0 << ',' << edep/keV << '\n';
    }

    if (edep <= 0.0) return;
    const auto touchable = step->GetPreStepPoint()->GetTouchableHandle();
    if (!touchable || !touchable->GetVolume()) return;
    auto* logical = touchable->GetVolume()->GetLogicalVolume();

    // Updated from MikeyAActiveCrystal to the pure-cylinder volume name.
    if (logical->GetName() != "GimpyBPureCylinderActiveCrystal") return;

    fEventAction->AddEdep(edep);
    const auto hitPos = 0.5*(pre+post);
    fHitsOut << eventID << ',' << trackID << ','
             << hitPos.x()/mm << ',' << hitPos.y()/mm << ','
             << hitPos.z()/mm << ',' << hitPos.z()/mm + 45.0 << ','
             << edep/keV << '\n';
}
