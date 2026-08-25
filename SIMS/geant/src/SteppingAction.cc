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
: G4UserSteppingAction(),
  fEventAction(eventAction)
{
    if (!fHeaderWritten)
    {
        fHitsOut.open("geant_hits.csv");
        fTracksOut.open("geant_gamma_tracks.csv");

        fHitsOut
            << "event_id,"
            << "track_id,"
            << "x_mm,"
            << "y_mm,"
            << "z_g4_mm,"
            << "z_ssd_mm,"
            << "edep_keV"
            << std::endl;

        fTracksOut
            << "event_id,"
            << "track_id,"
            << "parent_id,"
            << "particle,"
            << "step_id,"
            << "x_pre_mm,"
            << "y_pre_mm,"
            << "z_pre_g4_mm,"
            << "z_pre_ssd_mm,"
            << "x_post_mm,"
            << "y_post_mm,"
            << "z_post_g4_mm,"
            << "z_post_ssd_mm,"
            << "edep_keV"
            << std::endl;

        fHeaderWritten = true;
    }
}

SteppingAction::~SteppingAction()
{
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();

    int eventID =
        G4EventManager::GetEventManager()
            ->GetConstCurrentEvent()
            ->GetEventID();

    int trackID = track->GetTrackID();
    int parentID = track->GetParentID();

    G4String particleName = track->GetParticleDefinition()->GetParticleName();

    G4ThreeVector pre = step->GetPreStepPoint()->GetPosition();
    G4ThreeVector post = step->GetPostStepPoint()->GetPosition();

    double edep = step->GetTotalEnergyDeposit();

    // ------------------------------------------------------------
    // Write gamma ray path steps.
    // This records primary and secondary gamma tracks.
    // ------------------------------------------------------------
    if (particleName == "gamma")
    {
        fTracksOut
            << eventID << ","
            << trackID << ","
            << parentID << ","
            << particleName << ","
            << track->GetCurrentStepNumber() << ","
            << pre.x() / mm << ","
            << pre.y() / mm << ","
            << pre.z() / mm << ","
            << pre.z() / mm + 45.0 << ","
            << post.x() / mm << ","
            << post.y() / mm << ","
            << post.z() / mm << ","
            << post.z() / mm + 45.0 << ","
            << edep / keV
            << std::endl;
    }

    // ------------------------------------------------------------
    // Write energy-deposition hit positions inside the active Ge.
    // ------------------------------------------------------------
    if (edep <= 0.0)
        return;

    auto volume =
        step->GetPreStepPoint()
            ->GetTouchableHandle()
            ->GetVolume()
            ->GetLogicalVolume();

    if (volume->GetName() != "MikeyAActiveCrystal")
        return;

    fEventAction->AddEdep(edep);

    // Use midpoint of the step instead of the pre-step point.
    // Pre-step positions can lie exactly on a boundary, which can later
    // make SSD think the electron cloud starts outside the semiconductor.
    G4ThreeVector hitPos = 0.5 * (pre + post);

    fHitsOut
        << eventID << ","
        << trackID << ","
        << hitPos.x() / mm << ","
        << hitPos.y() / mm << ","
        << hitPos.z() / mm << ","
        << hitPos.z() / mm + 45.0 << ","
        << edep / keV
        << std::endl;
}
