#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4Gamma.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"
#include <cmath>

namespace {
    // Pure-cylinder crystal front face is z_g4 = +45 mm.
    // Source is 100 mm in front of the face.
    const G4double SOURCE_Z = 145.0 * mm;

    // Preserve the requested off-corner source placement for contact 3.
    // The pure cylinder has radius 30 mm, but 29 mm is retained to match
    // the source coordinates of the original shaped-detector setup.
    const G4double SOURCE_X = (29.0/std::sqrt(2.0) + 1.0) * mm;
    const G4double SOURCE_Y = (-29.0/std::sqrt(2.0) - 1.0) * mm;

    const G4double CO60_E1 = 1173.2 * keV;
    const G4double CO60_E2 = 1332.5 * keV;
    const bool USE_CONE_SOURCE = true;
    const G4double CONE_HALF_ANGLE = 40.0 * deg;

    // false: one 1332.5-keV gamma per Geant4 event.
    // true: one 1173.2-keV and one 1332.5-keV gamma per event.
    const bool EMIT_BOTH_CO60_GAMMAS = false;
}

PrimaryGeneratorAction::PrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction(), fParticleGun(new G4ParticleGun(1)) {
    fParticleGun->SetParticleDefinition(G4Gamma::Definition());
    fParticleGun->SetParticlePosition(G4ThreeVector(SOURCE_X,SOURCE_Y,SOURCE_Z));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete fParticleGun; }

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
    const auto dir1 = USE_CONE_SOURCE ? SampleDirectionToDetectorCone()
                                     : SampleIsotropicDirection();
    const auto dir2 = USE_CONE_SOURCE ? SampleDirectionToDetectorCone()
                                     : SampleIsotropicDirection();
    if (EMIT_BOTH_CO60_GAMMAS) {
        fParticleGun->SetParticleEnergy(CO60_E1);
        fParticleGun->SetParticleMomentumDirection(dir1);
        fParticleGun->GeneratePrimaryVertex(event);
    }
    fParticleGun->SetParticleEnergy(CO60_E2);
    fParticleGun->SetParticleMomentumDirection(dir2);
    fParticleGun->GeneratePrimaryVertex(event);
}

G4ThreeVector PrimaryGeneratorAction::SampleIsotropicDirection() const {
    const G4double cosTheta = 2.0*G4UniformRand() - 1.0;
    const G4double sinTheta = std::sqrt(1.0-cosTheta*cosTheta);
    const G4double phi = 2.0*CLHEP::pi*G4UniformRand();
    return G4ThreeVector(sinTheta*std::cos(phi),
                         sinTheta*std::sin(phi), cosTheta).unit();
}

G4ThreeVector PrimaryGeneratorAction::SampleDirectionToDetectorCone() const {
    // Cone centered on global -z, uniform in solid angle.
    const G4double cosThetaMax = std::cos(CONE_HALF_ANGLE);
    const G4double cosTheta = cosThetaMax +
        (1.0-cosThetaMax)*G4UniformRand();
    const G4double sinTheta = std::sqrt(1.0-cosTheta*cosTheta);
    const G4double phi = 2.0*CLHEP::pi*G4UniformRand();
    return G4ThreeVector(sinTheta*std::cos(phi),
                         sinTheta*std::sin(phi), -cosTheta).unit();
}
