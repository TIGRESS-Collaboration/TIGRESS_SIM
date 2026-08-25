#include "PrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4Gamma.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

#include <cmath>

// ============================================================
// Source settings
// ============================================================
//
// Geant4 detector geometry:
//   crystal centered at z = 0
//   crystal extends from z = -45 mm to +45 mm
//
// SSD equivalent:
//   z_ssd = z_g4 + 45 mm
//   tapered/front end is z_ssd = 90 mm
//   tapered/front end is z_g4 = +45 mm
//
// Core offset convention:
//   Contact 3/4 flat side is the +x flat side.
//   +x flat side is at x = +OCT_APOTHEM = +29.0 mm.
//   Measured distance from core center to that flat side is 27.2 mm.
//   Therefore core center is at x = +1.8 mm, y = 0.0 mm.
//
// Important:
//   DetectorConstruction.cc and SSD YAML must use the same core offset.
// ============================================================

namespace
{
    const G4double CRYSTAL_HEIGHT = 90.0 * mm;
    const G4double CRYSTAL_HALF_HEIGHT = CRYSTAL_HEIGHT / 2.0;

    const G4double OCT_APOTHEM = 29.0 * mm;
    const G4double CORE_DISTANCE_TO_CONTACT34_FLAT_SIDE = 27.2 * mm;

    const G4double CORE_OFFSET_X =
        OCT_APOTHEM - CORE_DISTANCE_TO_CONTACT34_FLAT_SIDE;

    const G4double CORE_OFFSET_Y = 0.0 * mm;

    // Crystal front/taper face is at z_g4 = +45 mm.
    // Source is 100 mm in front of that face.
    const G4double SOURCE_DISTANCE_FROM_FRONT_FACE = 100.0 * mm;
    const G4double SOURCE_Z =
        CRYSTAL_HALF_HEIGHT + SOURCE_DISTANCE_FROM_FRONT_FACE;

    // For balanced superpulse statistics across all contacts, keep the
    // source and aim point centered on the crystal axis.
    //
    // If the real source was aligned to the shifted bore/core center,
    // change this to true.
    const bool AIM_AT_CORE_CENTER = false;

    const G4double SOURCE_X =
        AIM_AT_CORE_CENTER ? CORE_OFFSET_X : 0.0 * mm;

    const G4double SOURCE_Y =
        AIM_AT_CORE_CENTER ? CORE_OFFSET_Y : 0.0 * mm;

    const G4double AIM_X =
        AIM_AT_CORE_CENTER ? CORE_OFFSET_X : 0.0 * mm;

    const G4double AIM_Y =
        AIM_AT_CORE_CENTER ? CORE_OFFSET_Y : 0.0 * mm;

    const G4double AIM_Z = CRYSTAL_HALF_HEIGHT;

    // Co-60 main gamma lines.
    const G4double CO60_E1 = 1173.2 * keV;
    const G4double CO60_E2 = 1332.5 * keV;

    // Use cone source for efficiency.
    const bool USE_CONE_SOURCE = true;

    // 25 degrees should cover the crystal comfortably from 100 mm away.
    // If contact 1 remains underpopulated, try 30 deg.
    const G4double CONE_HALF_ANGLE = 25.0 * deg;

    // For clean 1332-keV superpulse generation, emit only 1332.5 keV.
    // Set true later for realistic Co-60 cascade validation.
    const bool EMIT_BOTH_CO60_GAMMAS = false;
}

PrimaryGeneratorAction::PrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction()
{
    fParticleGun = new G4ParticleGun(1);

    fParticleGun->SetParticleDefinition(G4Gamma::Definition());

    fParticleGun->SetParticlePosition(
        G4ThreeVector(SOURCE_X, SOURCE_Y, SOURCE_Z)
    );
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    G4ThreeVector dir1;
    G4ThreeVector dir2;

    if (USE_CONE_SOURCE) {
        dir1 = SampleDirectionToDetectorCone();
        dir2 = SampleDirectionToDetectorCone();
    } else {
        dir1 = SampleIsotropicDirection();
        dir2 = SampleIsotropicDirection();
    }

    if (EMIT_BOTH_CO60_GAMMAS) {
        fParticleGun->SetParticleEnergy(CO60_E1);
        fParticleGun->SetParticleMomentumDirection(dir1);
        fParticleGun->GeneratePrimaryVertex(event);
    }

    fParticleGun->SetParticleEnergy(CO60_E2);
    fParticleGun->SetParticleMomentumDirection(dir2);
    fParticleGun->GeneratePrimaryVertex(event);
}

G4ThreeVector PrimaryGeneratorAction::SampleIsotropicDirection() const
{
    G4double cosTheta = 2.0 * G4UniformRand() - 1.0;
    G4double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
    G4double phi = 2.0 * CLHEP::pi * G4UniformRand();

    G4double ux = sinTheta * std::cos(phi);
    G4double uy = sinTheta * std::sin(phi);
    G4double uz = cosTheta;

    return G4ThreeVector(ux, uy, uz).unit();
}

G4ThreeVector PrimaryGeneratorAction::SampleDirectionToDetectorCone() const
{
    const G4ThreeVector source(SOURCE_X, SOURCE_Y, SOURCE_Z);
    const G4ThreeVector aim(AIM_X, AIM_Y, AIM_Z);

    const G4ThreeVector axis = (aim - source).unit();

    // Construct a stable orthonormal basis perpendicular to the cone axis.
    G4ThreeVector helper(0.0, 0.0, 1.0);

    if (std::abs(axis.dot(helper)) > 0.99) {
        helper = G4ThreeVector(1.0, 0.0, 0.0);
    }

    const G4ThreeVector e1 = axis.cross(helper).unit();
    const G4ThreeVector e2 = axis.cross(e1).unit();

    // Uniform sampling in solid angle within the cone.
    const G4double cosThetaMax = std::cos(CONE_HALF_ANGLE);

    const G4double cosTheta =
        cosThetaMax +
        (1.0 - cosThetaMax) * G4UniformRand();

    const G4double sinTheta =
        std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));

    const G4double phi =
        2.0 * CLHEP::pi * G4UniformRand();

    const G4ThreeVector direction =
        cosTheta * axis +
        sinTheta * std::cos(phi) * e1 +
        sinTheta * std::sin(phi) * e2;

    return direction.unit();
}
