#include "DetectorConstruction.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

G4VPhysicalVolume* DetectorConstruction::Construct() {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* germanium = nist->FindOrBuildMaterial("G4_Ge");

    constexpr G4double outerRadius = 30.0 * mm;
    constexpr G4double halfHeight = 45.0 * mm;
    constexpr G4double coreRadius = 5.0 * mm;
    constexpr G4double coreCylinderHalfHeight = 35.0 * mm;
    constexpr G4double coreCylinderCenterZG4 = -10.0 * mm;
    constexpr G4double coreSphereRelativeZ = 35.0 * mm;

    auto* solidWorld = new G4Box("World", 500*mm, 500*mm, 500*mm);
    auto* logicWorld = new G4LogicalVolume(solidWorld, vacuum, "World");
    auto* physWorld = new G4PVPlacement(nullptr, {}, logicWorld, "World", nullptr, false, 0, true);

    auto* outer = new G4Tubs("PureCylinderOuter", 0.0, outerRadius, halfHeight, 0.0, 360.0*deg);
    auto* coreCylinder = new G4Tubs("CoreCylinder", 0.0, coreRadius,
        coreCylinderHalfHeight, 0.0, 360.0*deg);
    auto* coreSphere = new G4Sphere("CoreSphereCap", 0.0, coreRadius,
        0.0, 360.0*deg, 0.0, 180.0*deg);
    auto* roundedCore = new G4UnionSolid("RoundedCoreBore", coreCylinder,
        coreSphere, nullptr, G4ThreeVector(0,0,coreSphereRelativeZ));
    auto* crystal = new G4SubtractionSolid("GimpyBPureCylinderActiveCrystalSolid",
        outer, roundedCore, nullptr, G4ThreeVector(0,0,coreCylinderCenterZG4));

    // This exact logical-volume name is checked by SteppingAction.
    auto* logicCrystal = new G4LogicalVolume(
        crystal, germanium, "GimpyBPureCylinderActiveCrystal");
    new G4PVPlacement(nullptr, {}, logicCrystal,
        "GimpyBPureCylinderActiveCrystal", logicWorld, false, 0, true);

    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());
    auto* vis = new G4VisAttributes(G4Colour(0.0,0.35,1.0,0.45));
    vis->SetForceSolid(true);
    logicCrystal->SetVisAttributes(vis);
    return physWorld;
}
