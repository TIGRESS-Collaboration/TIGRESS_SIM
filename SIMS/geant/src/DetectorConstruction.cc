#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4IntersectionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"

#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include <cmath>

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction()
{
}

DetectorConstruction::~DetectorConstruction()
{
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4NistManager* nist = G4NistManager::Instance();

    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");
    G4Material* germanium = nist->FindOrBuildMaterial("G4_Ge");

    // ============================================================
    // Coordinate convention
    // ============================================================
    //
    // SSD geometry:
    //   z = 0 mm to 90 mm
    //
    // Geant4 geometry here:
    //   crystal centered at z = 0
    //   z = -45 mm to +45 mm
    //
    // Conversion for exported hits:
    //
    //   x_ssd = x_geant4
    //   y_ssd = y_geant4
    //   z_ssd = z_geant4 + 45 mm
    //
    // ============================================================

    const G4double R_OUTER = 30.0 * mm;
    const G4double OCT_APOTHEM = 29.0 * mm;

    const G4double CRYSTAL_HEIGHT = 90.0 * mm;
    const G4double CRYSTAL_HALF_HEIGHT = CRYSTAL_HEIGHT / 2.0;

    const G4double CORE_RADIUS = 5.0 * mm;
    const G4double CORE_HEIGHT = 75.0 * mm;
    const G4double CORE_CAP_RADIUS = CORE_RADIUS;

    // Rounded blind bore:
    //   cylinder from z_ssd = 0 to 70 mm
    //   spherical cap centered at z_ssd = 70 mm
    //   cap reaches z_ssd = 75 mm
    const G4double CORE_CYL_HEIGHT = CORE_HEIGHT - CORE_CAP_RADIUS; // 70 mm
    const G4double CORE_CYL_HALF_HEIGHT = CORE_CYL_HEIGHT / 2.0;    // 35 mm

    // SSD:
    //   core cylinder center z = 35 mm
    //   cap sphere center z = 70 mm
    //
    // Geant4-centered:
    //   core cylinder center z = 35 - 45 = -10 mm
    //   cap sphere center z = 70 - 45 = +25 mm
    //
    // In the union construction below, the core cylinder local origin
    // is its own center. The sphere is translated +35 mm relative to
    // the cylinder center. The whole bore is placed at z = -10 mm
    // when subtracting from the crystal.
    const G4double CORE_CYL_CENTER_Z_G4 = -10.0 * mm;
    const G4double CORE_SPHERE_RELATIVE_Z = 35.0 * mm;

    const G4double TAPER_START_Z_SSD = 60.0 * mm;
    const G4double TAPER_START_Z_G4 = TAPER_START_Z_SSD - 45.0 * mm; // +15 mm

    const G4double TAPER_ANGLE = 22.5 * deg;
    const G4double TAPER_ANGLE_RAD = 22.5 * CLHEP::pi / 180.0;

    const G4double CUT_BOX_WIDTH = 200.0 * mm;
    const G4double CUT_BOX_HALF_WIDTH = CUT_BOX_WIDTH / 2.0;

    // Same as Julia:
    // CUT_BOX_CENTER_OUTER = OCT_APOTHEM + (CUT_BOX_WIDTH / 2) / cosd(TAPER_ANGLE)
    const G4double CUT_BOX_CENTER_OUTER =
        OCT_APOTHEM + CUT_BOX_HALF_WIDTH / std::cos(TAPER_ANGLE_RAD);

    // ============================================================
    // Offset core bore
    // ============================================================

    const G4double CORE_DISTANCE_TO_CONTACT34_FLAT_SIDE = 27.2 * mm;

    const G4double CORE_OFFSET_X =
    OCT_APOTHEM - CORE_DISTANCE_TO_CONTACT34_FLAT_SIDE;

    const G4double CORE_OFFSET_Y = 0.0 * mm;
    // ============================================================
    // World volume
    // ============================================================

    G4Box* solidWorld = new G4Box(
        "World",
        500.0 * mm,
        500.0 * mm,
        500.0 * mm
    );

    G4LogicalVolume* logicWorld = new G4LogicalVolume(
        solidWorld,
        air,
        "World"
    );

    G4VPhysicalVolume* physWorld = new G4PVPlacement(
        nullptr,
        G4ThreeVector(),
        logicWorld,
        "World",
        nullptr,
        false,
        0,
        true
    );

    // ============================================================
    // Main cylinder
    // ============================================================

    G4Tubs* solidOuterCylinder = new G4Tubs(
        "OuterCylinder",
        0.0,
        R_OUTER,
        CRYSTAL_HALF_HEIGHT,
        0.0,
        360.0 * deg
    );

    // ============================================================
    // Octagonal mask = box intersect rotated box
    // ============================================================

    G4Box* solidBox0 = new G4Box(
        "OctBox0",
        OCT_APOTHEM,
        OCT_APOTHEM,
        CRYSTAL_HALF_HEIGHT
    );

    G4Box* solidBox45 = new G4Box(
        "OctBox45",
        OCT_APOTHEM,
        OCT_APOTHEM,
        CRYSTAL_HALF_HEIGHT
    );

    G4RotationMatrix* rotZ45 = new G4RotationMatrix();
    rotZ45->rotateZ(45.0 * deg);

    G4IntersectionSolid* solidOctMask = new G4IntersectionSolid(
        "OctagonalMask",
        solidBox0,
        solidBox45,
        rotZ45,
        G4ThreeVector()
    );

    // Main body = cylinder intersect octagonal mask.
    G4IntersectionSolid* solidMainBody = new G4IntersectionSolid(
        "MainBody",
        solidOuterCylinder,
        solidOctMask
    );

    // ============================================================
    // Rounded offset core bore = cylinder union sphere cap
    // ============================================================
    //
    // The bore solid is built in its own local coordinates:
    //
    //   cylinder centered at local z = 0
    //   sphere centered at local z = +35 mm
    //
    // The entire bore union is then subtracted from the crystal at:
    //
    //   x = CORE_OFFSET_X
    //   y = CORE_OFFSET_Y
    //   z = CORE_CYL_CENTER_Z_G4
    //
    // This is the Geant4 equivalent of the SSD bore:
    //
    //   tube translate:
    //     x = CORE_OFFSET_X
    //     y = CORE_OFFSET_Y
    //     z = 35 mm
    //
    //   sphere translate:
    //     x = CORE_OFFSET_X
    //     y = CORE_OFFSET_Y
    //     z = 70 mm
    //
    // after converting z_ssd -> z_g4 by subtracting 45 mm.
    //
    // ============================================================

    G4Tubs* solidCoreCylinder = new G4Tubs(
        "CoreCylinder",
        0.0,
        CORE_RADIUS,
        CORE_CYL_HALF_HEIGHT,
        0.0,
        360.0 * deg
    );

    G4Sphere* solidCoreSphere = new G4Sphere(
        "CoreSphereCap",
        0.0,
        CORE_CAP_RADIUS,
        0.0,
        360.0 * deg,
        0.0,
        180.0 * deg
    );

    G4UnionSolid* solidRoundedCoreBore = new G4UnionSolid(
        "RoundedOffsetCoreBore",
        solidCoreCylinder,
        solidCoreSphere,
        nullptr,
        G4ThreeVector(0.0, 0.0, CORE_SPHERE_RELATIVE_Z)
    );

    G4SubtractionSolid* solidAfterCore = new G4SubtractionSolid(
        "CrystalMinusRoundedOffsetCore",
        solidMainBody,
        solidRoundedCoreBore,
        nullptr,
        G4ThreeVector(
            CORE_OFFSET_X,
            CORE_OFFSET_Y,
            CORE_CYL_CENTER_Z_G4
        )
    );

    // ============================================================
    // Taper cuts
    // ============================================================
    //
    // SSD taper cuts:
    //
    // box widths: [200, 200, 200]
    //
    // Cut 1:
    //   origin: [-CUT_BOX_CENTER_OUTER, 0, 60]
    //   rotate Y: 22.5 deg
    //
    // Cut 2:
    //   origin: [0, +CUT_BOX_CENTER_OUTER, 60]
    //   rotate X: 22.5 deg
    //
    // Convert z origin from SSD to Geant4:
    //
    //   z_g4 = 60 - 45 = +15 mm
    //
    // Note:
    //   The negative rotations below were present in your original
    //   Geant4 construction and are kept here to preserve the same
    //   taper orientation that matched your previous Geant4 geometry.
    //
    // ============================================================

    G4Box* solidTaperBox = new G4Box(
        "TaperCutBox",
        CUT_BOX_HALF_WIDTH,
        CUT_BOX_HALF_WIDTH,
        CUT_BOX_HALF_WIDTH
    );

    G4RotationMatrix* rotY22p5 = new G4RotationMatrix();
    rotY22p5->rotateY(-TAPER_ANGLE);

    G4SubtractionSolid* solidAfterTaper1 = new G4SubtractionSolid(
        "CrystalMinusTaper1",
        solidAfterCore,
        solidTaperBox,
        rotY22p5,
        G4ThreeVector(
            -CUT_BOX_CENTER_OUTER,
            0.0,
            TAPER_START_Z_G4
        )
    );

    G4RotationMatrix* rotX22p5 = new G4RotationMatrix();
    rotX22p5->rotateX(-TAPER_ANGLE);

    G4SubtractionSolid* solidCrystal = new G4SubtractionSolid(
        "MikeyAActiveCrystal",
        solidAfterTaper1,
        solidTaperBox,
        rotX22p5,
        G4ThreeVector(
            0.0,
            CUT_BOX_CENTER_OUTER,
            TAPER_START_Z_G4
        )
    );

    // ============================================================
    // Logical and physical crystal volume
    // ============================================================

    G4LogicalVolume* logicCrystal = new G4LogicalVolume(
        solidCrystal,
        germanium,
        "MikeyAActiveCrystal"
    );

    new G4PVPlacement(
        nullptr,
        G4ThreeVector(0.0, 0.0, 0.0),
        logicCrystal,
        "MikeyAActiveCrystal",
        logicWorld,
        false,
        0,
        true
    );

    // ============================================================
    // Visualization
    // ============================================================

    G4VisAttributes* worldVis = new G4VisAttributes();
    worldVis->SetVisibility(false);
    logicWorld->SetVisAttributes(worldVis);

    G4VisAttributes* crystalVis = new G4VisAttributes(
        G4Colour(0.0, 0.35, 1.0, 0.45)
    );

    crystalVis->SetForceSolid(true);
    logicCrystal->SetVisAttributes(crystalVis);

    return physWorld;
}