#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "FTFP_BERT.hh"

int main(int argc, char** argv)
{
    // Serial mode is easier for debugging output files and visualization.
    auto* runManager =
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);

    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new FTFP_BERT());
    runManager->SetUserInitialization(new ActionInitialization());

    runManager->Initialize();

    // Always initialize visualization.
    G4VisManager* visManager = new G4VisExecutive();
    visManager->Initialize();

    G4UImanager* uiManager = G4UImanager::GetUIpointer();

    if (argc == 1) {
        // Interactive mode.
        G4UIExecutive* ui = new G4UIExecutive(argc, argv);

        // assumes you run from the build directory:
        //   ./ge_cylinder_test
        // and vis.mac is one directory above:
        //   ../vis.mac
        uiManager->ApplyCommand("/control/execute ../vis.mac");

        ui->SessionStart();

        delete ui;
    } else {
        // Batch mode.
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        uiManager->ApplyCommand(command + fileName);
    }

    delete visManager;
    delete runManager;

    return 0;
}