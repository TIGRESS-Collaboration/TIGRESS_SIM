
#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "FTFP_BERT.hh"

int main(
    int argc,
    char** argv
)
{
    // Use serial mode because the output CSV streams are not
    // protected for multithreaded writing.
    auto* runManager =
        G4RunManagerFactory::CreateRunManager(
            G4RunManagerType::SerialOnly
        );

    runManager->SetUserInitialization(
        new DetectorConstruction()
    );

    // Match the physics list used by the old compact-output simulation.
    //
    // Do not explicitly enable fluorescence, Auger, or PIXE here.
    runManager->SetUserInitialization(
        new FTFP_BERT()
    );

    runManager->SetUserInitialization(
        new ActionInitialization()
    );

    runManager->Initialize();

    auto* visManager =
        new G4VisExecutive();

    visManager->Initialize();

    auto* uiManager =
        G4UImanager::GetUIpointer();

    if (argc == 1)
    {
        auto* ui =
            new G4UIExecutive(
                argc,
                argv
            );

        uiManager->ApplyCommand(
            "/control/execute ../vis.mac"
        );

        ui->SessionStart();

        delete ui;
    }
    else
    {
        G4String command =
            "/control/execute ";

        G4String fileName =
            argv[1];

        uiManager->ApplyCommand(
            command + fileName
        );
    }

    delete visManager;
    delete runManager;

    return 0;
}
