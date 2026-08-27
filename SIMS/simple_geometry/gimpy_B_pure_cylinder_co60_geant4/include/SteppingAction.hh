#ifndef SteppingAction_h
#define SteppingAction_h 1
#include "G4UserSteppingAction.hh"
#include <fstream>
class G4Step;
class EventAction;
class SteppingAction : public G4UserSteppingAction {
public:
    explicit SteppingAction(EventAction*);
    ~SteppingAction() override;
    void UserSteppingAction(const G4Step*) override;
private:
    EventAction* fEventAction = nullptr;
    static std::ofstream fHitsOut;
    static std::ofstream fTracksOut;
    static bool fHeaderWritten;
};
#endif
