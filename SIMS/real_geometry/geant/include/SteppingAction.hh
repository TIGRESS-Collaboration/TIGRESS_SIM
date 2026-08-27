#ifndef SteppingAction_h
#define SteppingAction_h

#include "G4UserSteppingAction.hh"

#include <fstream>

class EventAction;

class SteppingAction : public G4UserSteppingAction
{
public:
    SteppingAction(EventAction* eventAction);
    virtual ~SteppingAction();

    virtual void UserSteppingAction(const G4Step* step);

private:
    EventAction* fEventAction;

    static std::ofstream fHitsOut;
    static std::ofstream fTracksOut;
    static bool fHeaderWritten;
};

#endif