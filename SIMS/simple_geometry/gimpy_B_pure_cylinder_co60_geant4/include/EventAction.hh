#ifndef EventAction_h
#define EventAction_h 1
#include "G4UserEventAction.hh"
#include "globals.hh"
#include <fstream>
class G4Event;
class EventAction : public G4UserEventAction {
public:
    EventAction();
    ~EventAction() override;
    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;
    void AddEdep(G4double value) { fTotalEdep += value; }
private:
    G4double fTotalEdep = 0.0;
    static std::ofstream fEventOut;
    static bool fHeaderWritten;
};
#endif
