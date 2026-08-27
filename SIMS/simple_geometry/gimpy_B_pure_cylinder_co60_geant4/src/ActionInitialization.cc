#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
void ActionInitialization::Build() const {
    SetUserAction(new PrimaryGeneratorAction());
    auto* eventAction = new EventAction();
    SetUserAction(eventAction);
    SetUserAction(new SteppingAction(eventAction));
}
