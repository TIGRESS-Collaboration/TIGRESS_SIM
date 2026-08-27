#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ThreeVector.hh"
class G4ParticleGun;
class G4Event;
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;
    void GeneratePrimaries(G4Event*) override;
private:
    G4ThreeVector SampleIsotropicDirection() const;
    G4ThreeVector SampleDirectionToDetectorCone() const;
    G4ParticleGun* fParticleGun = nullptr;
};
#endif
