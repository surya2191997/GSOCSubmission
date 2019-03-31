#include <iostream>
#include <string>
#include <vector>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

void RNNClassification()
{
   TMVA::Tools::Instance();  

   TFile *input(0);
   TString fname = "./tmva_class_example.root";

   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
      TFile::SetCacheFileDir(".");
      input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD");
   }
   if (!input) {
      std::cout << "Could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- RNN Classification  : Using input file: " << input->GetName() << std::endl;
   
   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "TMVA_DNN.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Creating the factory object
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");

   // get the signal and background trees from input
   TTree *signalTree     = (TTree*)input->Get("TreeS");
   TTree *background     = (TTree*)input->Get("TreeB");

   // Add the variable used to train MVA methods
   dataloader->AddVariable( "myvar1 := var1*var2", 'F' );
   dataloader->AddVariable( "myvar2 := var2+var3", "Expression 2", "", 'F' );
   dataloader->AddVariable( "var3",                "Variable 3", "units", 'F' );
   dataloader->AddVariable( "var4",                "Variable 4", "units", 'F' );
   
   // Spectator Variables for correlation tests
   dataloader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );
   dataloader->AddSpectator( "spec2 := var2*5",  "Spectator 2", "units", 'F' );


   // global event weights per tree 
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 2.0;

   // Add an arbitrary number of signal or background trees
   dataloader->AddSignalTree    ( signalTree,     signalWeight );
   dataloader->AddBackgroundTree( background, backgroundWeight );

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|4");

   // Batch Layout
   TString batchLayoutString("BatchLayout=256|1|4");

   // General layout.
   TString layoutString ("Layout=RNN|128|4|1|0,RESHAPE|1|1|128|FLAT,DENSE|64|TANH,DENSE|2|LINEAR");

   // Training strategies to be booked by factory
   TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");

   TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
  
   TString trainingStrategyString ("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1;

   // General Options.
   TString rnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
                       "WeightInitialization=XAVIERUNIFORM");

   rnnOptions.Append(":"); rnnOptions.Append(inputLayoutString);
   rnnOptions.Append(":"); rnnOptions.Append(batchLayoutString);
   rnnOptions.Append(":"); rnnOptions.Append(layoutString);
   rnnOptions.Append(":"); rnnOptions.Append(trainingStrategyString);
   rnnOptions.Append(":Architecture=CPU");

   TCut mycuts = "abs(var2-0.5)<1"; 
   TCut mycutb = "abs(var1)<1"; 
   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb, "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );

   factory->BookMethod(dataloader, TMVA::Types::kDL, "DNN_CPU", rnnOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();
   
   // Save the output
   outputFile->Close();
   
   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;
   
   delete factory;
   delete dataloader;
}

int main(int argc, char ** argv) 
{
   RNNClassification();
   return 0;
}

