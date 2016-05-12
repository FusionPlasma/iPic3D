
#include "../include/iPic3D.h"

using namespace iPic3D;

int c_Solver::Init(int argc, char **argv) {
  // initialize MPI environment
  // nprocs = number of processors
  // myrank = rank of tha process*/
  mpi = new MPIdata(&argc, &argv);
  nprocs = mpi->nprocs;
  myrank = mpi->rank;

  col = new Collective(argc, argv); // Every proc loads the parameters of simulation from class Collective
  verbose = col->getVerbose();
  restart_cycle = col->getRestartOutputCycle();
  SaveDirName = col->getSaveDirName();
  RestartDirName = col->getRestartDirName();
  restart = col->getRestart_status();
  numberSpecies = col->getNs();            // get the number of particle species involved in simulation
  first_cycle = col->getLast_cycle() + 1; // get the last cycle from the restart
  // initialize the virtual cartesian topology 
  vct = new VCtopology3D(col);
  int tempNproc = vct->getNprocs();
  // Check if we can map the processes into a matrix ordering defined in Collective.cpp
  if (nprocs != vct->getNprocs()) {
    if (myrank == 0) {
      cerr << "Error: " << nprocs << " processes cant be mapped into " << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN() << " matrix: Change XLEN,YLEN, ZLEN in method VCtopology3D.init()" << endl;
      mpi->finalize_mpi();
      return (1);
    }
  }
  // We create a new communicator with a 3D virtual Cartesian topology
  vct->setup_vctopology(MPI_COMM_WORLD);
  // initialize the central cell index

#ifdef BATSRUS
  // set index offset for each processor
  col->setGlobalStartIndex(vct);
#endif

  nx0 = col->getNxc() / vct->getXLEN(); // get the number of cells in x for each processor
  ny0 = col->getNyc() / vct->getYLEN(); // get the number of cells in y for each processor
  nz0 = col->getNzc() / vct->getZLEN(); // get the number of cells in z for each processor
  // Print the initial settings to stdout and a file
  if (myrank == 0) {
    mpi->Print();
    vct->Print();
    col->Print();
    col->save();
  }
  // Create the local grid
  MPI_Barrier(MPI_COMM_WORLD);
  grid = new Grid3DCU(col, vct);  // Create the local grid
  EMf = new EMfields3D(col, grid);  // Create Electromagnetic Fields Object

    double kw, Vye, Vze, Vyp, Vzp;

  if (col->getSolInit()) {
    /* -------------------------------------------- */
    /* If using parallel H5hut IO read initial file */
    /* -------------------------------------------- */
    ReadFieldsH5hut(numberSpecies, false, EMf, col, vct, grid);

  }
  else {
    /* --------------------------------------------------------- */
    /* If using 'default' IO initialize fields depending on case */
    /* --------------------------------------------------------- */
    if      (col->getCase()=="GEMnoPert") EMf->initGEMnoPert(vct,grid,col);
    else if (col->getCase()=="ForceFree") EMf->initForceFree(vct,grid,col);
    else if (col->getCase()=="GEM")       EMf->initGEM(vct, grid,col);
    else if (col->getCase()=="BATSRUS")   EMf->initBATSRUS(vct,grid,col);
    else if (col->getCase()=="Dipole")    EMf->init(vct,grid,col);
    else if (col->getCase()=="shock") EMf->initShock(vct,grid,col);
    else if (col->getCase()=="langmuir") EMf->initLangmuir(vct, grid, col);
    else if (col->getCase()=="alfven") EMf->initAlfvenWave(vct,grid,col, 1, 0.01, kw, Vye, Vze, Vyp, Vzp);
    else {
      if (myrank==0) {
        cout << " =========================================================== " << endl;
        cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
        cout << "          Runing simulation with the default initialization. " << endl;
        cout << " =========================================================== " << endl;
      }
      EMf->init(vct,grid,col);
    }
  }

  // OpenBC
  EMf->updateInfoFields(grid,vct,col);

  // Allocation of particles
  part = new Particles3D[numberSpecies];
  if (col->getSolInit()) {
    if (col->getPartInit()=="File") ReadPartclH5hut(numberSpecies, part, col, vct, grid);
    else {
      if (myrank==0) cout << "WARNING: Particle drift velocity from ExB " << endl;
      for (int i = 0; i < numberSpecies; i++){
        part[i].allocate(i, 0, col, vct, grid);
        if (col->getPartInit()=="EixB") {
            part[i].MaxwellianFromFields(grid, EMf, vct);
        } else if (col->getPartInit()=="alfven"){
            if (i == 0) {
                part[i].maxwellianAlfven(grid, EMf, vct, kw, Vye, Vze);
            } else {
                part[i].maxwellianAlfven(grid, EMf, vct, kw, Vyp, Vzp);
            }
        } else
            part[i].maxwellian(grid, EMf, vct);
      }
    }
  }
  else {
    for (int i = 0; i < numberSpecies; i++)
      part[i].allocate(i, 0, col, vct, grid);

    // Initial Condition for PARTICLES if you are not starting from RESTART
    if (restart == 0) {
      // wave = new Planewave(col, EMf, grid, vct);
      // wave->Wave_Rotated(part); // Single Plane Wave
      for (int i = 0; i < numberSpecies; i++) {
          if (col->getCase() == "ForceFree") {
              part[i].force_free(grid, EMf, vct);
          } else if (col->getCase() == "BATSRUS") {
              part[i].MaxwellianFromFluid(grid, EMf, vct, col, i);
          } else if (col->getPartInit() == "alfven") {
              if (i == 0) {
                  part[i].maxwellianAlfven(grid, EMf, vct, kw, Vye, Vze);
              } else {
                  part[i].maxwellianAlfven(grid, EMf, vct, kw, Vyp, Vzp);
              }
          } else {
              part[i].maxwellian(grid, EMf, vct);
          }
      }
    }
  }

  if (col->getWriteMethod() == "default") {
    // Initialize the output (simulation results and restart file)
    // PSK::OutputManager < PSK::OutputAdaptor > output_mgr; // Create an Output Manager
    // myOutputAgent < PSK::HDF5OutputAdaptor > hdf5_agent; // Create an Output Agent for HDF5 output
    hdf5_agent.set_simulation_pointers(EMf, grid, vct, mpi, col);
    for (int i = 0; i < numberSpecies; ++i)
      hdf5_agent.set_simulation_pointers_part(&part[i]);
    output_mgr.push_back(&hdf5_agent);  // Add the HDF5 output agent to the Output Manager's list
    if (myrank == 0 & restart < 2) {
      hdf5_agent.open(SaveDirName + "/settings.hdf");
      output_mgr.output("collective + total_topology + proc_topology", 0);
      hdf5_agent.close();
      hdf5_agent.open(RestartDirName + "/settings.hdf");
      output_mgr.output("collective + total_topology + proc_topology", 0);
      hdf5_agent.close();
    }
    // Restart
    num_proc << myrank;
    if (restart == 0) {           // new simulation from input file
      hdf5_agent.open(SaveDirName + "/proc" + num_proc.str() + ".hdf");
      output_mgr.output("proc_topology ", 0);
      hdf5_agent.close();
    }
    else {                        // restart append the results to the previous simulation 
      hdf5_agent.open_append(SaveDirName + "/proc" + num_proc.str() + ".hdf");
      output_mgr.output("proc_topology ", 0);
      hdf5_agent.close();
    }
  }

  Eenergy, Benergy, TOTenergy = 0.0, TOTmomentum = 0.0;
  Ke = new double[numberSpecies];
  momentum = new double[numberSpecies];
  cq = SaveDirName + "/ConservedQuantities.txt";
  if (myrank == 0) {
    ofstream my_file(cq.c_str());
    my_file.close();
  }
  
  // // Distribution functions
  // nDistributionBins = 1000;
  // VelocityDist = new unsigned long[nDistributionBins];
  // ds = SaveDirName + "/DistributionFunctions.txt";
  // if (myrank == 0) {
  //   ofstream my_file(ds.c_str());
  //   my_file.close();
  // }
  cqsat = SaveDirName + "/VirtualSatelliteTraces" + num_proc.str() + ".txt";
  // if(myrank==0){
  ofstream my_file(cqsat.c_str(), fstream::binary);
  nsat = 3;
  for (int isat = 0; isat < nsat; isat++) {
    for (int jsat = 0; jsat < nsat; jsat++) {
      for (int ksat = 0; ksat < nsat; ksat++) {
        int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
        int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
        int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
        my_file << grid->getXC(index1, index2, index3) << "\t" << grid->getYC(index1, index2, index3) << "\t" << grid->getZC(index1, index2, index3) << endl;
      }}}
  my_file.close();

  Qremoved = new double[numberSpecies];

  my_clock = new Timing(myrank);

  return 0;
}

void c_Solver::GatherMoments(){
  // timeTasks.resetCycle();
  // interpolation
  // timeTasks.start(TimeTasks::MOMENTS);

  EMf->updateInfoFields(grid,vct,col);
  EMf->setZeroDensities();                  // set to zero the densities

  for (int i = 0; i < numberSpecies; i++)
    part[i].interpP2G(EMf, grid, vct);      // interpolate Particles to Grid(Nodes)

  EMf->sumOverSpecies(vct);                 // sum all over the species
    EMf->sumOverSpeciesJ();
  //
  // Fill with constant charge the planet
  if (col->getCase()=="Dipole") {
    EMf->ConstantChargePlanet(grid, vct, col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());
  }

  // EMf->ConstantChargeOpenBC(grid, vct);     // Set a constant charge in the OpenBC boundaries

}

void c_Solver::UpdateCycleInfo(int cycle) {

  EMf->UpdateFext(cycle);
  if (myrank == 0) cout << " Fext = " << EMf->getFext() << endl;
  if (cycle == first_cycle) {
    if (col->getCase()=="Dipole") {
      EMf->SetDipole_2Bext(vct,grid,col);
      EMf->SetLambda(grid);
    }
  }


}

void c_Solver::CalculateField() {

  // timeTasks.resetCycle();
  // interpolation
  // timeTasks.start(TimeTasks::MOMENTS);

  EMf->interpDensitiesN2C(vct, grid);       // calculate densities on centers from nodes
  EMf->calculateHatFunctions(grid, vct);    // calculate the hat quantities for the implicit method
  MPI_Barrier(MPI_COMM_WORLD);
  // timeTasks.end(TimeTasks::MOMENTS);

  // MAXWELL'S SOLVER
  // timeTasks.start(TimeTasks::FIELDS);
  EMf->calculateE(grid, vct, col);               // calculate the E field
  // timeTasks.end(TimeTasks::FIELDS);

}

void c_Solver::CalculateBField() {
  /* --------------------- */
  /* Calculate the B field */
  /* --------------------- */

  // timeTasks.start(TimeTasks::BFIELD);
  EMf->calculateB(grid, vct, col);   // calculate the B field
  // timeTasks.end(TimeTasks::BFIELD);

  // print out total time for all tasks
  // timeTasks.print_cycle_times();
}

bool c_Solver::ParticlesMover() {

  /*  -------------- */
  /*  Particle mover */
  /*  -------------- */

  // timeTasks.start(TimeTasks::PARTICLES);
  for (int i = 0; i < numberSpecies; i++)  // move each species
  {
    // #pragma omp task inout(part[i]) in(grid) target_device(booster)
    mem_avail = part[i].mover_PC_sub(grid, vct, EMf); // use the Predictor Corrector scheme 
  }
  // timeTasks.end(TimeTasks::PARTICLES);

  if (mem_avail < 0) {          // not enough memory space allocated for particles: stop the simulation
    if (myrank == 0) {
      cout << "*************************************************************" << endl;
      cout << "Simulation stopped. Not enough memory allocated for particles" << endl;
      cout << "*************************************************************" << endl;
    }
    return (true);              // exit from the time loop
  }

  /* -------------------------------------- */
  /* Repopulate the buffer zone at the edge */
  /* -------------------------------------- */

  InjectBoundaryParticles();

  if (mem_avail < 0) {          // not enough memory space allocated for particles: stop the simulation
    if (myrank == 0) {
      cout << "*************************************************************" << endl;
      cout << "Simulation stopped. Not enough memory allocated for particles" << endl;
      cout << "*************************************************************" << endl;
    }
    return (true);              // exit from the time loop
  }

  return (false);

}

void c_Solver::InjectBoundaryParticles(){

  for (int i=0; i < numberSpecies; i++) {
    if (col->getRHOinject(i)>0.0){

      mem_avail = part[i].particle_repopulator(grid,vct,EMf,i);

      /* --------------------------------------- */
      /* Remove particles from depopulation area */
      /* --------------------------------------- */

      if (col->getCase()=="Dipole") {
        for (int i=0; i < numberSpecies; i++)
          Qremoved[i] = part[i].deleteParticlesInsideSphere(col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());

      }
    }
  }

}

void c_Solver::WriteRestart(int cycle) {
  // write the RESTART file
  if (cycle % restart_cycle == 0 && cycle != first_cycle) {
    if (col->getWriteMethod() != "h5hut") {
      // without ,0 add to restart file
      writeRESTART(RestartDirName, myrank, cycle, numberSpecies, mpi, vct, col, grid, EMf, part, 0);
    }
  }

}

void c_Solver::WriteConserved(int cycle) {
  // write the conserved quantities
  if (cycle % col->getDiagnosticsOutputCycle() == 0) {
    Eenergy = EMf->getEenergy();
    Benergy = EMf->getBenergy();
    TOTenergy = 0.0;
    TOTmomentum = 0.0;
    for (int is = 0; is < numberSpecies; is++) {
      Ke[is] = part[is].getKe();
      TOTenergy += Ke[is];
      momentum[is] = part[is].getTotalP();
      TOTmomentum += momentum[is];
    }
    if (myrank == 0) {
      ofstream my_file(cq.c_str(), fstream::app);
      my_file << cycle << "\t" << "\t" << (Eenergy + Benergy + TOTenergy) << "\t" << TOTmomentum << "\t" << Eenergy << "\t" << Benergy << "\t" << TOTenergy << endl;
      my_file.close();
    }
    // // Velocity distribution
    // for (int is = 0; is < ns; is++) {
    //   double maxVel = part[is].getMaxVelocity();
    //   VelocityDist = part[is].getVelocityDistribution(nDistributionBins, maxVel);
    //   if (myrank == 0) {
    //     ofstream my_file(ds.c_str(), fstream::app);
    //     my_file << cycle << "\t" << is << "\t" << maxVel;
    //     for (int i = 0; i < nDistributionBins; i++)
    //       my_file << "\t" << VelocityDist[i];
    //     my_file << endl;
    //     my_file.close();
    //   }
    // }
  }
  
  //if (cycle%(col->getFieldOutputCycle())==0){
  //  for (int is = 0; is < ns; is++) {
  //    part[is].Add_vDist3D();
  //    part[is].Write_vDist3D(SaveDirName);
  //  }
  //}
}

void c_Solver::WriteOutput(int cycle) {

  if (col->getWriteMethod() == "h5hut") {

    /* -------------------------------------------- */
    /* Parallel HDF5 output using the H5hut library */
    /* -------------------------------------------- */

    if (cycle%(col->getFieldOutputCycle())==0)        WriteFieldsH5hut(numberSpecies, grid, EMf,  col, vct, cycle);
    if (cycle%(col->getParticlesOutputCycle())==0 &&
        cycle!=col->getLast_cycle() && cycle!=0)      WritePartclH5hut(numberSpecies, grid, part, col, vct, cycle);

  }
  else
  {

    // OUTPUT to large file, called proc**
    if (cycle % (col->getFieldOutputCycle()) == 0 || cycle == first_cycle) {
      hdf5_agent.open_append(SaveDirName + "/proc" + num_proc.str() + ".hdf");
      output_mgr.output("Eall + Ball + rhos + Jsall + pressure", cycle);
      // Pressure tensor is available
      hdf5_agent.close();
    }
    if (cycle % (col->getParticlesOutputCycle()) == 0 && col->getParticlesOutputCycle() != 1) {
      hdf5_agent.open_append(SaveDirName + "/proc" + num_proc.str() + ".hdf");
      output_mgr.output("position + velocity + q ", cycle, 1);
      hdf5_agent.close();
    }
    // write the virtual satellite traces

    if (numberSpecies > 2) {
      ofstream my_file(cqsat.c_str(), fstream::app);
      for (int isat = 0; isat < nsat; isat++) {
        for (int jsat = 0; jsat < nsat; jsat++) {
          for (int ksat = 0; ksat < nsat; ksat++) {
            int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
            int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
            int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
            my_file << EMf->getBx(index1, index2, index3) << "\t" << EMf->getBy(index1, index2, index3) << "\t" << EMf->getBz(index1, index2, index3) << "\t";
            my_file << EMf->getEx(index1, index2, index3) << "\t" << EMf->getEy(index1, index2, index3) << "\t" << EMf->getEz(index1, index2, index3) << "\t";
            my_file << EMf->getJxs(index1, index2, index3, 0) + EMf->getJxs(index1, index2, index3, 2) << "\t" << EMf->getJys(index1, index2, index3, 0) + EMf->getJys(index1, index2, index3, 2) << "\t" << EMf->getJzs(index1, index2, index3, 0) + EMf->getJzs(index1, index2, index3, 2) << "\t";
            my_file << EMf->getJxs(index1, index2, index3, 1) + EMf->getJxs(index1, index2, index3, 3) << "\t" << EMf->getJys(index1, index2, index3, 1) + EMf->getJys(index1, index2, index3, 3) << "\t" << EMf->getJzs(index1, index2, index3, 1) + EMf->getJzs(index1, index2, index3, 3) << "\t";
            my_file << EMf->getRHOns(index1, index2, index3, 0) + EMf->getRHOns(index1, index2, index3, 2) << "\t";
            my_file << EMf->getRHOns(index1, index2, index3, 1) + EMf->getRHOns(index1, index2, index3, 3) << "\t";
          }}}
      my_file << endl;
      my_file.close();
    }
  }
}

void c_Solver::WriteSimpleOutput(int cycle) {
    if(cycle == 0){
        FILE *Xfile = fopen((col->getSaveDirName() + "/Xfile.dat").c_str(), "w");
        FILE *Yfile = fopen((col->getSaveDirName() + "/Yfile.dat").c_str(), "w");
        FILE *Zfile = fopen((col->getSaveDirName() + "/Zfile.dat").c_str(), "w");
        fclose(Xfile);
        fclose(Yfile);
        fclose(Zfile);
        FILE* Efile = fopen((col->getSaveDirName() + "/Efield.dat").c_str(), "w");
        FILE* Bfile = fopen((col->getSaveDirName() + "/Bfield.dat").c_str(), "w");
        fclose(Efile);
        fclose(Bfile);
        FILE* concentrationsFile = fopen((col->getSaveDirName() + "/concentrations.dat").c_str(), "w");
        fclose(concentrationsFile);
        FILE* protonsFile = fopen((col->getSaveDirName() + "/protons.dat").c_str(), "w");
        FILE* electronsFile = fopen((col->getSaveDirName() + "/electrons.dat").c_str(), "w");
        FILE* positronsFile = fopen((col->getSaveDirName() + "/positrons.dat").c_str(), "w");
        FILE* alphasFile = fopen((col->getSaveDirName() + "/alphas.dat").c_str(), "w");
        fclose(protonsFile);
        fclose(electronsFile);
        fclose(positronsFile);
        fclose(alphasFile);
        FILE* velocityProtonFile = fopen((col->getSaveDirName() + "/velocity.dat").c_str(), "w");
        FILE* velocityElectronFile = fopen((col->getSaveDirName() + "/velocity_electron.dat").c_str(), "w");
        fclose(velocityProtonFile);
        fclose(velocityElectronFile);
        FILE* fluxFile = fopen((col->getSaveDirName() + "/flux.dat").c_str(), "w");
        fclose(fluxFile);
        FILE* trajectoryProtonFile = fopen((col->getSaveDirName() + "/trajectory_proton.dat").c_str(), "w");
        FILE* trajectoryElectronFile = fopen((col->getSaveDirName() + "/trajectory_electron.dat").c_str(), "w");
        fclose(trajectoryProtonFile);
        fclose(trajectoryElectronFile);
        FILE* distributionProtonFile = fopen((col->getSaveDirName() + "/distribution_protons.dat").c_str(), "w");
        FILE* distributionElectronFile = fopen((col->getSaveDirName() + "/distribution_electrons.dat").c_str(), "w");
        fclose(distributionProtonFile);
        fclose(distributionElectronFile);
        FILE* divergenceFile = fopen((col->getSaveDirName() + "/divergence_error.dat").c_str(), "w");
        fclose(divergenceFile);
        FILE* generalFile = fopen((col->getSaveDirName() + "/general.dat").c_str(),"w");
        fclose(generalFile);
    }
    if(cycle % 50 == 0) {
        FILE *Xfile = fopen((col->getSaveDirName() + "/Xfile.dat").c_str(), "w");
        FILE *Yfile = fopen((col->getSaveDirName() + "/Yfile.dat").c_str(), "w");
        FILE *Zfile = fopen((col->getSaveDirName() + "/Zfile.dat").c_str(), "w");
        for(int i = 1; i <= grid->getNXN() - 1; ++i){
            fprintf(Xfile, "%g\n", grid->getXN(i, 0, 0));
        }
        for(int i = 1; i <= grid->getNYN() - 1; ++i){
            fprintf(Yfile, "%g\n", grid->getYN(0, i, 0));
        }
        for(int i = 1; i <= grid->getNZN() - 1; ++i){
            fprintf(Zfile, "%g\n", grid->getZN(0, 0, i));
        }
        fclose(Xfile);
        fclose(Yfile);
        fclose(Zfile);

        FILE* Efile = fopen((col->getSaveDirName() + "/Efield.dat").c_str(), "a");
        FILE* Bfile = fopen((col->getSaveDirName() + "/Bfield.dat").c_str(), "a");
        FILE* fluxFile = fopen((col->getSaveDirName() + "/flux.dat").c_str(), "a");

        for(int i = 1; i <= grid->getNXN() - 1; ++i){
            for(int j = 1; j <= grid->getNYN() - 1; ++j){
                for(int k = 1; k <= grid->getNZN() - 1; ++k){
                    fprintf(Efile, "%g %g %g\n", EMf->getEx(i, j, k), EMf->getEy(i, j, k), EMf->getEz(i, j, k));
                    fprintf(fluxFile, "%g %g %g\n", EMf->getJx(i, j, k), EMf->getJy(i, j, k), EMf->getJz(i, j, k));
                }
            }
        }

        for(int i = 1; i < grid->getNXN() - 1; ++i){
            for(int j = 1; j < grid->getNYN() - 1; ++j){
                for(int k = 1; k < grid->getNZN() - 1; ++k){
                    fprintf(Bfile, "%g %g %g\n", EMf->getBx(i, j, k), EMf->getBy(i, j, k), EMf->getBz(i, j, k));
                }
            }
        }
        fclose(Efile);
        fclose(Bfile);
        fclose(fluxFile);

        FILE* concentrationsFile = fopen((col->getSaveDirName() + "/concentrations.dat").c_str(), "a");
        FILE* velocityProtonFile = fopen((col->getSaveDirName() + "/velocity.dat").c_str(), "a");
        FILE* velocityElectronFile = fopen((col->getSaveDirName() + "/velocity_electron.dat").c_str(), "a");
        for(int i = 1; i < grid->getNXN() - 1; ++i){
            for(int j = 1; j < grid->getNYN() - 1; ++j){
                for(int k = 1; k < grid->getNZN() - 1; ++k){
                    fprintf(concentrationsFile, "%g ", EMf->getRHOn(i, j, k));
                    for(int m = 0; m < numberSpecies; ++m){
                        fprintf(concentrationsFile, "%g ", EMf->getRHOns(i, j, k, m));
                    }
                    fprintf(concentrationsFile, "\n");
                    double electronChargeDensity = EMf->getRHOns(i, j, k, 0);
                    double protonChargeDensity = EMf->getRHOns(i, j, k, 1);
                    fprintf(velocityElectronFile, "%g %g %g\n", EMf->getJxs(i, j, k, 0)/electronChargeDensity, EMf->getJys(i, j, k, 0)/electronChargeDensity, EMf->getJzs(i, j, k, 0)/electronChargeDensity);
                    fprintf(velocityProtonFile, "%g %g %g\n", EMf->getJxs(i, j, k, 1)/protonChargeDensity, EMf->getJys(i, j, k, 1)/protonChargeDensity, EMf->getJzs(i, j, k, 1)/protonChargeDensity);
                }
            }
        }

        fclose(concentrationsFile);
        fclose(velocityProtonFile);
        fclose(velocityElectronFile);

        FILE* electronsFile = fopen((col->getSaveDirName() + "/electrons.dat").c_str(), "w");
        FILE* protonsFile = fopen((col->getSaveDirName() + "/protons.dat").c_str(), "w");

        for(int i = 0; i < part[0].getNOP(); ++i){
            fprintf(electronsFile, "%20.15g %20.15g %20.15g %g %g %g\n", part[0].getX(i), part[0].getY(i), part[0].getZ(i), part[0].getU(i), part[0].getV(i), part[0].getW(i));
        }

        for(int i = 0; i < part[1].getNOP(); ++i){
            fprintf(protonsFile, "%20.15g %20.15g %20.15g %g %g %g\n", part[1].getX(i), part[1].getY(i), part[1].getZ(i), part[1].getU(i), part[1].getV(i), part[1].getW(i));
        }

        fclose(electronsFile);
        fclose(protonsFile);

        FILE* generalFile = fopen((col->getSaveDirName() + "/general.dat").c_str(),"a");
        Eenergy = EMf->getEenergy();
        Benergy = EMf->getBenergy();
        TOTenergy = 0.0;
        TOTmomentum = 0.0;
        double momentumX = 0;
        double momentumY = 0;
        double momentumZ = 0;
        double kinEnergy = 0;
        int numberOfParticles = 0;
        double maxE = EMf->getMaxE();
        double maxB = EMf->getMaxB();
        for (int is = 0; is < numberSpecies; is++) {
            Ke[is] = part[is].getKe();
            TOTenergy += Ke[is];
            kinEnergy += Ke[is];
            momentum[is] = part[is].getTotalP();
            momentumX += part[is].getTotalPx();
            momentumY += part[is].getTotalPy();
            momentumZ += part[is].getTotalPz();
            TOTmomentum += momentum[is];
            numberOfParticles += part[is].getNOP();
        }
        double energy = TOTenergy + Eenergy + Benergy;
        double simulation_time = cycle*EMf->getDt();
        //todo momentum - vector and theoretical
        fprintf(generalFile, "%d %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %d\n",
                cycle, simulation_time, simulation_time, kinEnergy, Eenergy,
                Benergy, energy, momentumX, momentumY, momentumZ,
                energy, momentumX, momentumY, momentumZ, maxE, maxB, EMf->getDt(), numberOfParticles);
        //printf("%15.10g\n", EMf->getDt());
        fclose(generalFile);
        FILE* distributionProtonFile = fopen((col->getSaveDirName() + "/distribution_protons.dat").c_str(), "a");
        FILE* distributionElectronFile = fopen((col->getSaveDirName() + "/distribution_electrons.dat").c_str(), "a");
        WriteParticleDistribution(part[0], distributionElectronFile);
        WriteParticleDistribution(part[1], distributionProtonFile);
        fclose(distributionProtonFile);
        fclose(distributionElectronFile);
    }
}

void c_Solver::Finalize() {
  if (mem_avail == 0) {          // write the restart only if the simulation finished succesfully
    if (col->getWriteMethod() != "h5hut") {
      writeRESTART(RestartDirName, myrank, (col->getNcycles() + first_cycle) - 1, numberSpecies, mpi, vct, col, grid, EMf, part, 0);
    }
  }

  // stop profiling
  my_clock->stopTiming();

  // deallocate
  delete[]Ke;
  delete[]momentum;
  // close MPI
  mpi->finalize_mpi();
}

void c_Solver::WriteParticleDistribution(Particles3D& part, FILE *outputFile) {
    const int pnumber = 200;
    double minMomentum = 0; //todo something else
    double maxMomentum = 0;
    for (int i = 0; i < part.getNOP(); ++i) {
        double p = part.getP(i);
        if(minMomentum <= 0){
            minMomentum = p;
        }
        if (p  < minMomentum) {
            minMomentum = p;
        } else {
            if (p > maxMomentum) {
                maxMomentum = p;
            }
        }
    }

    double pgrid[pnumber + 1];
    double distribution[pnumber];
    double logMinMomentum = log(minMomentum);
    pgrid[0] = minMomentum;
    distribution[0] = 0;
    double deltaLogP = (log(maxMomentum) - log(minMomentum)) / (pnumber);
    for (int i = 1; i < pnumber; ++i) {
        distribution[i] = 0;
        pgrid[i] = exp(logMinMomentum + i * deltaLogP);
    }
    pgrid[pnumber] = maxMomentum;

    double weight = 0;

    for (int i = 0; i < part.getNOP(); ++i) {
        double p = part.getP(i);
        int j = (log(p) - logMinMomentum) / deltaLogP;
        if (j >= 0 && j < pnumber) {
            //distribution[j] += particles[i]->weight;
            //weight += particles[i]->weight;
            distribution[j] += 1;
            weight += 1;
        }
    }

    for (int i = 0; i < pnumber; ++i) {
        distribution[i] /= (weight * (pgrid[i + 1] - pgrid[i]));
    }

    for (int i = 0; i < pnumber; ++i) {
        fprintf(outputFile, "%20.15g %20.15g\n", (pgrid[i] + pgrid[i + 1]) / 2, distribution[i]);
    }

}




