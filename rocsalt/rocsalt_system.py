"""
rocsalt_system.py
Open source toolkit for creating an OpenMM System_ and ParmEd Structure_ for a receptor and pair of compounds screened in ROCS (<https://www.eyesopen.com/rocs >)
.. _System : http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html.
.. _Structure : https://parmed.github.io/ParmEd/html/api/parmed/parmed.structure.html?highlight=structure#module-parmed.structu
"""

import os
import copy
import time
import yaml
import pickle
import pathlib
import argparse
import itertools
import progressbar
import ruamel.yaml

import mdtraj as md
import parmed as pmd

from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer

import random
import numpy as np
from math import floor

import yank
import openmmtools
import openmoltools
from openmmtools import forces, states
from openmoltools import utils, amber


class RocsaltSystem(object):
    """
    OpenMM System_ and ParmEd Structure_ for a Rocsalt simulation.
    .. _System : http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
    .. _Structure : https://parmed.github.io/ParmEd/html/api/parmed/parmed.structure.html?highlight=structure#module-parmed.structu

    Parameters
    ----------
        ligand_1_file : str
            Mol2 file of ligand 1.
        ligand_1_file : str
            Mol2 file of ligand 2.
         receptor_files : list of str
            Prmtop/inpcrd or pdb files of receptor.

    Properties
    ----------
        system : openmm.System 
        stucture : parmed.Structure
        ligands : list(Ligand)
        dummies_to_ions_idx : list of int
            Indices of dummy atoms that will be transformed into counterions.
        ions_to_dummies_idx : list of int
            Indices of ions that will be transformed into dummies.
        receptor_end_idx : list of int
            Indices of first and last atom of receptor.

    """
    def __init__(self, ligand_1_file, ligand_2_file, *receptor_files):
        # kwargs for creating a minimal periodic OpenMM system
        # rigidWater False is required for ParmEd to get water parameters
        kwargs = { 'nonbondedMethod' : app.PME, 'rigidWater' : False}
        # Determine the source of receptor file by the size of the list
        if len(receptor_files) == 1:
            system, topology, positions, total_charge = self._system_from_pdb(receptor_files[0], **kwargs)
        elif len(receptor_files) == 2:
            system, topology, positions, total_charge = self._system_from_amber(*receptor_files, **kwargs)
        # Raise an error if system is not neutral
        if total_charge != 0:
            raise(ValueError('Must provide a NEUTRAL receptor system.'))
        # create ParmEd structure containing the receptor 
        self.structure = pmd.openmm.load_topology(topology, system=system, xyz=positions)
        # I am testing with host-guest amber files so I need to re-center the receptor: tleap shifts coordinates after solvating
        center = self.structure.get_coordinates(frame=0).mean(0)
        self.structure.coordinates = self.structure.get_coordinates(frame=0) - center
        #
        structure_indices = md.Topology.from_openmm(self.structure.topology).select('all').tolist()
        # Determine which ions are present
        receptor_ions, receptor_ions_idx = self._ions_in_receptor(self.structure.topology)
        # Determine indexes of receptor atoms
        receptor_idx = list(set(structure_indices) - set(md.Topology.from_openmm(self.structure.topology).select('not water').tolist() + receptor_ions_idx))
        # first and last index in receptor_idx
        self.receptor_end_idx = [receptor_idx[0], receptor_idx[-1]]

        self.ligands = []
        for index, file in enumerate([ligand_1_file, ligand_2_file]):
            name, ext = utils.parse_ligand_filename(file)
            self._generate_amber_files(name, file)
            prm = pmd.amber.AmberParm('data/' + name + '.prmtop')
            total_charge = int(floor(0.5 + pmd.tools.netCharge(prm).execute()))
            resname = self._get_ligand_resname('data/' + name + '.prmtop')
            # add ligand to structure
            self.structure += pmd.load_file('data/' + name + '.prmtop', xyz='data/' + name + '.inpcrd')
            ligand_indices = list(set( md.Topology.from_openmm(self.structure.topology).select('all').tolist()) - set(structure_indices))
            structure_indices += ligand_indices
            self.ligands.append(Ligand(resname, total_charge, 'off', ligand_indices))

        # Determine counterions and dummy atoms required
        counter_counterions, ions_to_dummies, dummies_to_ions = self._ions_and_dummies_required(*self.ligands)

        kwargs = { 'nonbondedMethod' : app.PME, 'constraints' : app.HBonds, 'rigidWater' : True,
        'ewaldErrorTolerance' : 1.0e-4, 'removeCMMotion' : True, 'hydrogenMass' : 3.0*unit.amu }
        self.system = self.structure.createSystem(**kwargs)
        receptor_charge = [(ion_id, yank.pipeline.compute_net_charge(system, [ion_id]))
                          for ion_id in receptor_ions_idx]
        # Determine indexes of dummies and ions/dummies that will be transformed along the alchemical protocol
        dummies_idx = self._ions_subset(receptor_charge, counter_counterions)

        # dummies_to_ions_idx: subset of dummies_idx
        if (dummies_to_ions):
            self.dummies_to_ions_idx = dummies_idx[:abs(dummies_to_ions)]
            to_dummies_idx = list(set(dummies_idx) - set(self.dummies_to_ions_idx))
        else:
            self.dummies_to_ions_idx = None
            to_dummies_idx = dummies_idx        
        self.ions_to_dummies_idx = self._ions_subset(receptor_charge, ions_to_dummies)
        # Modify OpenMM system to transform ions into dummies
        for force in system.getForces():
            if force.__class__.__name__ == 'NonbondedForce':
                for index in to_dummies_idx:
                    force.setParticleParameters(index, 0.0, 1.0, 0.0)

        positions = self._anneal_ligand(receptor_idx) 
        pos_value = positions.value_in_unit(unit.angstroms)
        coords = np.array(list(pos_value), dtype=np.float64)

        self.structure.coordinates = coords.reshape((-1, len(self.structure.atoms), 3))

    def _generate_amber_files(self, ligand_name, file):
        """
        Generates the prmtop and inpcrd files for a ligand.
        Parameters
        ----------
            ligand_name : str
                The name of the ligand.
            file : str
                Mol2 file of the ligand.
        Returns
        -------
            prmtop_filename : str
                Amber prmtop file produced by tleap.
            inpcrd_filename : str
                Amber inpcrd file produced by tleap.
        """
        gaff_mol2_filename1, frcmod_filename1 = amber.run_antechamber('data/' + ligand_name, file, charge_method=None)
        return amber.run_tleap(ligand_name, gaff_mol2_filename1, frcmod_filename1, 'data/' + ligand_name + '.prmtop', 'data/' + ligand_name + '.inpcrd')

    def _fix_particle_sigmas(self, system):
        """
        Fix particles with zero LJ sigma
            Parameters
            ----------
                system : openmm.System
            Returns
            -------
                system : fixed openmm.System
        """
        for force in system.getForces():
            if force.__class__.__name__ == 'NonbondedForce':
                for index in range(system.getNumParticles()):
                    [charge, sigma, epsilon] = force.getParticleParameters(index)
                    if sigma / unit.nanometers == 0.0:
                        force.setParticleParameters(index, charge, 1.0*unit.angstroms, epsilon)
        return system

    def _get_ligand_resname(self, prmtop_filename):
        """
        Get the ligand resname from the Amber prmtop file
            Parameters
            ----------
       	       	prmtop_filename : str
                    Amber prmtop file.
            Returns
            -------
       	       	resname : str
                    resname of ligand.
        """
        with open(prmtop_filename) as f:
            _res_label = False
            for line in f:
                if (_res_label):
                    resname = str(next(f).split()[0])
                    break
                if line.startswith('%'+'FLAG RESIDUE_LABEL'):
                    _res_label = True
        return resname

    def _system_from_pdb(self, receptor_file, **kwargs):
        """
	Creates an OpenMM system from a pdb
            Parameters
            ----------
                receptor_file : str 
                    Receptor pdbfile.
            Returns
            -------
                system : openmm.System
                pdb.topology: openmm.Topology
                pdb.positions: list
                    List of atomic positions.
                total_charge: int
                    Total charge of the system.
                
        """
        pdb = app.PDBFile(receptor_file)
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = forcefield.createSystem(pdb.topology, **kwargs)
        for i in range(system.getNumForces()):
            if isinstance(system.getForce(i), NonbondedForce):
                nonbonded = system.getForce(i)
                break

        total_charge = 0.0
        for i in range(nonbonded.getNumParticles()):
            nb_i = nonbonded.getParticleParameters(i)
            total_charge += nb_i[0].value_in_unit(elementary_charge)
        total_charge = int(floor(0.5 + total_charge))

        return self._fix_particle_sigmas(system), pdb.topology, pdb.positions, total_charge

    def _system_from_amber(self, *receptor_file, **kwargs):
        """
        Creates an OpenMM system from the prmtop/inpcrd Amber files
            Parameters
            ----------
                receptor_file : list of str
                    List of filenames corresponding to the prmtop and inpcrd Amber files of the receptor.
            Returns
            -------
                system : openmm.System
                prmtop.topology: openmm.Topology
                inpcrd.positions: list
                    List of atomic positions.
                total_charge: int
                    Total charge of the system.

        """
        input = {}
        for file in receptor_file:
            name, ext = utils.parse_ligand_filename(file)
            input[ext] = file

        prmtop = app.amberprmtopfile.AmberPrmtopFile(input['.prmtop'])
        inpcrd = app.amberinpcrdfile.AmberInpcrdFile(input['.inpcrd'])
        prm = pmd.amber.AmberParm(input['.prmtop'])
        total_charge = int(floor(0.5 + pmd.tools.netCharge(prm).execute()))
        system = prmtop.createSystem(**kwargs)

        return self._fix_particle_sigmas(system), prmtop.topology, inpcrd.positions, total_charge

    def _ions_subset(self, ions_net_charges, counterions):
        """
	Finds minimal subset of ion indexes whose charge sums to counterions  
            Parameters
            ----------
                ions_net_charges : list of tuples
                    (index, charge) of ions in system.
                counterions : int
                    Total charge.
                
             Returns
             -------
                counterions_indices : list of int
                    Indices of ions.

        """

        for n_ions in range(1, len(ions_net_charges) + 1):
            for ion_subset in itertools.combinations(ions_net_charges, n_ions):
                counterions_indices, counterions_charges = zip(*ion_subset)
                if sum(counterions_charges) == counterions:
                    return(counterions_indices)

    def _ions_in_receptor(self, topology):
        """
        Determines resnames and indexes of ions present in the receptor system.
        The supported species are sodium, chlorine and potassium.
       
        Parameters
        ----------
            topology : openmm.Topology
                The topology corresponding to the receptor system.
        Returns
        -------
            ions : list os str
                Residue names of ions.
            ions_idx : list of int
                Indexes of ions.
        """

        ION_RESIDUE_NAMES = {'NA', 'CL', 'K'}
        ions = []
        ions_idx = []
        for res in topology.residues():
            if (('-' in res.name) or ('+' in res.name) or (res.name in ION_RESIDUE_NAMES)):
                ions.append(res.name)
                ions_idx += [atom.index for atom in res.atoms()]

        return ions, ions_idx

    def _ions_and_dummies_required(self, ligand_0, ligand_1):
        """
        Sets which ligand will have full interactions in the initial state of the alchemical protocol.
        Determines total charges to be zeroed after adding the ligands, along the alchemical
        protocol and to be created along the alchemical protocol. 
        Parameters
        ----------
            ligand_0 : (Ligand)
            ligand_1 : (Ligand)
        Returns
        -------
            counter_counterions : int
                Total charge to be zeroed after adding ligands.
            ions_to_dummies : int
                Total charge to be zeroed in the alchemical protocol. 
            dummies_to_ions : int
                Total charge to be created in the alchemical protocol.
        """

        if abs(ligand_0.charge) > abs(ligand_1.charge):
            num = ligand_0.charge
            ligand_0.state = 'on'
            den = ligand_1.charge
        else:
            num = ligand_1.charge
            ligand_1.state = 'on'
            den = ligand_0.charge

        ions_to_dummies = None
        dummies_to_ions = None

        counter_counterions = num
        if (num/den) >= 1:
            ions_to_dummies = den - num
        elif num == -den:
            ions_to_dummies = -num
            dummies_to_ions = num
        elif (num/den) < -1:
            ions_to_dummies = -num
            dummies_to_ions = den
        
        return counter_counterions, ions_to_dummies, dummies_to_ions

    def _alchemically_modify_ligand(self, reference_system):
        """
        Creates an alchemical system.
        Returns
        -------
	    alchemical_system : simtk.openmm.AlchemicalSystem
                An alchemically modified system. 
        """

        from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion, AlchemicalState
        factory = openmmtools.alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False, split_alchemical_forces = True)
        for ligand in self.ligands:
            if ligand.state == 'on':
                alchemical_region_zero = openmmtools.alchemy.AlchemicalRegion(alchemical_atoms=ligand.indices, name='zero')
            else:
                alchemical_region_one = openmmtools.alchemy.AlchemicalRegion(alchemical_atoms=ligand.indices, name='one')

        alchemical_system = factory.create_alchemical_system(reference_system, alchemical_regions = [alchemical_region_zero, alchemical_region_one])
        alchemical_state_zero = openmmtools.alchemy.AlchemicalState.from_system(alchemical_system, parameters_name_suffix = 'zero')
        alchemical_state_one = openmmtools.alchemy.AlchemicalState.from_system(alchemical_system, parameters_name_suffix = 'one')

        return alchemical_system

    def _anneal_ligand(self, receptor_idx):
        """
        Anneal ligand interactions to clean up clashes.
        Returns
        -------
            positions : unit.Quantity
                Positions of all atoms after annealing the ligand
        """

        reference_system = copy.deepcopy(self.system)
        guests_restraints = openmm.CustomCentroidBondForce(2, "(k/2)*distance(g1,g2)^2")
        guests_restraints.addGlobalParameter('k', 100.0*unit.kilocalories_per_mole/unit.angstrom**2)
        guests_restraints.addGroup(self.ligands[0].indices)
        guests_restraints.addGroup(self.ligands[1].indices)
        guests_restraints.addBond([0,1], [])
        reference_system.addForce(guests_restraints)
        flat_bottom = forces.FlatBottomRestraintForce(spring_constant=100.0*unit.kilocalories_per_mole/unit.angstrom**2,
                                              well_radius=10.0*unit.angstroms,
                                              restrained_atom_indices1=receptor_idx,
                                              restrained_atom_indices2=self.ligands[0].indices)
        reference_system.addForce(flat_bottom)

        alchemical_system = self._alchemically_modify_ligand(reference_system)

        from openmmtools.alchemy import AlchemicalState
        alchemical_state_zero = openmmtools.alchemy.AlchemicalState.from_system(alchemical_system, parameters_name_suffix = 'zero')
        alchemical_state_one = openmmtools.alchemy.AlchemicalState.from_system(alchemical_system, parameters_name_suffix = 'one')

        thermodynamic_state = states.ThermodynamicState(system=alchemical_system, temperature=300*unit.kelvin)
        
        composable_states = [alchemical_state_zero, alchemical_state_one]
        compound_states = states.CompoundThermodynamicState(thermodynamic_state, composable_states=composable_states)
        sampler_state = states.SamplerState(positions=self.structure.positions, box_vectors=self.structure.topology.getPeriodicBoxVectors())
        # Anneal
        n_annealing_steps = 1000
        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 90.0/unit.picoseconds, 1.0*unit.femtoseconds)
        context, integrator = openmmtools.cache.global_context_cache.get_context(compound_states, integrator)
        sampler_state.apply_to_context(context)
        compound_states.lambda_sterics_one = 0.0
        compound_states.lambda_electrostatics_one = 0.0
        compound_states.apply_to_context(context)
        print('Annealing sterics...')
        for step in progressbar.progressbar(range(n_annealing_steps)):
            compound_states.lambda_sterics_zero = float(step) / float(n_annealing_steps)
            compound_states.lambda_electrostatics_zero = 0.0
            compound_states.apply_to_context(context)
            integrator.step(1)
        print('Annealing electrostatics...')
        for step in progressbar.progressbar(range(n_annealing_steps)):
            compound_states.lambda_sterics_zero = 1.0
            compound_states.lambda_electrostatics_zero = float(step) / float(n_annealing_steps)
            compound_states.apply_to_context(context)
            integrator.step(1)
        sampler_state.update_from_context(context)

        # Compute the final energy of the system.
        final_energy = thermodynamic_state.reduced_potential(context)
        print('final alchemical energy {:8.3f}kT'.format(final_energy))

        return sampler_state.positions

class EditYaml(object):
    """
    Edits a Yank yaml configuration file for running a Rocsalt simulation.
    """
        
    def __init__(self, ligands, dummies_to_ions_idx, ions_to_dummies_idx, receptor_end_idx):
 
        file_name = 'rocsalt_system.yaml'
        config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(file_name))
        instances_system = config['systems']
        instances_experiment = config['experiment']

        for ligand in ligands:
            index = 'zero' if ligand.state == 'on' else 'one'
            instances_system['relative-system'][f'ligand_{index}'] = copy.deepcopy(ligand.indices)
            instances_experiment['ligands-restraint'][f'restrained_ligand_{index}_atoms'] = copy.deepcopy(ligand.indices)
        instances_system['relative-system']['ions_one'] = ions_to_dummies_idx
        instances_system['relative-system']['ions_zero'] = dummies_to_ions_idx
        instances_experiment['restraint']['restrained_receptor_atoms'] = '[index for index in range(receptor_end_idx[0], receptor_end_idx[1])]'
        instances_experiment['restraint']['restrained_ligand_atoms'] = copy.deepcopy(ligand.indices)       
        with open(file_name, 'w') as fp:
            ruamel.yaml.round_trip_dump(config, fp,  default_flow_style=True )

class Ligand(object):
    """
    Sets attributes to ligand

        Parameters
        ----------
            name : str
                Resname.
            charge : int
                Total charge.
            state : str
               State in the beginning of the alchemical protocol.
            indices : list of int
                Indices of atoms of the ligand in the rocsalt.structure

    """
    def __init__(self, name, charge, state, indices):
        self.name = name
        self.charge = charge
        self.indices = indices
        self.set_state(state)

    def set_state(self, state):
        self.state = state

def main():
    """Set up for a ROCSALT simulation.
    """

    parser = argparse.ArgumentParser(description='Compute relative affinities of compounds screened in ROCS')
    parser.add_argument('--ligands_filenames', dest='ligands_filenames', action='store',
                       nargs='*', help='ligands mol2 files')
    parser.add_argument('--receptor_filenames', dest='receptor_filenames', action='store',
                       nargs='*', help='receptor pdb file or prmtop/inpcrd amber files')

    args = parser.parse_args()

    # TODO: Check all required arguments have been provided

    # Determine the path to files input
    ligands_filenames = [os.path.abspath(i) for i in args.ligands_filenames]
    receptor_filenames = [os.path.abspath(i) for i in args.receptor_filenames]

    # Set up the OpenMM system and ParmEd structure
    rocsalt = RocsaltSystem(ligands_filenames[0], ligands_filenames[1], *receptor_filenames)
   
    # Serialize OpenMM system
    with open('rocsalt_system.xml', 'w') as f:
         f.write(XmlSerializer.serialize(rocsalt.system))

    # Write pdb
    rocsalt.structure.write_pdb('rocsalt_system.pdb')

    # Serialize ParmEd structure
    with open('rocsalt_structure.pickle', 'wb') as file:
        pickle.dump(rocsalt.structure, file)

    # Edit yaml file
    EditYaml(rocsalt.ligands, rocsalt.dummies_to_ions_idx, rocsalt.ions_to_dummies_idx, rocsalt.receptor_end_idx)

if __name__ == "__main__":

    main()
