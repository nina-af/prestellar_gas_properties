#!/usr/bin/env python3

import os
import glob
import numpy as np
import h5py
import pytreegrav as pg


class SimulationData:
    """
    Class for getting accreted gas property data for all sink 
    particles across all simulation snapshots.
    """
    
    def __init__(self, snapdir, bhdir, datadir, 
                 M0, R0, alpha0, G_code=4300.71, B_unit=1e4,   # Cloud properties.
                 fname_gas=None, fname_sink=None,
                 skiprows_s=20, skiprows_f=20,
                 sim_imin=0, sim_imax=489):
        self.snapdir    = snapdir     # Snapshot files found here.
        self.bhdir      = bhdir       # Input bhswallow, bhformation data here.
        self.datadir    = datadir     # Output sorted accretion data files here.
        self.skiprows_s = skiprows_s  # Rows to skip in bhswallow.dat
        self.skiprows_f = skiprows_f  # Rows to skip in bhformation.dat 
        self.sim_imin   = sim_imin    # Min. simulation snapshot.
        self.sim_imax   = sim_imax    # Max. simulation snapshot.
        self.M0         = M0          # Initial cloud mass [Msun].
        self.R0         = R0          # Initial cloud radius [pc].
        self.alpha0     = alpha0      # Initial turbulent alpha.
        self.G_code     = G_code      # Gravitational constant [code_units].
        self.B_unit     = B_unit      # Unit B in Gauss.
        
        # Get accretion dict data for this simulation.
        self.acc_dict = SinkAccretionHistory(bhdir, datadir, 
                                        fname_gas=fname_gas, fname_sink=fname_sink,
                                        skiprows_s=skiprows_s, skiprows_f=skiprows_f).accretion_dict
        
        # Get Cloud object for this simulation.
        self.cloud = Cloud(self.M0, self.R0, self.alpha0, G_code=self.G_code)
        
        # Track last snapshot for which accreted gas property data has
        # already been saved as an HDF5 file.
        self.first_unexamined = self.get_first_unexamined_snapshot()
        
        
    # Look for previously-generated accreted_gas_properties.hdf5 files and return next snapshot.
    def get_first_unexamined_snapshot(self, verbose=True):
        gasproperty_fnames = glob.glob(self.datadir + 'snapshot_*_accreted_gas_properties.hdf5')
        # Check if list is empty.
        if not gasproperty_fnames:
            if verbose:
                print('No accreted_gas_properties HDF5 files found; starting from snapshot {0:d}...'.format(self.sim_imin), flush=True)
            return self.sim_imin
        # Else, find largest snapshot number among filenames.
        else:
            snap_list = []
            for gp_fname in gasproperty_fnames:
                snap_list.append(int(gp_fname.split('/')[-1].split('_')[1]))
                
            if verbose:
                print('Found the following snapshot numbers in accreted_gas_properties HDF5 files:')
                print(snap_list)
            snap_arr = np.asarray(snap_list)
    
            # Mask any snapshot numbers outside of the range [imin, imax]
            mask_1 = (snap_arr >= self.sim_imin)
            mask_2 = (snap_arr <= self.sim_imax)
            mask   = np.logical_and(mask_1, mask_2)
            if verbose:
                print('The following snapshots numbers found are in the specified range:')
                print(snap_arr[mask])

            # Check for empty array:
            if not snap_arr[mask]:
                if verbose:
                    print('No snapshot numbers found in range; starting from snapshot {0:d}...'.format(self.sim_imin), flush=True)
                return self.sim_imin
                
            first_unexamined = np.max(snap_arr[mask]) + 1
            if verbose:
                print('Found accreted_gas_properties HDF5 files; starting at snapshot {0:d}...'.format(first_unexamined), flush=True)
            return first_unexamined
            
    # Get snapshot filename.
    def get_fname_snap(self, i):
        return os.path.join(self.snapdir, 'snapshot_{0:03d}.hdf5'.format(i))
        
    # Get and write accreted gas properties data for a single snapshot.
    def get_data_single_snap(self, i):
        fname_snap   = self.get_fname_snap(i)
        gas_props    = SnapshotGasProperties(fname_snap, self.cloud, self.datadir, B_unit=self.B_unit)
        all_gas_data = gas_props.get_all_gas_data(self.acc_dict, skip_potential=False, write_to_file=True, verbose=True)
        return all_gas_data
            
    # Loop over all snapshots in simulation and write accreted gas properties data
    # for each snapshot.
    def get_simulation_data(self, verbose=True):
        imin, imax = self.first_unexamined, self.sim_imax
        
        # Check if all snapshots in simulation range have already been examined.
        if (imin > imax):
            if verbose:
                print('ALL SNAPSHOTS IN RANGE [{0:d}-{1:d}] HAVE BEEN EXAMINED.'.format(self.sim_imin, self.sim_imax), flush=True)
            return
        
        # Otherwise, loop through unexamined snapshots in simulation range.
        if verbose:
            print('GETTING ACCRETED GAS PROPERTY DATA FOR ENTIRE SIMULATION...')
            print('SNAPSHOT RANGE: [{0:d}->{1:d}]'.format(imin, imax), flush=True)
        for i in range(imin, imax+1, 1):
            if verbose:
                print('SNAPSHOT {0:d}: GETTING GAS PROPERTY DATA'.format(i), flush=True)
            single_snap_data = self.get_data_single_snap(i)
            

class SinkAccretionHistory:

    """
    Class for getting particle IDs and accretion times of all
    gas particles accreted onto a sink particle for all sink
    particles formed in a STARFORGE simulation.
      - Need bhswallow.dat, bhformation.dat files.
      - Adapted from A. Kaalva.
    """
    
    # Initialize with path to blackhole_details directory (bhdir).
    def __init__(self, bhdir, datadir, fname_gas=None, fname_sink=None,
                 skiprows_s=20, skiprows_f=20):
        self.bhdir          = bhdir       # Input bhswallow, bhformation data here.
        self.datadir        = datadir     # Output sorted accretion data files here.
        self.skiprows_s     = skiprows_s  # Rows to skip in bhswallow.dat
        self.skiprows_f     = skiprows_f  # Rows to skip in bhformation.dat 
        self.accretion_dict = self.accretion_history_to_dict(fname_gas=fname_gas, 
                                                             fname_sink=fname_sink)
        
    def get_accretion_history(self, fname=None, save_txt=True, verbose=True):
        
        if verbose:
            print('Getting sink particle accretion history from bhswallow.dat...', flush=True)
    
        # Get sink particle IDs, accreted gas cell IDs, and accretion times from bhswallow.dat:
        fname_bhswallow = os.path.join(self.bhdir, 'bhswallow.dat')
        bhswallow_data  = np.loadtxt(fname_bhswallow, skiprows=self.skiprows_s)
        sink_ids        = bhswallow_data[:, 1].astype(np.int64)
        gas_ids         = bhswallow_data[:, 6].astype(np.int64)
        times           = bhswallow_data[:, 0]
        
        # Sort by sink ID to group accretion history for each sink particle.
        if verbose:
            print('Sorting data by sink ID...', flush=True)
        idx_sink_sort    = np.argsort(sink_ids)
        sink_ids_grouped = sink_ids[idx_sink_sort]
        gas_ids_grouped  = gas_ids[idx_sink_sort]
        times_grouped    = times[idx_sink_sort]
    
        # Get first occurrence of each unique sink ID in grouped list:
        unique_sinks, first_occurrence = np.unique(sink_ids_grouped, return_index=True)
    
        # Within each sink particle group, sort accreted gas particles by accretion time.
        if verbose:
            print('Sorting data by accretion time...', flush=True)
        idx_times_sort = []
        for i in range(len(first_occurrence) - 1):
            imin = first_occurrence[i]
            imax = first_occurrence[i+1]
            # Sort times[imin:imax]
            idx_t = np.argsort(times_grouped[imin:imax]) + imin
            idx_times_sort.extend(idx_t.tolist())
        idx_t = np.argsort(times_grouped[imax:]) + imax
        idx_times_sort.extend(idx_t.tolist())

        sink_ids_sort = sink_ids_grouped[idx_times_sort]
        gas_ids_sort  = gas_ids_grouped[idx_times_sort]
        times_sort    = times_grouped[idx_times_sort]
    
        # Stack into (N_gas_accreted, 3) array to save to text file.
        accretion_data = np.vstack((sink_ids_sort, gas_ids_sort, times_sort)).T

        # (Optionally) save as text file.
        if save_txt:
            if verbose:
                print('Saving as text file...', flush=True)
            # If no filename, save in datadir directory.
            if fname is None:
                fname = os.path.join(self.datadir, 'sink_accretion_data.txt')
            fmt = '%d', '%d', '%.8f'
            np.savetxt(fname, accretion_data, fmt=fmt)
        
        return accretion_data
    
    # Get sink particle properties at formation time.
    def get_sink_formation_details(self, fname=None, save_txt=True, verbose=True):
        
        if verbose:
            print('Getting sink particle properties at formation time from bhformation.dat...', flush=True)
        
        # Get sink particle properties at formation time from bhformation.dat:
        fname_bhformation = os.path.join(self.bhdir, 'bhformation.dat')
        bhformation_data  = np.loadtxt(fname_bhformation, skiprows=self.skiprows_f)
    
        # Data: sink IDs, formation time, mass, position, velocity.
        sink_ids = bhformation_data[:, 1].astype(np.int64)
        t = bhformation_data[:, 0]
        m = bhformation_data[:, 2]
        x = bhformation_data[:, 3]
        y = bhformation_data[:, 4]
        z = bhformation_data[:, 5]
        u = bhformation_data[:, 6]
        v = bhformation_data[:, 7]
        w = bhformation_data[:, 8]

        sink_formation_data = np.vstack((t, m, x, y, z, u, v, w)).T
    
        # (Optionally) save as text file.
        if save_txt:
            if verbose:
                print('Saving as text file...', flush=True)
            # If no filename, save in datadir directory.
            if fname is None:
                fname = os.path.join(self.datadir, 'sink_formation_data.txt')
            fmt   = '%d','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g'  
            np.savetxt(fname, np.vstack((sink_ids, t, m, x, y, z, u, v, w)).T, fmt=fmt)
            
        return sink_ids, sink_formation_data
    
    # Parse accretion_data, sink_formation_data to return dict containing
    # gas IDs, accretion times, formation properties for each sink particle.
    # dict keys: sink IDs (int_arr)
    # dict values:
    #   - accretion_dict[sink_ID] = {accreted_gas_ids (int_arr), accretion_times (float_arr),
    #                                (non-feedback) ids, (non-feedback) counts,
    #                                (feedback) ids, (feedback) counts,
    #                                formation_details}
    #   - accretion_dict[sink_ID]['formation_details] = sink [t, m, x, y, z, vx, vy, vz] at formation.
    def accretion_history_to_dict(self, fname_gas=None, fname_sink=None, verbose=True):
        
        if verbose:
            print('Converting accretion history to accretion dict...', flush=True)
        
        # Read accretion_data from stored .txt file.
        if fname_gas is not None:
            if verbose:
                print('Loading accretion history from text file...', flush=True)
            sink_ids  = np.loadtxt(fname_gas, dtype=int, usecols=0)
            gas_ids   = np.loadtxt(fname_gas, dtype=int, usecols=1)
            acc_times = np.loadtxt(fname_gas, dtype=float, usecols=2)
        else:
            if verbose:
                print('No accretion history text file; calling get_accretion_history()...', flush=True)
            accretion_data = self.get_accretion_history()
            sink_ids  = accretion_data[:, 0]
            gas_ids   = accretion_data[:, 1]
            acc_times = accretion_data[:, 2]
    
        unique_sinks, first_occurrence = np.unique(sink_ids, return_index=True)
    
        # Split gas IDs, accretion times into list of subarrays per sink particle.
        gas_ids_split   = np.split(gas_ids, first_occurrence[1:])
        acc_times_split = np.split(acc_times, first_occurrence[1:])
    
        accretion_dict = dict.fromkeys(unique_sinks.astype(np.int64))
        for i in range(len(unique_sinks)):

            sink_accretion_dict = {'all_gas_ids':[], 'all_acc_times':[],
                                   'non_fb_ids':[], 'non_fb_counts':[],
                                   'fb_ids':[], 'fb_counts':[],
                                   'formation_details':[]}

            # All accreted gas IDs for this sink particle.
            gas_ids_all, acc_times_all = gas_ids_split[i].astype(np.int64), acc_times_split[i]

            # Find unique gas IDs among list accreted gas IDs. Use to classify gas particles as
            # "feedback" (multiple occurrences of particle ID) vs. "non-feedback" (appears only
            # once in list of accreted gas IDs).
            u, c = np.unique(gas_ids_all, return_counts=True)
            mask = (c == 1)
            u_n, c_n = u[mask], c[mask]    # Particle IDs, counts of non-feedback particles.
            u_f, c_f = u[~mask], c[~mask]  # Particle IDs, counts of feedback particles.
            sink_accretion_dict['all_gas_ids']   = gas_ids_all
            sink_accretion_dict['all_acc_times'] = acc_times_all
            sink_accretion_dict['non_fb_ids']    = u_n
            sink_accretion_dict['non_fb_counts'] = c_n
            sink_accretion_dict['fb_ids']        = u_f
            sink_accretion_dict['fb_counts']     = c_f
            accretion_dict[unique_sinks[i]]      = sink_accretion_dict
        accretion_dict['sink_ids'] = unique_sinks.astype(np.int64)
    
        # Add sink particle properties at formation time to dict.
        if fname_sink is not None:
            if verbose:
                print('Loading sink properties at formation time from text file...', flush=True)
            s_ids  = np.loadtxt(fname_sink, dtype=int, usecols=0)
            s_data = np.loadtxt(fname_sink, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7, 8))
        else:
            if verbose:
                print('No sink porperties text file; calling get_sink_formation_details()...', flush=True)
            s_ids, s_data = self.get_sink_formation_details()
            
        for i, sink_id in enumerate(s_ids):
            if sink_id in accretion_dict:
                sink_accretion_dict = accretion_dict[sink_id]
                sink_accretion_dict['formation_details'] = s_data[i]
                accretion_dict[sink_id] = sink_accretion_dict
    
        return accretion_dict
    
class Cloud:
    """
    Class for calculating bulk cloud properties.
    Parameters:
        - M0: initial cloud mass [code_mass].
        - R0: initial cloud radius [code_length].
        - alpha0: initial cloud turbulent virial parameter.
        - G_code: gravitational constant in code units
        (default: 4300.17 in [pc * Msun^-1 * (m/s)^2]).
    """

    def __init__(self, M, R, alpha, G_code=4300.71, verbose=False):

        # Initial cloud mass, radius, and turbulent virial parameter.
        # G_code: gravitational constant in code units [default: 4300.71].
        self.M = M
        self.R = R
        self.L = (4.0 * np.pi * self.R**3 / 3.0)**(1.0/3.0)
        self.alpha  = alpha
        self.G_code = G_code

        self.rho     = self.get_initial_density(verbose=verbose)
        self.Sigma   = self.get_initial_surface_density(verbose=verbose)
        self.vrms    = self.get_initial_sigma_3D(verbose=verbose)
        self.t_cross = self.get_initial_t_cross(verbose=verbose)
        self.t_ff    = self.get_initial_t_ff(verbose=verbose)

    # ----------------------------- FUNCTIONS ---------------------------------

    # FROM INITIAL CLOUD PARAMETERS: surface density, R, vrms, Mach number.
    def get_initial_density(self, verbose=False):
        """
        Calculate the initial cloud density [code_mass/code_length**3].
        """
        rho = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        if verbose:
            print('Density: {0:.2f} Msun pc^-2'.format(rho))
        return rho


    def get_initial_surface_density(self, verbose=False):
        """
        Calculate the initial cloud surface density [code_mass/code_length**2].
        """
        Sigma = self.M / (np.pi * self.R**2)
        if verbose:
            print('Surface density: {0:.2f} Msun pc^-2'.format(Sigma))
        return Sigma

    # Initial 3D rms velocity.
    def get_initial_sigma_3D(self, verbose=False):
        """
        Calculate the initial 3D rms velocity [code_velocity].
        """
        sig_3D = np.sqrt((3.0 * self.alpha * self.G_code * self.M) / (5.0 * self.R))
        if verbose:
            print('sigma_3D = {0:.3f} m s^-1'.format(sig_3D))
        return sig_3D

    # Initial cloud trubulent crossing time.
    def get_initial_t_cross(self, verbose=False):
        """
        Calculate the initial turbulent crossing time as R/v_rms [code_time].
        """
        sig_3D  = self.get_initial_sigma_3D()
        t_cross = self.R / sig_3D
        if verbose:
            print('t_cross = {0:.3g} [code]'.format(t_cross))
        return t_cross

    # Initial cloud gravitational free-fall time.
    def get_initial_t_ff(self, verbose=False):
        """
        Calculate the initial freefall time [code_time].
        """
        rho  = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        t_ff = np.sqrt((3.0 * np.pi) / (32.0 * self.G_code * rho))
        if verbose:
            print('t_ff = {0:.3g} [code]'.format(t_ff))
        return t_ff


class SnapshotGasProperties:
    """
    Class for reading gas cell data from STARFORGE HDF5 snapshot files
    and computing gas properties.
    """

    def __init__(self, fname, cloud, datadir, B_unit=1e4):
        
        # Write accreted gas properties to datadir.
        self.datadir = datadir

        # Physical constants.
        self.PROTONMASS_CGS     = 1.6726e-24
        self.ELECTRONMASS_CGS   = 9.10953e-28
        self.BOLTZMANN_CGS      = 1.38066e-16
        self.HYDROGEN_MASSFRAC  = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS        = 2.9979e10

        # Initial cloud parameters.
        self.fname   = fname
        self.snapdir = self.get_snapdir()
        self.Cloud   = cloud
        self.M0      = cloud.M      # Initial cloud mass, radius.
        self.R0      = cloud.R
        self.L0      = cloud.L      # Volume-equivalent length.
        self.alpha0  = cloud.alpha  # Initial virial parameter.
        
        # PROBLEM - some feedback particles not captured by uniqueness
        # check in acc_dict[sink_ID]['non_fb_ids'] - need an additional
        # uniqueness check in list of matching particle IDs in snapshot.

        # Open HDF5 file.
        with h5py.File(fname, 'r') as f:
            header = f['Header']
            p0     = f['PartType0']   # Particle type 0 (gas).

            # Header attributes.
            self.box_size = header.attrs['BoxSize']
            self.num_p0   = header.attrs['NumPart_Total'][0]
            self.t        = header.attrs['Time']

            # Unit conversions to cgs; note typo in header for G_code.
            self.G_code      = header.attrs['Gravitational_Constant_In_Code_Inits']
            if 'Internal_UnitB_In_Gauss' in header.attrs:
                self.B_code = header.attrs['Internal_UnitB_In_Gauss']  # sqrt(4*pi*unit_pressure_cgs)
            else:
                self.B_code = 2.916731267922059e-09
            self.l_unit      = header.attrs['UnitLength_In_CGS']
            self.m_unit      = header.attrs['UnitMass_In_CGS']
            self.v_unit      = header.attrs['UnitVelocity_In_CGS']
            self.B_unit      = B_unit                           # Magnetic field unit in Gauss (default: 1e4).
            self.t_unit      = self.l_unit / self.v_unit
            self.t_unit_myr  = self.t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
            self.rho_unit    = self.m_unit / self.l_unit**3
            self.nH_unit     = self.rho_unit/self.PROTONMASS_CGS
            self.P_unit      = self.m_unit / self.l_unit / self.t_unit**2
            self.spec_L_unit = self.l_unit * self.v_unit        # Specific angular momentum.
            self.L_unit      = self.spec_L_unit * self.m_unit   # Angular momentum.
            self.E_unit      = self.l_unit**2 / self.t_unit**2  # Energy per mass.
            # Convert internal energy to temperature units.
            self.u_to_temp_units = (self.PROTONMASS_CGS/self.BOLTZMANN_CGS)*self.E_unit

            # Other useful conversion factors.
            self.cm_to_AU = 6.6845871226706e-14
            self.cm_to_pc = 3.2407792896664e-19

            # PartType0 data.
            self.p0_ids   = p0['ParticleIDs'][()]         # Particle IDs.
            self.p0_m     = p0['Masses'][()]              # Masses.
            self.p0_rho   = p0['Density'][()]             # Density.
            self.p0_hsml  = p0['SmoothingLength'][()]     # Particle smoothing length.
            self.p0_E_int = p0['InternalEnergy'][()]      # Internal energy.
            self.p0_P     = p0['Pressure'][()]            # Pressure.
            self.p0_cs    = p0['SoundSpeed'][()]          # Sound speed.
            self.p0_x     = p0['Coordinates'][()][:, 0]   # Coordinates.
            self.p0_y     = p0['Coordinates'][()][:, 1]
            self.p0_z     = p0['Coordinates'][()][:, 2]
            self.p0_u     = p0['Velocities'][()][:, 0]    # Velocities.
            self.p0_v     = p0['Velocities'][()][:, 1]
            self.p0_w     = p0['Velocities'][()][:, 2]
            self.p0_Ne    = p0['ElectronAbundance'][()]   # Electron abundance.
            if 'MagneticField' in p0.keys():              # Magnetic field.
                self.p0_Bx    = p0['MagneticField'][()][:, 0]
                self.p0_By    = p0['MagneticField'][()][:, 1]
                self.p0_Bz    = p0['MagneticField'][()][:, 2]
                self.p0_B_mag = np.sqrt(self.p0_Bx**2 + self.p0_By**2 + self.p0_Bz**2)
            else:
                self.p0_Bx    = np.zeros(len(self.p0_ids))
                self.p0_By    = np.zeros(len(self.p0_ids))
                self.p0_Bz    = np.zeros(len(self.p0_ids))
                self.p0_B_mag = np.zeros(len(self.p0_ids))

            # Hydrogen number density and total metallicity.
            self.p0_n_H  = (1.0 / self.PROTONMASS_CGS) * \
                       np.multiply(self.p0_rho * self.rho_unit, 1.0 - p0['Metallicity'][()][:, 0])
            self.p0_total_metallicity = p0['Metallicity'][()][:, 0]
            # Calculate mean molecular weight.
            self.p0_mean_molecular_weight = self.get_mean_molecular_weight(self.p0_ids)

            # Neutral hydrogen abundance, molecular mass fraction.
            self.p0_neutral_H_abundance = p0['NeutralHydrogenAbundance'][()]
            self.p0_molecular_mass_frac = p0['MolecularMassFraction'][()]

            # Calculate gas adiabatic index and temperature.
            fH, f, xe            = self.HYDROGEN_MASSFRAC, self.p0_molecular_mass_frac, self.p0_Ne
            f_mono, f_di         = fH*(xe + 1.-f) + (1.-fH)/4., fH*f/2.
            gamma_mono, gamma_di = 5./3., 7./5.
            gamma                = 1. + (f_mono + f_di) / (f_mono/(gamma_mono-1.) + f_di/(gamma_di-1.))
            self.p0_temperature  = (gamma - 1.) * self.p0_mean_molecular_weight * \
                                    self.u_to_temp_units * self.p0_E_int
            # Dust temperature.
            if 'Dust_Temperature' in p0.keys():
                self.p0_dust_temp = p0['Dust_Temperature'][()]

            # Simulation timestep.
            if 'TimeStep' in p0.keys():
                self.p0_timestep = p0['TimeStep'][()]

            # For convenience, stack coordinates/velocities/B_field in a (n_gas, 3) array.
            self.p0_coord = np.vstack((self.p0_x, self.p0_y, self.p0_z)).T
            self.p0_vel   = np.vstack((self.p0_u, self.p0_v, self.p0_w)).T
            self.p0_mag   = np.vstack((self.p0_Bx, self.p0_By, self.p0_Bz)).T

    # Try to get snapshot number from filename.
    def get_i(self):
        return int(self.fname.split('snapshot_')[1].split('.hdf5')[0])

    # Try to get snapshot datadir from filename.
    def get_snapdir(self):
        return self.fname.split('snapshot_')[0]

    # Calculate gas mean molecular weight.
    def get_mean_molecular_weight(self, gas_ids):
        idx_g                 = np.isin(self.p0_ids, gas_ids)
        T_eff_atomic          = 1.23 * (5.0/3.0-1.0) * self.u_to_temp_units * self.p0_E_int[idx_g]
        nH_cgs                = self.p0_rho[idx_g] * self.nH_unit
        T_transition          = self._DMIN(8000., nH_cgs)
        f_mol                 = 1./(1. + T_eff_atomic**2/T_transition**2)
        return 4. / (1. + (3. + 4.*self.p0_Ne[idx_g] - 2.*f_mol) * self.HYDROGEN_MASSFRAC)
    
    # Return only those particle IDs in gas_ids which occur only once in
    # list of all matching snapshot particle IDs.
    def get_unique_ids(self, gas_ids):
        mask_1       = np.isin(self.p0_ids, gas_ids)
        matching_ids = self.p0_ids[mask_1]
        u, c         = np.unique(matching_ids, return_counts=True)
        unique_ids   = u[c == 1]
        num_excluded = len(u) - len(unique_ids)
        return num_excluded, unique_ids

    # Get mask based on gas_ids (include only unique IDs).
    def get_mask(self, gas_ids, verbose=False):
        num_excluded, unique_ids = self.get_unique_ids(gas_ids)
        mask = np.isin(self.p0_ids, unique_ids)
        return mask

    # Check that mask based on gas_ids is non-empty.
    def check_gas_ids(self, gas_ids, verbose=False):
        mask_all = self.get_mask(gas_ids)
        if np.sum(mask_all) > 0:
            return True
        else:
            if verbose:
                print('No unique (i.e., non-feedback) gas ids.', flush=True)
            return False
            
    # Get indices of selected gas particles.
    def get_idx(self, gas_ids):
        num_excluded, unique_ids = self.get_unique_ids(gas_ids)
        if np.isscalar(unique_ids):
            idx_g = np.where(self.p0_ids == unique_ids)[0][0]
        else:
            idx_g = np.isin(self.p0_ids, unique_ids)
        return idx_g, num_excluded
        
    # Get center of mass for selected gas particles.
    def get_center_of_mass(self, m, pos, vel):
        M       = np.sum(m)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]

        x_cm = np.sum(np.multiply(m, x))/M; u_cm = np.sum(np.multiply(m, u))/M
        y_cm = np.sum(np.multiply(m, y))/M; v_cm = np.sum(np.multiply(m, v))/M
        z_cm = np.sum(np.multiply(m, z))/M; w_cm = np.sum(np.multiply(m, w))/M

        cm_x = np.asarray([x_cm, y_cm, z_cm])
        cm_v = np.asarray([u_cm, v_cm, w_cm])

        return M, cm_x, cm_v
        
    # Get gas kinematics relative to x, v vectors.
    def get_relative_kinematics(self, m, pos, vel, point_x, point_v):
        x0, y0, z0 = point_x[0], point_x[1], point_x[2]
        u0, v0, w0 = point_v[0], point_v[1], point_v[2]
        
        x1, y1, z1 = pos[:, 0], pos[:, 1], pos[:, 2]
        u1, v1, w1 = vel[:, 0], vel[:, 1], vel[:, 2]
        
        x, y, z = x1 - x0, y1 - y0, z1 - z0
        u, v, w = u1 - u0, v1 - v0, w1 - w0
        
        if np.isscalar(m):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T
            
    # Get total mass of remaining gas.
    def get_total_mass(self, m):
        return np.sum(m)
        
    # Get effective radius of selected gas particles.
    def get_effective_radius(self, m, rho):
        vol   = np.sum(m/rho)
        r     = ((3.0 * vol) / (4.0 * np.pi))**(1.0/3.0)
        return r
        
    # Get velocity dispersion of selected gas particles.
    def get_velocity_dispersion(self, m, vel):
        u, v, w  = vel[:, 0], vel[:, 1], vel[:, 2]
        sigma_3D = np.sqrt((self.weight_std(u, m)**2.0 + self.weight_std(v, m)**2.0 + \
                            self.weight_std(w, m)**2.0))
        return sigma_3D
        
    # Get angular momentum (with respect to center of mass) of selected gas particles.
    def get_angular_momentum(self, m, pos, vel):
        m_cm, x_cm, v_cm = self.get_center_of_mass(m, pos, vel)
        m_g, x_g, v_g    = self.get_relative_kinematics(m, pos, vel, x_cm, v_cm)
        ang_mom_vec      = np.sum(np.cross(x_g, np.multiply(np.reshape(m_g, (len(m_g), 1)), v_g)), axis=0)
        ang_mom_mag      = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag

    # Get specific angular momentum (with respect to center of mass) of selected gas particles.
    def get_specific_angular_momentum(self, m, pos, vel):
        if len(m) == 1:
            return np.asarray([0.0, 0.0, 0.0]), 0.0
        m_cm, x_cm, v_cm = self.get_center_of_mass(m, pos, vel)
        m_g, x_g, v_g    = self.get_relative_kinematics(m, pos, vel, x_cm, v_cm)
        ang_mom_vec      = np.sum(np.cross(x_g, v_g), axis=0)
        ang_mom_mag      = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag

    # Get gravitational potential energy (need pytreegrav module).
    def get_potential_energy(self, m, h, pos):
        E_pot   = 0.5 * np.sum(m * pg.Potential(pos, m, h, G=self.G_code))
        return E_pot
        
    # Get kinetic energy [code units].
    def get_kinetic_energy(self, m, vel):
        dv      = vel - np.average(vel, weights=m, axis=0)
        v_sqr   = np.sum(dv**2,axis=1)
        E_kin   = 0.5 * np.sum(m * v_sqr)
        return E_kin
        
    # Get magnetic energy [code units].
    def get_magnetic_energy(self, m, rho, B_mag):
        vol     = (m / rho) * self.l_unit**3
        E_mag   = (1.0/(8.0 * np.pi)) * np.sum(B_mag * B_mag * vol) / (self.E_unit * self.m_unit)
        return E_mag

    # Get internal energy [code units].
    def get_internal_energy(self, m, int_en):
        E_int = np.sum(m * int_en)
        return E_int

    # Get average (mass-weighted) temperature [K].
    def get_average_temperature(self, m, T_K):
        return self.weight_avg(T_K, m)

    # Get average (mass-weighted) magnetic field strength [gauss].
    def get_average_magnetic_field(self, m, B_mag):
        return self.weight_avg(B_mag, m)

    # TO-DO: get average ionization fraction. For now, just get
    # average number of electrons per H nucleon.
    def get_average_electron_abundance(self, m, elec):
        return self.weight_avg(elec, m)

    # Get aspect ratio of selected gas particles (prinicpal component analysis).
    def get_aspect_ratio_version_1(self, pos):
        dx    = pos - np.mean(pos, axis=0)
        R     = np.linalg.norm(dx, axis=1)
        ndim  = pos.shape[1]
        I     = np.eye(ndim) * np.mean(R**2)
        for i in range(ndim):
            for j in range(ndim):
                I[i,j] -= np.mean(dx[:,i] * dx[:,j])
        w, v = np.linalg.eig(I)
        R_p  = np.sqrt(w)
        #return np.max(R_p)/np.min(R_p), np.sum(R_p/R_p.max()), R_p
        return R_p  # array of principle components.
    
    # Alternate version, used in protostellar disk analysis scripts.
    def get_aspect_ratio_version_2(self, pos):
        centroid         = np.mean(pos, axis=0)
        centered_pos     = pos - centroid
        covar_matrix     = np.cov(centered_pos.T)
        eigvals, eigvecs = np.linalg.eig(covar_matrix)
        #q = np.min(np.sqrt(eigvals))/np.max(np.sqrt(eigvals))
        return eigvals, eigvecs

    # Get gas property data for a single set of accreted gas cells in a single snapshot.
    def get_gas_data(self, idx_g, num_feedback, num_feedback_new, skip_potential=False, verbose=True):
        data          = np.zeros(27)
        num_particles = np.sum(idx_g)
        if verbose:
            print('Analyzing {0:d} gas particles.'.format(num_particles), flush=True)
        
        # Speedup: get idx_g once, get m, v, etc. once, and pass as arguments
        # to get_X functions.
        m, h, rho = self.p0_m[idx_g], self.p0_hsml[idx_g], self.p0_rho[idx_g]
        x, y, z   = self.p0_x[idx_g], self.p0_y[idx_g], self.p0_z[idx_g]
        u, v, w   = self.p0_u[idx_g], self.p0_v[idx_g], self.p0_w[idx_g]
        pos, vel  = np.vstack((x, y, z)).T, np.vstack((u, v, w)).T
        B_mag     = self.p0_B_mag[idx_g] * self.B_unit
        T_K       = self.p0_temperature[idx_g]
        int_en    = self.p0_E_int[idx_g]
        elec      = self.p0_Ne[idx_g]
        
        # Compute gas properties.
        M_tot, x_cm, v_cm = self.get_center_of_mass(m, pos, vel)
        L_unit_vec, L_mag = self.get_specific_angular_momentum(m, pos, vel)
        L_vec  = L_mag * L_unit_vec
        R_eff  = self.get_effective_radius(m, rho)
        R_p    = self.get_aspect_ratio_version_1(pos)  # Returns array of principal components.
        Q1, Q2 = self.get_aspect_ratio_version_2(pos)  # Return eigenvalues, eignenvectors.
        T      = self.get_average_temperature(m, T_K)
        B      = self.get_average_magnetic_field(m, B_mag)
        Ne     = self.get_average_electron_abundance(m, elec)
        sig3D  = self.get_velocity_dispersion(m, vel)
        if skip_potential:
            if verbose:
                print('Skipping potential energy calculation...', flush=True)
            E_grav = 0.0
        else:
            #if verbose:
            #    print('Getting potential energy with pytreegrav...', flush=True)
            E_grav = self.get_potential_energy(m, h, pos)
            #if verbose:
            #    print('Done with potential energy calculation.', flush=True)
        E_kin  = self.get_kinetic_energy(m, vel)
        E_mag  = self.get_magnetic_energy(m, rho, B_mag)
        E_int  = self.get_internal_energy(m, int_en)
        N_inc  = np.sum(idx_g)
        N_fb   = num_feedback + num_feedback_new

        data[0]     = M_tot   # Total mass.
        data[1:4]   = x_cm    # Center of mass coordinates.
        data[4:7]   = v_cm    # Center of mass velocity.
        data[7:10]  = L_vec   # Specific angular momentum with respect to center of mass.
        data[10]    = R_eff   # Effective radius.
        data[11:14] = R_p     # Shape parameters (PCA).
        data[14:17] = Q1      # Eigenvalues of covariance matrix (PCA).
        data[17]    = T       # Average temperature.
        data[18]    = B       # Average magnetic field magnitude (may need to use volume).
        data[19]    = Ne      # Average number e- per H nucleon.
        data[20]    = sig3D   # Average velocity dispersion.
        data[21]    = E_grav  # Gravitational potential energy.
        data[22]    = E_kin   # Kinetic energy.
        data[23]    = E_mag   # Magnetic energy.
        data[24]    = E_int   # Internal energy (temperature proxy).
        data[25]    = N_inc   # Number of gas particles included in calculations.
        data[26]    = N_fb    # Number of gas particles excluded due to being (presumed) feedback particles.

        return data

    # Get gas property data for each set of gas_ids in accretion_dict.
    def get_all_gas_data(self, acc_dict, skip_potential=False, write_to_file=True, verbose=True):

        # Unique sink IDs:
        sink_ids  = acc_dict['sink_ids']
        num_sinks = len(sink_ids)

        if verbose:
            print('Getting data for {0:d} sink particles in this snapshot...'.format(num_sinks), flush=True)

        all_data = np.zeros((num_sinks, 28))  # Entry 0: sink ID.

        # Loop over unique sinks.
        count = 0
        for j, sink_id in enumerate(sink_ids):

            if verbose:
                progress_ratio = float(count)/float(num_sinks)
                print('[{0:d}/{1:d} = {2:.2f} percent complete]: getting data for sink ID {3:d}...'.format(count, num_sinks, progress_ratio*100.0, sink_id), flush=True)

            # Get particle IDs of accreted non-feedback gas particles.
            acc_gas_ids  = acc_dict[sink_id]['non_fb_ids']
            num_feedback = np.sum(acc_dict[sink_id]['fb_counts'])
            
            # Get idx of unique accreted non-feedback gas particles.
            idx_g, num_feedback_new = self.get_idx(acc_gas_ids)
            
            if np.sum(idx_g) == 0:
                #if verbose:
                #    print('No unique gas IDs found in snapshot; continuing to next sink ID...', flush=True)
                continue

            # Get gas properties.
            data            = self.get_gas_data(idx_g, num_feedback, num_feedback_new, 
                                                skip_potential=skip_potential)
            all_data[j, 0]  = sink_id  # Record sink ID.
            all_data[j, 1:] = data
            count += 1
            
        if write_to_file:
            if verbose:
                i = self.get_i()
                #print('Writing data for snapshot {0:d} to file...'.format(i))
            self.write_to_file(all_data)

        return all_data

    # Write gas properties data to HDF5 file.
    def write_to_file(self, all_data):

        sink_IDs = all_data[:, 0]
        M_tot    = all_data[:, 1]
        x_cm     = all_data[:, 2:5]
        v_cm     = all_data[:, 5:8]
        L_vec    = all_data[:, 8:11]
        R_eff    = all_data[:, 11]
        R_p      = all_data[:, 12:15]
        Q1       = all_data[:, 15:18]
        T        = all_data[:, 18]
        B        = all_data[:, 19]
        Ne       = all_data[:, 20]
        sig3D    = all_data[:, 21]
        E_grav   = all_data[:, 22]
        E_kin    = all_data[:, 23]
        E_mag    = all_data[:, 24]
        E_int    = all_data[:, 25]
        N_inc    = all_data[:, 26]
        N_fb     = all_data[:, 27]

        i = self.get_i()

        fname = os.path.join(self.datadir, 'snapshot_{0:03d}_accreted_gas_properties.hdf5'.format(i))

        f      = h5py.File(fname, 'w')
        header = f.create_dataset('header', (1,))
        header.attrs.create('time', self.t, dtype=float)
        header.attrs.create('m_unit', self.m_unit, dtype=float)
        header.attrs.create('l_unit', self.l_unit, dtype=float)
        header.attrs.create('v_unit', self.v_unit, dtype=float)
        header.attrs.create('t_unit', self.t_unit, dtype=float)
        header.attrs.create('B_unit', self.B_unit, dtype=float)

        # Sink IDs dataset.
        f.create_dataset('sink_IDs', data=sink_IDs, dtype=int)
        f.create_dataset('M_tot', data=M_tot)
        f.create_dataset('X_cm', data=x_cm)
        f.create_dataset('V_cm', data=v_cm)
        f.create_dataset('specific_angular_momentum', data=L_vec)
        f.create_dataset('effective_radius', data=R_eff)
        f.create_dataset('aspect_ratio_v1', data=R_p)
        f.create_dataset('aspect_ratio_v2', data=Q1)
        f.create_dataset('temperature', data=T)
        f.create_dataset('magnetic_field_strength', data=B)
        f.create_dataset('electron_abundance', data=Ne)
        f.create_dataset('velocity_dispersion', data=sig3D)
        f.create_dataset('potential_energy', data=E_grav)
        f.create_dataset('kinetic_energy', data=E_kin)
        f.create_dataset('magnetic_energy', data=E_mag)
        f.create_dataset('internal_energy', data=E_int)
        f.create_dataset('included_particle_number', data=N_inc, dtype=int)
        f.create_dataset('feedback_particle_number', data=N_fb, dtype=int)

        f.close()

    # Utility functions.
    def weight_avg(self, data, weights):
        "Weighted average"
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sum(data * weights) / weightsum
        else:
            return 0
    def weight_std(self, data, weights):
        "Weighted standard deviation."
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sqrt(np.sum(((data - self.weight_avg(data, weights))**2) * weights) / weightsum)
        else:
            return 0
    def _sigmoid_sqrt(self, x):
        return 0.5*(1 + x/np.sqrt(1+x*x))
    def _DMIN(self, a, b):
        return np.where(a < b, a, b)
    def _DMAX(self, a, b):
        return np.where(a > b, a, b)
