import torch
from deepchem.utils.dft_utils.data.datastruct import AtomCGTOBasis
from deepchem.utils.dft_utils.hamilton.hcgto import HamiltonCGTO
from deepchem.utils.dft_utils.api.parser import parse_moldesc
from deepchem.utils.dft_utils.api.loadbasis import loadbasis
from deepchem.utils.dft_utils import overlap
from deepchem.utils.dft_utils.system.mol import Mol
from deepchem.utils.dft_utils.xc.libxc import LibXCLDA
from deepchem.utils.dft_utils.grid import multiatoms_grid
from deepchem.utils.dft_utils import LebedevGrid, BeckeGrid, RadialGrid
torch.set_printoptions(linewidth=1000)

# 1. Define geometry (element symbol and Cartesian coordinates in Bohr or Angstrom)
# Format: "Element x y z; Element x y z; ..."
mol_spec = "O 0.0 0.0 0.2217; H 0.0 1.4309 -0.8868; H 0.0 -1.4309 -0.8868"
atomzs, atomposs = parse_moldesc(mol_spec)
print("atomzs = ", atomzs)
print("atomposs = ", atomposs)

xc = LibXCLDA("lda_x")
print(xc.a_x)

E_z = torch.tensor([0.0, 0.0, 0.01], dtype=torch.double)
# Wrap it in a list (each element corresponds to index i in the loop)
efield = [E_z]

# 2. Load standard contracted Gaussian basis sets (e.g., STO-3G or 6-31G)
basis_name = "sto-3g"
atombases = []
for z, pos in zip(atomzs, atomposs):
    # loadbasis pulls the exponents (alphas) and contraction coefficients
    bases = loadbasis("%d:%s" % (z, basis_name))
    atombases.append(AtomCGTOBasis(atomz=z, bases=bases, pos=pos))
print("bases = ", bases)
print("atombases = ", atombases)
print("\n")

# 3. Pass the constructed atombases into HamiltonCGTO and build
hamilton = HamiltonCGTO(atombases=atombases, spherical=True, efield = efield)
print("build = ", hamilton.build())
print("\n")

# 4. Inspect the output matrices (all 7x7 for H2O in STO-3G)
print("Overlap matrix shape:", hamilton.olp_mat.shape)
print("Overlap matrix (orthonormalized -> Identity):\n", hamilton.olp_mat)
print("Core Hamiltonian (T + V_nuc) shape:", hamilton.kinnucl_mat)
print("_orthozer = ", hamilton._orthozer)
print("hamilton.libcint_wrapper = ", hamilton.libcint_wrapper)
s_raw = overlap(hamilton.libcint_wrapper)
transformed_s = hamilton._orthozer.convert2(s_raw)
print("X(T)SX = ", transformed_s)
print(hamilton.nao)
print(hamilton.df)
print("\n")

#5. Make the grid for H2O using multiatoms_grid.py
grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
grid = BeckeGrid(atomgrid, atomposs)
print("grid.get_rgrid().shape = ", grid.get_rgrid().shape)
print("grid.get_dvolume().shape = ", grid.get_dvolume().shape)
print("grid = ", grid)
setup = hamilton.setup_grid(grid, xc=xc)
print("setup = ", setup)
print("\n")

# 6. Calculate the density matrix using ao_orb2dm in hcgto.py
# To generate the coefficients C for the density matrix go to HFEngine in deepchem.utils.dft_utils.qccalc.hf
from deepchem.utils.dft_utils.qccalc.hf import HFEngine

# Mol defines the system (atoms, coordinates, charge, spin)
# It automatically loads basis functions, builds HamiltonCGTO, and sets occupation numbers
system = Mol(moldesc=mol_spec, basis="sto-3g", spin=0, charge=0)
print("system = ", system)

# HFEngine wraps the system and its Hamiltonian
engine = HFEngine(system, restricted=True)
print("engine = ", engine)
engine.set_eigen_options(eigen_options={"method": "exacteig"})
print("\n")

# Inspection: see the orb_weights the system assigned
print("System orbital weights:", engine._orb_weight)

# Initial Fock / SCP matrix (for the 0-th iteration, use core Hamiltonian Kinnucl)
h_core_scp = engine._core1e_linop.fullmatrix()
print("h_core_scp = \n", h_core_scp)
print("\n")

# scp2dm calls __fock2dm -> diagonalize -> ao_orb2dm internally
dm = engine.scp2dm(h_core_scp)
print("Density Matrix from HFEngine:\n", dm)
print("\n")

# 7. get_vxc function in hcgto.py
vxc_linop = hamilton.get_vxc(dm)
print("vxc_linop = \n", vxc_linop)
print("vxc_linop shape = ", vxc_linop.shape)
print("\n")

print("energy from the one-electron Hamiltonian = \n", hamilton.get_e_hcore(dm))
print("\n")

# 8. Fetch densinfo from _dm2densinfo
densinfo = hamilton._dm2densinfo(dm)

print("densinfo =", densinfo)
print("density =", densinfo.value)
print("density shape =", densinfo.value.shape)
print("\n")

# 9. Get J
elrep = hamilton.get_elrep(dm)

print("Coulomb = \n", elrep)
print("coulomb shape = ", elrep.shape)
print("\n")

e_xc = hamilton.get_e_xc(dm)
print("e_xc = ", e_xc)

# 10.
libxclda_getvxc = xc.get_vxc(densinfo)
print("\n")
print("potinfo = ", libxclda_getvxc)
print("shape of potinfo = ", libxclda_getvxc.value.shape)
print("\n")

# 11. get_edensityxc in libxc.py
libxclda_get_edensityxc = xc.get_edensityxc(densinfo)
print("libxclda_get_edensityxc = ", libxclda_get_edensityxc)
print("===========================================================================Ending LDA_X=============================================================================")

from deepchem.utils.dft_utils.xc.libxc import LibXCGGA

xc = LibXCGGA("gga_c_pbe")
setup = hamilton.setup_grid(grid, xc=xc)

vxc_linop = hamilton.get_vxc(dm)
print("vxc_linop = \n", vxc_linop)
print("vxc_linop shape = ", vxc_linop.shape)
print("\n")
print("======================================================Ending gga_c_pbe===============================================")
print("\n")
from deepchem.utils.dft_utils.xc.libxc import LibXCMGGA

xc = LibXCMGGA("mgga_x_scan")
print(xc.a_x)
setup = hamilton.setup_grid(grid, xc=xc)
vxc_linop = hamilton.get_vxc(dm)
print("vxc_linop = \n", vxc_linop)
print("vxc_linop shape = ", vxc_linop.shape)
print("\n")
print(LibXCMGGA.__mro__)
print("========================================================Ending mgga_x_scan===================================================")
xc_b3lyp = LibXCGGA("hyb_gga_xc_b3lyp")
print("B3LYP a_x =", xc_b3lyp.a_x)
setup = hamilton.setup_grid(grid, xc=xc)
vxc_linop = hamilton.get_vxc(dm)
print("vxc_linop = \n", vxc_linop)
print("vxc_linop shape = ", vxc_linop.shape)
e_xc = hamilton.get_e_xc(dm)
print("e_xc = ", e_xc)
e_core = hamilton.get_e_hcore(dm)
e_coul = hamilton.get_e_elrep(dm)
e_xc = hamilton.get_e_xc(dm)
e_nuc = system.get_nuclei_energy()

e_tot_deepchem = e_core + e_coul + e_xc + e_nuc

print("--- Energy Breakdown (DeepChem) ---")
print(f"E_core:    {e_core.item():.6f}")
print(f"E_coulomb: {e_coul.item():.6f}")
print(f"E_xc:      {e_xc.item():.6f}")
print(f"E_nuc:     {e_nuc.item():.6f}")
print("-----------------------------------")
print(f"Total E on initial dm: {e_tot_deepchem.item():.6f}")

# Run the SCF operation
