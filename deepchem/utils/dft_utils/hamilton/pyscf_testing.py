import numpy as np
from pyscf import gto, dft

# 1. Define identical geometry (in Angstroms)
mol = gto.M(
    atom="O 0.0 0.0 0.2217; H 0.0 1.4309 -0.8868; H 0.0 -1.4309 -0.8868",
    basis="sto-3g",
    unit="Bohr",
    charge=0,
    spin=0
)

# 2. Build standard DFT calculation with B3LYP
mf = dft.RKS(mol)
mf.xc = "hyb_gga_xc_b3lyp"

# 3. Compute 0-th iteration Fock / Vxc matrix from a core guess
dm0 = mf.get_init_guess(key="1e")  # Core Hamiltonian density matrix
vxc0 = mf.get_veff(dm=dm0)        # J + Vxc (includes 0.2 * K)

# 4. Full SCF energy
e_tot = mf.kernel()

print("=== PySCF Reference ===")
print("Total Converged Energy:", e_tot)
print("Initial Veff (J + Vxc) [0, 0]:", vxc0[0, 0])
e_tot_0 = mf.energy_tot(dm=dm0)
print("PySCF Initial Energy on dm0:", e_tot_0)