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
from deepchem.utils.dft_utils.qccalc.hf import HFEngine
from deepchem.utils.dft_utils.xc.libxc import LibXCGGA
from deepchem.utils.dft_utils.qccalc.ks import KS, KSEngine
torch.set_printoptions(linewidth=1000)

mol_spec = "O 0.0 0.0 0.2217; H 0.0 1.4309 -0.8868; H 0.0 -1.4309 -0.8868"
atomzs, atomposs = parse_moldesc(mol_spec)
xc_b3lyp = LibXCGGA("hyb_gga_xc_b3lyp")
basis_name = "sto-3g"
atombases = []
for z, pos in zip(atomzs, atomposs):
    # loadbasis pulls the exponents (alphas) and contraction coefficients
    bases = loadbasis("%d:%s" % (z, basis_name))
    atombases.append(AtomCGTOBasis(atomz=z, bases=bases, pos=pos))
system = Mol(moldesc=mol_spec, basis="sto-3g", spin=0, charge=0)
engine = HFEngine(system, restricted=True)
engine.set_eigen_options(eigen_options={"method": "exacteig"})
h_core_scp = engine._core1e_linop.fullmatrix()
dm = engine.scp2dm(h_core_scp)
hamilton = HamiltonCGTO(atombases=atombases, spherical=True, efield = None)
hamilton.build()
grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
grid = BeckeGrid(atomgrid, atomposs)
setup = hamilton.setup_grid(grid, xc=xc_b3lyp)
vxc_linop = hamilton.get_vxc(dm)

qc = KS(system, xc=xc_b3lyp)
qc._engine.set_eigen_options(eigen_options={"method": "exacteig"})
print("type(qc) = ", type(qc))
print("type(qc._engine) = ", type(qc._engine))
print("qc.get_system = ", qc.get_system)
print("\n")
print("qc.get_system() = ", qc.get_system())
print("\n")
print("qc.shape = ", qc._engine.shape) # even though _engine is inside scf_qccalc, the object stored inside _engine is a KSEngine. You can see this inside self._engine in scf_qccalc
print("qc.dtype = ", qc.dtype)
print("qc._polarized = ", qc._polarized)
print("qc._engine.polarized = ", qc._engine.polarized)
print("\n")
scp = qc._engine.dm2scp(dm)   # contains the full matrix addition of H_core + V_xc + V_ext_field(if any) + J. Basically its the Fock Matrix
print("qc._engine.dm2scp(dm) = \n", qc._engine.dm2scp(dm))  
print("qc._engine.scp2dm(scp) = \n", qc._engine.scp2dm(scp))  # calculates the new density matrix using the Fock Matrix
print("qc._engine.scp2scp(scp) = \n", qc._engine.scp2scp(scp))
print("qc._engine.dm2energy(dm) = ", qc._engine.dm2energy(dm))