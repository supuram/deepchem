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

qc = KS(system, xc=xc_b3lyp)
qc._engine.set_eigen_options(eigen_options={"method": "exacteig"})
scp = qc._engine.dm2scp(dm)   # contains the full matrix addition of H_core + V_xc + V_ext_field(if any) + J. Basically its the Fock Matrix
dm = qc._engine.scp2dm(scp)
print(qc.run(dm))
print("Final density matrix = ", qc.aodm())
print("SCF energy = ", qc.energy())
print(type(dm))
print(KSEngine.mro())