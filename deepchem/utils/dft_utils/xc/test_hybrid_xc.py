import pylibxc

print("Top-level attributes in pylibxc:")
print(dir(pylibxc))

print(dir(pylibxc.LibXCFunctional))
print("\n")

func = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp", "unpolarized")
print("B3LYP exact exchange a_x:", func.get_hyb_exx_coef())

print("\nAttributes and methods on the LibXCFunctional instance:")
print(func.describe())

# help(pylibxc.LibXCFunctional.compute)

available = pylibxc.util.xc_available_functional_names()
print("Total functionals found:", len(available))

# Filter specifically for hybrid functionals
hybrids = [name for name in available if "hyb" in name]
print("Hybrid functionals sample:", hybrids[:10])
a_x = float(func.get_hyb_exx_coef()) if hasattr(func, "get_hyb_exx_coef") else 0.0
print(a_x)