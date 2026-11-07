# DOLFINx 0.11 API Reference Sheet

*AI-based Materials and Process Simulation I & II — paste this file into your LLM assistant as context.*

> **Instructions for the AI assistant**
> - Target **DOLFINx 0.11.x** (Basix 0.11, UFL 2026.1), real PETSc, Python, run under MPI in Google Colab via the `%%fenicsx` cell magic.
> - Use **only** the API patterns in this sheet. Do **not** use legacy FEniCS (`from fenics import *`, `dolfin`) or older DOLFINx APIs (see the table in Section 13).
> - All code must be MPI-parallel-safe: create meshes with `MPI.COMM_WORLD`, reduce global quantities with `comm.allreduce`, print only on `rank == 0`.
> - Derive nothing silently: state the weak form you implemented as a comment above the UFL code.

All snippets below were executed with DOLFINx 0.11.0 (serial and 2 MPI ranks) unless marked *not executed*.

---

## 0. Colab setup and cell magic

```python
# Cell 1 — once per runtime (see https://github.com/SeoulTechPSE/fenicsx-colab for the full Quick Start cell)
DOLFINX_VERSION = '0.11'
```

```text
%%fenicsx -np 2 --time      ← first line of every FEniCSx cell (runs the cell under mpirun)
```

## 1. Imports

```python
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

comm = MPI.COMM_WORLD
```

## 2. Meshes

```python
# Built-in meshes (comm is always the first argument)
msh = mesh.create_unit_square(comm, 32, 32, mesh.CellType.triangle)
msh_rect = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([2.2, 0.41])],
                                 [88, 16], mesh.CellType.quadrilateral)
msh_box = mesh.create_unit_cube(comm, 4, 4, 4, mesh.CellType.tetrahedron)

tdim = msh.topology.dim          # topological dimension
gdim = msh.geometry.dim          # geometric dimension
x_coords = msh.geometry.x        # shape (num_nodes, 3), always 3 columns
```

```python
# Gmsh → DOLFINx  (module is dolfinx.io.gmsh, NOT dolfinx.io.gmshio)   [not executed]
import gmsh
from dolfinx.io import gmsh as gmshio
gmsh.initialize()
if comm.rank == 0:
    gmsh.model.add("plate")
    # ... build geometry, add physical groups, gmsh.model.mesh.generate(2) ...
mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
msh_g, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
gmsh.finalize()
# Reading a .msh file: mesh_data = gmshio.read_from_msh("file.msh", comm, 0, gdim=2)
```

### 2a. Topology, geometry and owned/ghost entities

```python
# Entities of dimension d (edges/faces are created on demand)
msh.topology.create_entities(1)
n_edges = msh.topology.index_map(1).size_local        # OWNED on this rank; .num_ghosts, .size_global

# Adjacency Adj_{d1,d2}: create first, then read. Without create_connectivity → RuntimeError.
msh.topology.create_connectivity(2, 1)
c2e = msh.topology.connectivity(2, 1)                 # AdjacencyList: .links(i), .array, .offsets
c2e.links(0)                                          # global edges of cell 0; local edge i is opposite local vertex i
c2v = msh.topology.connectivity(2, 0).array.reshape(-1, 3)   # triangles: cell → vertex table (incl. ghost cells)

# Geometry: node coordinates and the nodes of each cell
gd = msh.geometry.dofmaps[0]                          # NumPy (n_cells_incl_ghosts, nodes_per_cell); geometry.dofmap is deprecated
P  = msh.geometry.x[gd[c], :gdim]                     # vertex coordinates of cell c — index gd with [c], NOT .links(c)
# det J of a DOLFINx cell can be negative (no counter-clockwise ordering): use |det J| for areas.

# Owned vs ghost cells: local cells 0..size_local-1 are owned, the rest are ghosts
n_own = msh.topology.index_map(tdim).size_local

# Boundary facets (owned exterior facets only); needs facet → cell connectivity
msh.topology.create_connectivity(tdim - 1, tdim)
bnd = mesh.exterior_facet_indices(msh.topology)
n_bnd = comm.allreduce(len(bnd), op=MPI.SUM)

# DOF map
V.dofmap.cell_dofs(c)                       # process-local DOF indices of cell c, in Basix local order
V.dofmap.list                               # (n_cells incl. ghosts, ndofs_per_cell) array
V.dofmap.dof_layout.entity_dofs(1, 0)       # local DOFs on local edge 0
V.tabulate_dof_coordinates()                # (n_local_dofs + ghosts, 3)
W0, maps = W.sub(0).collapse()              # maps is a LIST of arrays: W0 DOF i = W DOF maps[0][i]
```

## 3. Function spaces and elements

```python
V  = fem.functionspace(msh, ("Lagrange", 1))              # scalar P1
V2 = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))     # vector P2 (displacement, velocity)
T  = fem.functionspace(msh, ("Lagrange", 1, (gdim, gdim)))# tensor-valued
Q0 = fem.functionspace(msh, ("DG", 0))                    # piecewise constant (material data)

# Mixed space (Taylor–Hood P2–P1) via basix.ufl
P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
W  = fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))

# Quadrature space (internal variables: plastic strain, history variables)
Qe = basix.ufl.quadrature_element(msh.basix_cell(), value_shape=(), degree=2)
Qs = fem.functionspace(msh, Qe)

# DOF counts (global). For blocked (vector) spaces multiply by the block size!
ndofs = V2.dofmap.index_map.size_global * V2.dofmap.index_map_bs

# Lagrange variant (node placement). ("Lagrange", k) uses gll_warped for every k.
el = basix.ufl.element(basix.ElementFamily.P, msh.basix_cell(), 4, basix.LagrangeVariant.equispaced)
V4 = fem.functionspace(msh, el)
V4.ufl_element().lagrange_variant          # LagrangeVariant.equispaced
V4.element.basix_element                   # the underlying basix element
```

### 3a. Basix elements on the reference cell (Basix 0.11)

```python
import basix
e = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 3,
                         basix.LagrangeVariant.equispaced)   # degree >= 3: variant REQUIRED (RuntimeError otherwise)
e.dim; e.points; e.entity_dofs             # DOF order: vertices, then edges (edge i opposite vertex i), then interior
                                           # P2 interval: points = [0, 1, 0.5]
pts, wts = basix.make_quadrature(basix.CellType.triangle, 4)
tab = e.tabulate(1, pts)                   # shape (n_derivs, n_pts, n_dofs, value_size) = (3, 6, 10, 1)
phi, dphi_dx, dphi_dy = tab[0, :, :, 0], tab[1, :, :, 0], tab[2, :, :, 0]   # each (n_pts, n_dofs)
lat = basix.create_lattice(basix.CellType.triangle, 40, basix.LatticeType.equispaced, True)  # plotting points
e.coefficient_matrix                       # coefficients w.r.t. an ORTHONORMAL polynomial basis — NOT monomial V^{-1}
basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 3, pts)   # that basis
h = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.triangle, 3)
# Hermite DOFs: per vertex (value, d/dx, d/dy), then centroid value
```

## 4. Functions, constants, interpolation

```python
u_h = fem.Function(V, name="u")
u_h.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))   # x has shape (3, N)

v_vec = fem.Function(V2)
v_vec.interpolate(lambda x: np.vstack((x[1], -x[0])))                    # return shape (gdim, N)

k = fem.Constant(msh, default_scalar_type(1.0))      # change later with k.value = 2.0
f_vec = fem.Constant(msh, np.zeros(gdim, dtype=default_scalar_type))

# Interpolate a UFL expression (note: interpolation_points is a PROPERTY, no parentheses)
x = ufl.SpatialCoordinate(msh)
expr = fem.Expression(ufl.grad(u_h)[0] + x[0], V.element.interpolation_points)
w = fem.Function(V)
w.interpolate(expr)

# Raw DOF values (local + ghosts). After modifying them in parallel, update ghosts:
u_h.x.array[:] *= 2.0
u_h.x.scatter_forward()

# Callables must be vectorised: a constant needs np.full(x.shape[1], c), not `lambda x: 1.0` (TypeError).
# Coordinates on x = 0 can be -3e-18: use np.abs(x[0])**a for non-integer powers (otherwise nan).

# Interpolate on a subset of cells only (piecewise data, e.g. DG0 coefficients)
Q0 = fem.functionspace(msh, ("DG", 0))
kappa = fem.Function(Q0)
right = mesh.locate_entities(msh, tdim, lambda x: x[0] >= 0.5)
kappa.interpolate(lambda x: np.full(x.shape[1], 10.0), cells0=right)

# Between DIFFERENT meshes (u_from on msh_a, u_to on msh_b); plain interpolate → "Meshes do no match."
n_cells = msh_b.topology.index_map(tdim).size_local + msh_b.topology.index_map(tdim).num_ghosts
cells = np.arange(n_cells, dtype=np.int32)
data = fem.create_interpolation_data(u_to.function_space, u_from.function_space, cells, padding=1e-14)
u_to.interpolate_nonmatching(u_from, cells, interpolation_data=data)
```

## 5. Marking boundaries and integration measures

```python
fdim = tdim - 1
left   = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
right  = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
msh.topology.create_connectivity(fdim, tdim)
all_bnd = mesh.exterior_facet_indices(msh.topology)

# Facet tags: 1 = left, 2 = right
facets = np.hstack([left, right])
values = np.hstack([np.full_like(left, 1), np.full_like(right, 2)])
order = np.argsort(facets)
facet_tags = mesh.meshtags(msh, fdim, facets[order], values[order])

dx = ufl.Measure("dx", domain=msh)
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
# ds(1) → left boundary, ds(2) → right boundary
n = ufl.FacetNormal(msh)
h = ufl.CellDiameter(msh)
```

## 6. Dirichlet boundary conditions

```python
# (a) Scalar, constant value, topological (preferred)
dofs_L = fem.locate_dofs_topological(V, fdim, left)
bc_L = fem.dirichletbc(default_scalar_type(0.0), dofs_L, V)

# (b) Scalar, spatially varying value
g = fem.Function(V)
g.interpolate(lambda x: 1.0 + x[1]**2)
dofs_bnd = fem.locate_dofs_topological(V, fdim, all_bnd)
bc_g = fem.dirichletbc(g, dofs_bnd)                   # no V argument when value is a Function

# (c) Vector space, whole vector fixed
bc_u = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type),
                       fem.locate_dofs_topological(V2, fdim, left), V2)

# (d) One component only (e.g. u_x = 0 on the left)
dofs_x = fem.locate_dofs_topological(V2.sub(0), fdim, left)
bc_x = fem.dirichletbc(default_scalar_type(0.0), dofs_x, V2.sub(0))

# (e) Sub-space of a mixed space (velocity of Taylor–Hood) — needs the collapsed space
W0, _ = W.sub(0).collapse()
u_noslip = fem.Function(W0)                           # zero velocity
dofs_W = fem.locate_dofs_topological((W.sub(0), W0), fdim, all_bnd)
bc_W = fem.dirichletbc(u_noslip, dofs_W, W.sub(0))
```

## 7. Linear problems — high level

```python
# Weak form: find u in V, u = 0 on ∂Ω, such that (k ∇u, ∇v) = (f, v) for all v
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = 2 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L = f * v * dx
bc0 = fem.dirichletbc(default_scalar_type(0.0), dofs_bnd, V)

problem = LinearProblem(
    a, L, bcs=[bc0],
    petsc_options_prefix="poisson_",          # REQUIRED keyword in 0.10+ (unique per problem)
    petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps"},
)
uh = problem.solve()                           # returns a fem.Function

# Iterative alternative: {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}
```

## 8. Linear problems — manual assembly (time stepping, reuse of the matrix)

```python
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting,
                               set_bc, create_vector)

# Backward Euler heat equation:  (u - u_n)/dt - Δu = 0
dt = fem.Constant(msh, default_scalar_type(0.01))
u_n = fem.Function(V); u_n.interpolate(lambda x: np.exp(-50*((x[0]-0.5)**2 + (x[1]-0.5)**2)))
a_t = fem.form(u * v * dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
L_t = fem.form(u_n * v * dx)

A = assemble_matrix(a_t, bcs=[bc0]); A.assemble()
b = create_vector(V)                           # 0.11: create_vector takes the FunctionSpace

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A); ksp.setType("preonly"); ksp.getPC().setType("lu")

u_new = fem.Function(V)
for step in range(5):
    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L_t)
    apply_lifting(b, [a_t], bcs=[[bc0]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc0])
    ksp.solve(b, u_new.x.petsc_vec)
    u_new.x.scatter_forward()
    u_n.x.array[:] = u_new.x.array

ksp.destroy(); A.destroy(); b.destroy()        # free PETSc objects explicitly
```

## 9. Nonlinear problems — SNES (Newton)

```python
# Weak form: find u, u = 0 on ∂Ω:  ((1 + u^2) ∇u, ∇v) - (f, v) = 0
u_nl = fem.Function(V)                         # unknown is a Function, NOT a TrialFunction
v = ufl.TestFunction(V)
F = (1 + u_nl**2) * ufl.inner(ufl.grad(u_nl), ufl.grad(v)) * dx - f * v * dx
# Jacobian is derived automatically (ufl.derivative); pass J=... to override.

nl_problem = NonlinearProblem(
    F, u_nl, bcs=[bc0],
    petsc_options_prefix="nlpoisson_",
    petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                   "snes_rtol": 1e-10, "snes_atol": 1e-12,
                   "ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps"},
)
nl_problem.solve()
snes = nl_problem.solver
reason, its = snes.getConvergedReason(), snes.getIterationNumber()
assert reason > 0, f"SNES diverged (reason {reason})"   # always check convergence!
```

> `dolfinx.nls.petsc.NewtonSolver` still exists in 0.11 but is **deprecated** — use `NonlinearProblem` (SNES) above.

## 10. Mixed problems — Stokes (Taylor–Hood)

```python
(u_s, p_s) = ufl.TrialFunctions(W)
(v_s, q_s) = ufl.TestFunctions(W)
a_s = (ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) - p_s * ufl.div(v_s) - q_s * ufl.div(u_s)) * dx
L_s = ufl.inner(fem.Constant(msh, np.array([0.0, -1.0], dtype=default_scalar_type)), v_s) * dx

# Lid-driven cavity: u = (1, 0) on top, no-slip elsewhere
top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
walls = np.setdiff1d(all_bnd, top)              # all boundary facets except the lid
lid = fem.Function(W0); lid.interpolate(lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]))))
bc_lid = fem.dirichletbc(lid, fem.locate_dofs_topological((W.sub(0), W0), fdim, top), W.sub(0))
bc_wall = fem.dirichletbc(u_noslip, fem.locate_dofs_topological((W.sub(0), W0), fdim, walls), W.sub(0))

# Pressure is determined up to a constant: pin one pressure DOF
W1, _ = W.sub(1).collapse()
p_dofs = fem.locate_dofs_geometrical((W.sub(1), W1), lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
bc_p = fem.dirichletbc(fem.Function(W1), p_dofs, W.sub(1))

stokes = LinearProblem(a_s, L_s, bcs=[bc_lid, bc_wall, bc_p],
                       petsc_options_prefix="stokes_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                      "pc_factor_mat_solver_type": "mumps"})
w_h = stokes.solve()
u_sol = w_h.sub(0).collapse()                  # velocity as an independent Function on W0
p_sol = w_h.sub(1).collapse()
```

## 11. Errors and global scalars (parallel-safe)

```python
def global_scalar(form):
    return comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)

u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])     # use a UFL expression, not an interpolant
err_L2 = np.sqrt(global_scalar((uh - u_exact)**2 * dx))
err_H1 = np.sqrt(global_scalar(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * dx))
area_left = global_scalar(fem.Constant(msh, default_scalar_type(1.0)) * ds(1))
if comm.rank == 0:
    print(f"L2 error {err_L2:.3e}, H1 error {err_H1:.3e}, |left| = {area_left:.3f}")
```

## 12. Output

```python
# XDMF: mesh + P1 (or lower) fields; interpolate higher-order fields to P1 first
with io.XDMFFile(comm, "poisson.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh, 0.0)

# VTX (ADIOS2): arbitrary-order Lagrange fields, time series; open the .bp folder in ParaView
u_out = fem.Function(fem.functionspace(msh, ("Lagrange", 2, (gdim,))))
u_out.interpolate(u_sol)
with io.VTXWriter(comm, "velocity.bp", [u_out], engine="BP4") as vtx:
    vtx.write(0.0)
```

## 13. Submeshes (interfaces, conjugate heat transfer)

```python
# Submesh of the cells with x < 0.5
cells = mesh.locate_entities(msh, tdim, lambda x: x[0] <= 0.5 + 1e-12)
submsh, entity_map, vertex_map, geom_map = mesh.create_submesh(msh, tdim, cells)
V_sub = fem.functionspace(submsh, ("Lagrange", 1))
# Forms coupling parent and submesh functions need entity_maps=[entity_map]
# in fem.form(...), LinearProblem(...), NonlinearProblem(...) and fem.Expression(...).
```

---

## 14. Legacy / outdated API → DOLFINx 0.11

| Wrong (legacy FEniCS or older DOLFINx) | Correct in DOLFINx 0.11 |
|---|---|
| `from fenics import *`, `from dolfin import *` | `from dolfinx import fem, mesh, io` + `import ufl` |
| `UnitSquareMesh(32, 32)` | `mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)` |
| `FunctionSpace(mesh, "P", 1)` / `fem.FunctionSpace(...)` | `fem.functionspace(msh, ("Lagrange", 1))` |
| `VectorFunctionSpace(mesh, "P", 2)` | `fem.functionspace(msh, ("Lagrange", 2, (gdim,)))` |
| `ufl.FiniteElement`, `ufl.VectorElement`, `ufl.MixedElement` | `basix.ufl.element(...)`, `basix.ufl.mixed_element([...])` |
| `DirichletBC(V, g, boundary_fn)` | `fem.dirichletbc(value, fem.locate_dofs_topological(V, fdim, facets), V)` |
| `solve(a == L, u, bc)` | `LinearProblem(a, L, bcs=[...], petsc_options_prefix="...").solve()` |
| `LinearProblem(a, L, bcs)` without prefix (≤ 0.9) | `petsc_options_prefix=` is **required** |
| `NewtonSolver(comm, NonlinearProblem(F, u, bcs))` from `dolfinx.nls.petsc` | `fem.petsc.NonlinearProblem(F, u, bcs=[...], petsc_options_prefix=...).solve()` (SNES) |
| `u.vector`, `u.vector.ghostUpdate(...)` | `u.x.petsc_vec` (PETSc) / `u.x.array` (NumPy) + `u.x.scatter_forward()` |
| `V.element.interpolation_points()` | `V.element.interpolation_points` (property) |
| `dolfinx.io.gmshio.model_to_mesh(...)` returning a tuple | `dolfinx.io.gmsh.model_to_mesh(...)` returning `MeshData` (`.mesh`, `.cell_tags`, `.facet_tags`) |
| `create_vector(L_form)` | `create_vector(V)` (function space) |
| `dofmap.dof_layout.num_entity_dofs(dim)` (removed in 0.11) | `len(dofmap.dof_layout.entity_dofs(dim, i))` |
| `msh.geometry.dofmap` (deprecated in 0.11), `geometry.dofmap.links(c)` | `msh.geometry.dofmaps[0][c]` (NumPy array) |
| `Constant(mesh, 1.0)` with Python float | `fem.Constant(msh, default_scalar_type(1.0))` |
| `assemble_scalar(...)` used directly as a global value | `comm.allreduce(fem.assemble_scalar(...), op=MPI.SUM)` |
| `Expression("sin(x[0])", degree=2)` (C++ strings) | `ufl.sin(x[0])` with `x = ufl.SpatialCoordinate(msh)`, or `u.interpolate(lambda x: ...)` |
| `MeshFunction`, `MeshValueCollection` | `mesh.meshtags(msh, dim, entities, values)` |
| `File("u.pvd") << u` | `io.XDMFFile` / `io.VTXWriter` |
| `plot(u)` | PyVista (`dolfinx.plot.vtk_mesh`) or matplotlib on `u.x.array` |

## 15. Prompt template

```text
Context: [paste this reference sheet]
Problem: <strong form, domain, boundary conditions, material data>
Weak form (derived by me): <weak form in math>
Spaces: <element family/degree, mixed or not>
Task: Write a DOLFINx 0.11 script for the %%fenicsx cell magic that
  1. builds the mesh <...>,
  2. solves with <LinearProblem / NonlinearProblem (SNES) / manual assembly>,
  3. verifies the result by <MMS convergence rates / benchmark value / conservation check>,
  4. saves <fields> with VTXWriter.
Constraints: use only the API in the reference sheet; MPI-safe; comment each block
with the corresponding term of the weak form.
```
