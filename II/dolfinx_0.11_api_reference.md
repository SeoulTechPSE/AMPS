# DOLFINx 0.11 API Reference Sheet (AMPS II)

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

## 16. Solid mechanics (Course II) — grows week by week

### 16a. Linear elasticity (small strain, isotropic)

```python
gdim = msh.geometry.dim
V = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))          # P2 vector space: P1 tetrahedra LOCK in bending
mu = E / (2 * (1 + nu)); lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
eps = lambda u: ufl.sym(ufl.grad(u))
sigma = lambda u: 2 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(gdim)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
L = ufl.dot(t, v) * ds(2)                                       # t = fem.Constant(msh, np.array([0, 0, -t0])) traction
bc = fem.dirichletbc(np.zeros(gdim), fem.locate_dofs_topological(V, fdim, left_facets), V)   # clamp: value is a numpy array
```

### 16b. Iterative solver: CG + algebraic multigrid with the rigid-body near-nullspace

```python
from dolfinx import la
def rigid_body_modes(V):                                        # 3D: 3 translations + 3 rotations
    x = V.tabulate_dof_coordinates()
    vecs = [la.vector(V.dofmap.index_map, V.dofmap.index_map_bs) for _ in range(6)]
    b = [w.array.reshape(-1, 3) for w in vecs]
    for i in range(3):
        b[i][:, i] = 1.0
    b[3][:, 0], b[3][:, 1] = -x[:, 1], x[:, 0]
    b[4][:, 0], b[4][:, 2] = x[:, 2], -x[:, 0]
    b[5][:, 2], b[5][:, 1] = x[:, 1], -x[:, 2]
    la.orthonormalize(vecs)
    return PETSc.NullSpace().create(vectors=[w.petsc_vec for w in vecs])

A = assemble_matrix(fem.form(a), bcs=[bc]); A.assemble()
A.setNearNullSpace(rigid_body_modes(V))                         # without it GAMG needs 3–4x more iterations, growing with h
ksp = PETSc.KSP().create(msh.comm); ksp.setOperators(A)
ksp.setType("cg"); ksp.getPC().setType("gamg"); ksp.setTolerances(rtol=1e-10)
# high level: LinearProblem(..., petsc_options={"ksp_type": "cg", "pc_type": "gamg"}) and then
# problem.A.setNearNullSpace(...) BEFORE problem.solve()
```

### 16c. Consistent reaction forces (residual of the UNconstrained system)

```python
A0 = assemble_matrix(fem.form(a)); A0.assemble()                # NO bcs: rows of the clamped DOFs are kept
f0 = assemble_vector(fem.form(L)); f0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
r = A0.createVecLeft(); A0.mult(uh.x.petsc_vec, r); r.axpy(-1.0, f0) # r = K u - f: nonzero only on constrained DOFs
dofs = fem.locate_dofs_topological(V, fdim, left_facets)
nloc = V.dofmap.index_map.size_local                             # blocked DOF d → entries 3d, 3d+1, 3d+2
Rz = msh.comm.allreduce(sum(r.array[3 * d + 2] for d in dofs if d < nloc), op=MPI.SUM)
# Integrating sigma(uh) n over the clamped face converges only slowly (stress singularity at the clamp) — prefer r.
```

### 16d. Hyperelasticity: energy → residual → Newton, load stepping

```python
u = fem.Function(V); v = ufl.TestFunction(V)
F = ufl.Identity(gdim) + ufl.grad(u); C = F.T * F; J = ufl.det(F)
psi = mu / 2 * (ufl.tr(C) - gdim) - mu * ufl.ln(J) + lmbda / 2 * ufl.ln(J) ** 2   # compressible neo-Hookean
t = fem.Constant(msh, np.zeros(gdim))                           # change t.value between load steps: no recompilation
Pi = psi * ufl.dx - ufl.dot(t, u) * ds(2)
R = ufl.derivative(Pi, u, v)                                    # residual; the Jacobian is ufl.derivative(R, u) (automatic)
problem = NonlinearProblem(R, u, bcs=[bc], petsc_options_prefix="hyper_",
                           petsc_options={"snes_type": "newtonls", "snes_rtol": 1e-10, "ksp_type": "preonly", "pc_type": "lu",
                                          "pc_factor_mat_solver_type": "mumps"})
for k in range(1, nsteps + 1):
    t.value = k / nsteps * t_final
    problem.solve()                                             # starts from the previous step's u
    assert problem.solver.getConvergedReason() > 0
# Reason -4 (DIVERGED_FUNCTION_NANORINF) in the first step: the first Newton update is the LINEAR solution; if it
# inverts elements (det F <= 0), ln J is NaN. Use more load steps.
```

### 16e. Plane stress, orthotropic material, Gmsh plate with a hole

```python
# plane stress (thin plate): the 2D law with lambda* instead of lambda
lam_s = 2 * lmbda * mu / (lmbda + 2 * mu)                       # = E nu / (1 - nu^2)
sigma = lambda u: 2 * mu * eps(u) + lam_s * ufl.tr(eps(u)) * ufl.Identity(2)

# orthotropic plane stress: compliance S as measured (Voigt, engineering shear 2 eps_12), C = S^-1 numerically,
# material axis 1 at angle theta from x: rotate the strain in, the stress out
S = np.array([[1 / E1, -nu12 / E1, 0], [-nu12 / E1, 1 / E2, 0], [0, 0, 1 / G12]])
C = ufl.as_matrix(np.linalg.inv(S).tolist())
Q = ufl.as_matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
def sigma_ortho(u):
    e = Q.T * eps(u) * Q
    s = C * ufl.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])
    return Q * ufl.as_matrix([[s[0], s[2]], [s[2], s[1]]]) * Q.T

# Gmsh: plate minus disk, built on rank 0; occ.cut removes object and tool (do not keep leftover surfaces)
gmsh.initialize()
if comm.rank == 0:
    s = gmsh.model.occ.addRectangle(0, 0, 0, L, L); d = gmsh.model.occ.addDisk(0, 0, 0, R, R)
    gmsh.model.occ.cut([(2, s)], [(2, d)]); gmsh.model.occ.synchronize()
    # addPhysicalGroup(2, surfaces, 1); addPhysicalGroup(1, curves, tag) per boundary part (sort curves by getCenterOfMass)
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)    # curved triangles on the hole
md = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2); msh, ft = md.mesh, md.facet_tags
gmsh.finalize()
area = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh))), op=MPI.SUM)   # check against the exact area

# symmetry plane x = 0 (roller): one component only
bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), 1, ft.find(LEFT)), V.sub(0))

# stress of the discrete solution, cell-wise, and its global maximum
W = fem.functionspace(msh, ("DG", 1, (2, 2)))                    # P2 displacement -> P1 stress per cell
s = fem.Function(W); s.interpolate(fem.Expression(sigma(uh), W.element.interpolation_points))
smax = comm.allreduce(s.x.array.reshape(-1, 4)[:, 0].max(), op=MPI.MAX)
```

### 16f. Axisymmetric elasticity in the (r, z) half-plane

```python
msh = mesh.create_rectangle(comm, [[ri, 0.0], [ro, h]], [nr, nz], cell_type=mesh.CellType.triangle)   # x[0] = r, x[1] = z
V = fem.functionspace(msh, ("Lagrange", 2, (2,)))               # u = (u_r, u_z)
r = ufl.SpatialCoordinate(msh)[0]
def eps_axi(u):                                                 # (rr, tt, zz) components of the 3D strain
    g = (u[0].dx(1) + u[1].dx(0)) / 2
    return ufl.as_matrix([[u[0].dx(0), 0, g], [0, u[0] / r, 0], [g, 0, u[1].dx(1)]])
sigma_axi = lambda u: 2 * mu * eps_axi(u) + lmbda * ufl.tr(eps_axi(u)) * ufl.Identity(3)
a = ufl.inner(sigma_axi(u), eps_axi(v)) * r * ufl.dx
L = ufl.dot(f, v) * r * ufl.dx + ufl.dot(t, v) * r * ds(tag)   # the factor 2 pi cancels in the equations
# totals are NOT per radian: a force is 2 pi * (integral with weight r), e.g. 2 pi * sum of consistent reactions (§16c)
# a fem.Constant for E, nu works (no JIT problem in 0.11)
```

### 16g. Generalized plane strain: one global unknown, static condensation, a curved mapped mesh

```python
# 3D law in the (x, y) cross-section; the reduction is the choice of eps_zz (0: plane strain, e0: GPS, from sigma_zz = 0: plane stress)
def eps3(u, ezz):
    g = (u[0].dx(1) + u[1].dx(0)) / 2
    return ufl.as_matrix([[u[0].dx(0), g, 0], [g, u[1].dx(1), 0], [0, 0, ezz]])
sigma3 = lambda u, ezz, dT: 2 * mu * eps3(u, ezz) + lmbda * ufl.tr(eps3(u, ezz)) * ufl.Identity(3) - beta * dT * ufl.Identity(3)
S2 = lambda S: ufl.as_matrix([[S[0, 0], S[0, 1]], [S[1, 0], S[1, 1]]])      # in-plane block for the virtual work
dx = ufl.Measure("dx", domain=msh)                     # bind measures to the mesh: a term that UFL simplifies to zero has no domain
zero = fem.Constant(msh, np.zeros(2)); lf, e0c = fem.Constant(msh, 1.0), fem.Constant(msh, 0.0)
a = fem.form(ufl.inner(S2(sigma3(u, 0.0, 0.0)), ufl.sym(ufl.grad(v))) * dx)
L = fem.form(ufl.dot(zero, v) * dx - ufl.inner(S2(sigma3(zero, e0c, lf * dT)), ufl.sym(ufl.grad(v))) * dx + lf * ufl.dot(t, v) * ds(tag))
# ufl.system / lhs / rhs fail on this form (list tensor mixing constants and arguments): write a and L explicitly
A = assemble_matrix(a, bcs=bcs); A.assemble()
ksp = PETSc.KSP().create(comm); ksp.setOperators(A); ksp.setType("preonly")
ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")        # one factorisation, reused by every ksp.solve
def solve(lf_, e0_):
    lf.value, e0c.value = lf_, e0_
    b = assemble_vector(L); apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE); set_bc(b, bcs)
    w = fem.Function(V); ksp.solve(b, w.x.petsc_vec); w.x.scatter_forward(); return w
Nz = lambda w, e0_, lf_: comm.allreduce(fem.assemble_scalar(fem.form(sigma3(w, e0_, lf_ * dT)[2, 2] * dx)), op=MPI.SUM)
u0, u1 = solve(1.0, 0.0), solve(0.0, 1.0)           # load case (e0 = 0) and unit case (e0 = 1, loads off: its load is lambda*e0 in sigma_xx, sigma_yy)
e0 = (N - Nz(u0, 0.0, 1.0)) / Nz(u1, 1.0, 0.0)       # Schur complement: N1 = c - g^T K^-1 g  (= E*|Omega| for a laterally free section)
uh = fem.Function(V); uh.x.array[:] = u0.x.array + e0 * u1.x.array
# single vertices as supports (pin + roller = statically determinate in 2D)
vt = mesh.locate_entities_boundary(msh, 0, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V.sub(0), 0, vt), V.sub(0))
# curved P2 geometry without Gmsh: map a structured grid, 6-node triangles in basix order [v0, v1, v2, mid(v1,v2), mid(v0,v2), mid(v0,v1)]
msh = mesh.create_mesh(comm, cells, ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))), x)   # rank 0: arrays, others: empty
# piecewise material: DG0 functions for mu, lmbda, beta (cells chosen by their midpoints, mesh.compute_midpoints)
```

### 16h. Thermoelasticity: thermal stress, one-way coupling, full coupling in one mixed space

```python
beta = (3 * lmbda + 2 * mu) * alpha                   # = E alpha / (1 - 2 nu); sigma = 2 mu eps + lambda tr(eps) I - beta * dT * I (3D, plane strain)
beta_ps = E * alpha / (1 - nu)                        # plane stress (eps_zz eliminated with sigma_zz = 0), together with lambda* = 2 lambda mu / (lambda + 2 mu)
# one-way: steady heat first, then the temperature Function as a load in the mechanics
Th = LinearProblem(k * ufl.inner(ufl.grad(T), ufl.grad(s)) * dx, Q * s * dx, bcs=bcs_T, petsc_options_prefix="heat_").solve()
L_mech = beta * Th * ufl.div(v) * dx                  # (+ tractions); the P1 temperature enters a P2 form directly
# full coupling (linearised about T0, theta = T - T0), implicit Euler, ONE mixed space
el = basix.ufl.mixed_element([basix.ufl.element("Lagrange", "triangle", 1), basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))])
W = fem.functionspace(msh, el)
(th, u), (s, v) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
w_n = fem.Function(W); th_n, u_n = ufl.split(w_n)
a = (rho_c / dt * th * s + T0 * beta / dt * ufl.tr(eps(u)) * s + k * ufl.inner(ufl.grad(th), ufl.grad(s))) * dx \
    + ufl.inner(2 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(2) - beta * th * ufl.Identity(2), eps(v)) * dx
L = (rho_c / dt * th_n * s + T0 * beta / dt * ufl.tr(eps(u_n)) * s) * dx
bc_T = fem.dirichletbc(fem.Constant(msh, dT0), fem.locate_dofs_topological(W.sub(0), 1, facets), W.sub(0))
bc_ux = fem.dirichletbc(0.0, fem.locate_dofs_topological(W.sub(1).sub(0), 1, facets), W.sub(1).sub(0))
# the matrix is constant in time: assemble + factorise once (KSP preonly/LU), per step only assemble b, solve, w_n.x.array[:] = w.x.array
# the matrix is NOT symmetric (LU / GMRES, not CG)
theta_h = w.sub(0).collapse(); u_h = w.sub(1).collapse()       # sub-functions with their own spaces (arrays match tabulate_dof_coordinates)
thL = fem.assemble_scalar(fem.form(w.sub(0) * ds(1)))          # boundary mean through a form (+ allreduce)
delta = T0 * beta**2 / (rho_c * (lmbda + 2 * mu))              # coupling number; 1D uniaxial strain: diffusivity kappa / (1 + delta)
```

### 16i. J2 plasticity: quadrature spaces, the return mapping in NumPy, an explicit Newton loop

```python
# internal variables live at quadrature points (degree 2 on triangles = 3 points per cell); Mandel vectors (xx, yy, zz, sqrt2 xy)
cell = msh.basix_cell()
Q4  = fem.functionspace(msh, basix.ufl.quadrature_element(cell, value_shape=(4,), degree=2))      # sigma, strain increment
Q44 = fem.functionspace(msh, basix.ufl.quadrature_element(cell, value_shape=(4, 4), degree=2))    # algorithmic tangent
Q1  = fem.functionspace(msh, basix.ufl.quadrature_element(cell, value_shape=(), degree=2))        # equivalent plastic strain p
sig, sig_n, Ct, p, p_n = fem.Function(Q4), fem.Function(Q4), fem.Function(Q44), fem.Function(Q1), fem.Function(Q1)
eps4 = lambda u: ufl.as_vector([ufl.sym(ufl.grad(u))[0, 0], ufl.sym(ufl.grad(u))[1, 1], 0, np.sqrt(2) * ufl.sym(ufl.grad(u))[0, 1]])  # plane strain
deps_expr = fem.Expression(eps4(Du), Q4.element.interpolation_points)     # Du = increment since the last CONVERGED step
deps.interpolate(deps_expr); X = deps.x.array.reshape(-1, 4)             # (n_q, 4) NumPy array -> return mapping on all points at once
# radial return (J2, linear isotropic hardening):  s_tr = dev(sig_n + C @ deps), f = sqrt(3/2)|s_tr| - (sig0 + H p_n)
#   dp = f / (3 mu + H) where f > 0;  sig = sig_tr - 2 mu sqrt(3/2) dp n,  n = s_tr / |s_tr|  (UNIT normal)
#   C_alg = C - 2 mu beta Idev - 2 mu (3 mu/(3 mu + H) - beta) n n^T,  beta = 3 mu dp / sigma_eq_tr   (elastic points: C)
F = fem.form(ufl.dot(sig, eps4(v)) * dx + pload * ufl.dot(n, v) * ds(INNER))   # sig, Ct are coefficients: UFL cannot differentiate the update
J = fem.form(ufl.dot(ufl.dot(Ct, eps4(du)), eps4(v)) * dx)
# a quadrature-function form is integrated with the element's own points (a quadrature_degree in metadata is ignored)
# Newton: Du = 0; loop { update -> residual -> check -> assemble J with bcs -> solve -> Du -= du }; commit u, sig_n, p_n ONLY after convergence
assemble_vector(b, F)
apply_lifting(b, [J], bcs=[bcs], x0=[Du.x.petsc_vec], alpha=-1.0)     # prescribed displacement increment enters the FIRST correction
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE); set_bc(b, bcs, Du.x.petsc_vec, -1.0)
# a non-converged step is not an exception: return a flag and stop (pressure control above the limit load has no solution)
# 1D radial model of a tube: eps4 = (u', u/r, 0, 0) with weight r: ufl.dot(sig, eps4(v)) * x[0] * dx
```

### 16j. Linear viscoelasticity: Prony series, branch strains at quadrature points, one linear solve per step

```python
# E(t) = E_inf + sum_i E_i exp(-t/tau_i); branch i: d(eps_v)/dt = (eps - eps_v)/tau_i; 3D: C_i = E_i * C_hat(nu) (Mandel 4 x 4, as 16i)
# one step:  eps_v_new = a eps_v + (1 - b) eps_new + (b - a) eps_old
#   BE: a = b = tau/(tau + dt);   EXP (exact for strain linear in time): a = exp(-dt/tau), b = (1 - a) tau/dt;   dt = 0 (a jump): a = b = 1 (elastic)
Ceff.x.array[:] = np.tile((Cinf + sum(b_i * C_i)).ravel(), nq)            # Q44 quadrature Function; constant for a fixed dt -> factorise once
hist.x.array[:] = sum((a_i * ev_i + (b_i - a_i) * e_old) @ C_i).ravel()   # Q4 quadrature Function: the history stress
a = fem.form(ufl.dot(ufl.dot(Ceff, eps4(du)), eps4(v)) * dx)              # linear problem, no Newton
L = fem.form(ufl.dot(hist, eps4(v)) * dx - p * ufl.dot(n, v) * ds(INNER))
e_new.interpolate(fem.Expression(eps4(u), Q4.element.interpolation_points))   # after the solve, then update ev_i and e_old in NumPy
# exact checks: relaxation sigma = eps0 E(t); creep eps = sigma0 J(t), J from Laplace: J(s) = 1/(s^2 E(s)) (residues);
# same nu in every branch: u(t) = u_elastic(E = 1) * sigma0 J(t) (correspondence principle)
```

### 16k. Phase-field fracture (AT2): history field, staggered (alternate minimisation) loop, a slit in the mesh

```python
g = (1 - d)**2 + k                                                         # k ~ 1e-6
a_u = fem.form(g * ufl.inner(sigma(du), eps(v)) * dx)                      # u-problem with d fixed ("hybrid": isotropic degradation)
a_d = fem.form((Gc / ell + 2 * H) * dd * q * dx + Gc * ell * ufl.inner(ufl.grad(dd), ufl.grad(q)) * dx)
L_d = fem.form(2 * H * q * dx)                                             # H: scalar quadrature Function (degree 2)
psi_expr = fem.Expression(psi_plus(u), Q.element.interpolation_points)     # Amor: K/2 <tr eps>+^2 + mu e:e (plane strain: 3D deviator)
for it in range(1, maxit + 1):                                             # one load step
    d_old = d.x.array.copy()
    solve(a_u, u); psi.interpolate(psi_expr); H.x.array[:] = np.maximum(H_n, psi.x.array); solve(a_d, d)
    if np.abs(d.x.array - d_old).max() < tol: break                        # (allreduce MAX in parallel)
H_n = H.x.array.copy()                                                     # return the convergence flag and the iteration count
F = fem.assemble_scalar(fem.form(g * sigma(u)[1, 1] * dx))                 # consistent reaction for u_y = U on top of a unit-height body
# checks: 1D profile d(0) = 1 -> energy Gc/2 tanh(L/ell); homogeneous bar sigma_max = 3 sqrt(3)/16 sqrt(E Gc/ell) at d = 1/4
# put the notch in the MESH (duplicate the nodes on the slit and give the cells below the slit the copies; mesh.create_mesh) -- not as an
# initial value of d (the linear d-problem forgets it) and not as one line of d = 1 nodes; h <= ell/2 along the crack path
```

### 16l. Structural dynamics: mass matrix, modes (SLEPc), Newmark with PETSc vectors

```python
K = assemble_matrix(fem.form(ufl.inner(sigma(u), eps(v)) * dx), bcs=[bc], diag=1.0); K.assemble()
M = assemble_matrix(fem.form(rho * ufl.inner(u, v) * dx), bcs=[bc], diag=0.0); M.assemble()      # BC rows 0 -> BC "modes" at infinity
eps_ = SLEPc.EPS().create(comm); eps_.setOperators(K, M); eps_.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps_.setDimensions(6); eps_.setTarget(0.0); eps_.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
st = eps_.getST(); st.setType("sinvert"); st.getKSP().setType("preonly"); st.getKSP().getPC().setType("lu")
eps_.solve(); x = K.createVecRight(); lam = eps_.getEigenpair(i, x).real                         # omega = sqrt(lam); classify modes by SHAPE
# Newmark (beta, gamma): A = M1 + gamma dt C + beta dt^2 K  (M1: mass with diag=1 on BC rows; C = aM M + aK K), factorise once
# step: up = u + dt v + (1/2 - beta) dt^2 a;  vp = v + (1 - gamma) dt a;  solve A a = f - K up - C vp (BC rows 0);  u = up + beta dt^2 a;  v = vp + gamma dt a
# a0 from M a0 = f - K u0 (a load applied at t = 0 has a0 != 0). (1/4, 1/2): unconditionally stable, energy conserving; central difference: omega_max dt <= 2
# Rayleigh: zeta_i = aM/(2 omega_i) + aK omega_i/2  ->  aM = zeta omega_1, aK = zeta/omega_1 gives zeta at omega_1 (not 2 zeta omega_1)
Kd = PETSc.Mat(); K.convert("dense", Kd)                 # inspect a matrix: convert into a NEW Mat -- K.convert("dense") converts K in place
uh = obj.static(); val = g.dot(uh.x.petsc_vec)           # keep the Function alive: g.dot(obj.static().x.petsc_vec) can segfault
```

### 16m. Linear buckling: pre-buckling stress, geometric stiffness, (-K_G) phi = (1/lambda) K phi

```python
u0 = solve(K, f_ref)                                     # pre-buckling under a reference load (BCs as usual)
s0 = sigma(u0)
mKG = assemble_matrix(fem.form(-ufl.inner(s0, ufl.dot(ufl.grad(u), ufl.grad(v).T)) * dx), bcs=[bc], diag=0.0); mKG.assemble()   # sigma0_ij u_k,i v_k,j
e = SLEPc.EPS().create(comm); e.setOperators(mKG, K); e.setProblemType(SLEPc.EPS.ProblemType.GHEP)   # B = K (positive definite)
e.setDimensions(6); e.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)                                  # theta = 1/lambda
e.getST().getKSP().setType("preonly"); e.getST().getKSP().getPC().setType("lu"); e.solve()
lam = 1 / e.getEigenpair(i, x).real                      # P_cr = lam * P_ref; match modes to Euler by deflection direction and order
# P1 hexahedra lock in bending (+71 %): use P2. Higher modes fall below Euler (shear) -- physics, not error
```

### 16n. Trusses and 3D Timoshenko frames on a 1D mesh (no Gmsh)

```python
if comm.rank == 0: x, cells = np.array(points), np.array(pairs, dtype=np.int64)          # joints + inner points, (i, j) per cell
else:              x, cells = np.zeros((0, gdim)), np.zeros((0, 2), dtype=np.int64)
msh = mesh.create_mesh(comm, cells, ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(gdim,))), x)
msh.topology.create_connectivity(0, 1)                    # needed by locate_dofs_topological on vertices
J = ufl.Jacobian(msh); t = ufl.as_vector([J[i, 0] for i in range(gdim)]); t = t / ufl.sqrt(ufl.dot(t, t))   # sign = cell orientation
eps_ax = lambda w: ufl.dot(ufl.dot(ufl.grad(w), t), t)    # truss: axial strain; grad on a manifold = tangential gradient
v0 = mesh.locate_entities(msh, 0, lambda p: np.isclose(p[0], x0) & np.isclose(p[1], y0))
dofs = fem.locate_dofs_topological(V.sub(c), 0, v0)       # joint DOFs (locate_dofs_geometrical refuses sub-spaces)
b.array[dofs[dofs < V.dofmap.index_map.size_local * bs]] += Fc   # point load at a joint (owned DOFs only)
# Timoshenko: V = mixed (u, theta); a1 = cross(up, t)/|.|, a2 = cross(t, a1); strains t.u', a1.u' - a2.theta, a2.u' + a1.theta,
#   a1.theta', a2.theta', t.theta'.  P1/P1 LOCKS (EI_eff = EI + kGA l^2/12): shear terms with
dxs = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 0})     # one-point rule, or u in P2
# - one cell per TRUSS member (inner nodes have no transverse stiffness: reason -11)
# - 'up' must not be parallel to any member (NaN, reason -11); one per member as a DG0 vector field if needed
# - 3 or more cells at a vertex: run SERIALLY -- in parallel such vertices are duplicated (Warren truss: 13 -> 17-21 vertices)
```

### 16o. Geometrically exact rod: rotation vector, energy, SNES

```python
w = fem.Function(V); u, th = ufl.split(w)                 # V = (P2 vector) x (P1 vector) on the x axis in 3D
x = ufl.sqrt(ufl.dot(th, th) + 1e-12); S = skew(th); I = ufl.Identity(3)
Lam = I + ufl.sin(x) / x * S + (1 - ufl.cos(x)) / x**2 * S * S                    # exp(S), Rodrigues
T = I + (1 - ufl.cos(x)) / x**2 * S + (x - ufl.sin(x)) / x**3 * S * S             # tangent operator
Gam = Lam.T * (t0 + u.dx(0)) - t0;  Kap = T.T * th.dx(0)                            # material strains
dxN = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2})              # reduced rule for the force term (P1/P1: 1)
Pi = 0.5 * ufl.dot(Gam, CN * Gam) * dxN + 0.5 * ufl.dot(Kap, CM * Kap) * dx - ufl.dot(F, u) * ds(tip) - ufl.dot(M, th) * ds(tip)
problem = NonlinearProblem(ufl.derivative(Pi, w, ufl.TestFunction(V)), w, bcs=bcs, petsc_options_prefix=f"rod{k}_",   # a NEW prefix per problem
                           petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "basic",               # bt stalls (reason -5)
                                          "snes_divergence_tolerance": 1e12,      # stiff rods: residual x 1e6 after the 1st update; -1 is IGNORED
                                          "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
for s in steps: F.value[:] = s * F_max; problem.solve(); assert problem.solver.getConvergedReason() > 0     # TOTAL loads
# Tests: end moment -> circle (exact); EI = GJ, dead end moment m -> theta = s m/EI exactly (helix); tip force -> elastica (+ F L/ES if ES = GS)
# Mixed-space BCs: dirichletbc(Function(V.sub(i).collapse()[0]), locate_dofs_topological((V.sub(i), Vc), 0, root), V.sub(i))
#   (a numpy constant of size 3 on V.sub(i) of a mixed space is refused)
```

### 16p. Reissner–Mindlin plates: mixed (w, theta), reduced integration of the shear term, point evaluation

```python
c = msh.basix_cell()                                       # quadrilateral mesh: mesh.create_unit_square(..., mesh.CellType.quadrilateral)
V = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Q", c, deg), basix.ufl.element("Q", c, deg, shape=(2,))]))  # "S" = serendipity
(w, th), (v, ps) = ufl.TrialFunctions(V), ufl.TestFunctions(V)
D, F = E * h**3 / (12 * (1 - nu**2)), 5 / 6 * E / (2 * (1 + nu)) * h
chi = lambda t: ufl.sym(ufl.grad(t)); M = lambda t: D * ((1 - nu) * chi(t) + nu * ufl.tr(chi(t)) * ufl.Identity(2))
dxs = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2 * deg - 2})        # SRI on the SHEAR term only (Q1: one point)
a = ufl.inner(M(th), chi(ps)) * ufl.dx + F * ufl.dot(ufl.grad(w) - th, ufl.grad(v) - ps) * dxs
# clamped: w = 0 and theta = 0; hard simple support: w = 0 and the TANGENTIAL rotation = 0; soft: w = 0 only
# point value (any element; serendipity DOFs are not all point values, tabulate_dof_coordinates() raises):
from dolfinx import geometry
p = np.array([[x0, y0, 0.0]])
cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(geometry.bb_tree(msh, 2), p), p)
own = len(cells.links(0)) and cells.links(0)[0] < msh.topology.index_map(2).size_local
w0 = comm.allreduce(float(uh.sub(0).eval(p, [cells.links(0)[0]])[0]) if own else 0.0, op=MPI.SUM)
# - Q1 with the full rule locks (0.0005 at L/h = 1000); TRIANGLES are not cured by one-point shear (P1: 0.008 at N = 32) -> P3 or quads
# - thick plates: compare with Reissner-Mindlin (hard SS: w_K + M/F, Navier + Marcus), not with Kirchhoff (18.9 % at L/h = 10, clamped)
# - V.sub(i).collapse() returns (space, [map]): the map is a LIST of arrays in 0.11 (np.concatenate it)
```

### 16q. Shells on surface meshes: a 2D mesh in 3D, DG0 local frame, P2/CR, drilling penalty

```python
el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,)))      # 2D cells, 3D points (no Gmsh needed)
msh = mesh.create_mesh(comm, cells, el, X)                                   # rank 0: X (n, 3), cells (m, 3); other ranks: empty arrays
# a non-manifold junction (I-beam web on a flange): just share the vertices of the junction line
t = ufl.Jacobian(msh)                                                        # 3 x 2
e3 = normalize(ufl.cross(ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]]), ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])))
e1 = ufl.cross(ufl.as_vector([0.0, 1.0, 0.0]), e3)                           # (use e_z where e3 is nearly along e_y), then e2 = e3 x e1
VT = fem.functionspace(msh, ("DG", 0, (3,))); f = fem.Function(VT); f.interpolate(fem.Expression(e3, VT.element.interpolation_points))
P = ufl.as_matrix([[e1[i], e2[i]] for i in range(3)])                         # tangent projector 3 x 2; ufl.grad is tangential on a manifold mesh
c = msh.basix_cell()
V = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", c, 2, shape=(3,)), basix.ufl.element("CR", c, 1, shape=(3,))]))
gu = ufl.dot(P.T, ufl.dot(ufl.grad(u), P)); beta = ufl.cross(e3, th)
eps, kap = ufl.sym(gu), ufl.sym(ufl.dot(P.T, ufl.dot(ufl.grad(beta), P)))
gam = ufl.dot(ufl.grad(ufl.dot(u, e3)), P) - ufl.dot(P.T, beta)
drill = (gu[0, 1] - gu[1, 0]) / 2 + ufl.dot(th, e3)                          # theta . e3 has no stiffness otherwise
a = (t_ * inner(Cps(eps), eps_) + t_**3 / 12 * inner(Cps(kap), kap_) + mu * t_ * dot(gam, gam_) + E * t_**3 / ufl.CellDiameter(msh)**2 * drill * drill_) * ufl.dx
# - without the drilling term: KSP reason -11 and NaN, LinearProblem.solve() does NOT raise -> check problem.solver.getConvergedReason()
# - u, theta are Cartesian: clamped = u = 0 and theta = 0; symmetry plane with normal n: u . n = 0 and theta x n = 0 (theta is a pseudo-vector)
# - Function.eval(p, [cell]) returns a FLAT array (3 values) in 0.11; flat facets: O(h^2) geometry error on curved shells
# - print what a boundary marker selected (count, midpoints): a box around "the ends" also catches free edges
```

### 16r. Periodic cells with dolfinx_mpc 0.11 (homogenization)

```python
# install: micromamba install -n fenicsx -c conda-forge dolfinx_mpc=0.11   (Colab: /content/micromamba/bin/micromamba, MAMBA_ROOT_PREFIX=/content/micromamba)
import dolfinx_mpc
on = lambda a, b: np.isclose(a, b, atol=1e-8)
pin = fem.dirichletbc(0.0, interior_vertex_dofs, V)          # pin the translation at an INTERIOR vertex, never at a master dof (silently ignored)
mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, lambda x: on(x[0], Lx) & ~on(x[1], Ly), lambda x: np.array([x[0] - Lx, x[1], x[2]]), [pin])
mpc.create_periodic_constraint_geometrical(V, lambda x: on(x[1], Ly) & ~on(x[0], Lx), lambda x: np.array([x[0], x[1] - Ly, x[2]]), [pin])
mpc.create_periodic_constraint_geometrical(V, lambda x: on(x[0], Lx) & on(x[1], Ly), lambda x: np.array([x[0] - Lx, x[1] - Ly, x[2]]), [pin])  # the corner ONCE
mpc.finalize()
v = dolfinx_mpc.LinearProblem(a, L, mpc, bcs=[pin], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                              petsc_options_prefix="cell_").solve()             # the prefix is required
# cell problem (elasticity): a = C eps(u) : eps(w) dx, L = -C E : eps(w) dx; C_hom[:, k] = (1/|Y|) * integral of C (E_k + eps(v_k)), |Y| = CELL area (voids included)
# periodic Gmsh mesh: gmsh.model.mesh.setPeriodic(1, right_edges, left_edges, [1,0,0,Lx, 0,1,0,0, 0,0,1,0, 0,0,0,1]) — pieces of a side paired IN ORDER;
#   OCC bounding boxes are enlarged by ~1e-7 (compare with 1e-6)
# checks: laminate (exact), symmetry of C_hom, -np 1 = -np 2, Keller k(k1,k2) k(k2,k1) = k1 k2 (2D), Hashin-Shtrikman bounds; kinematic >= periodic >= static
# very thin (graded) quadrilaterals: "Newton method failed to converge for non-affine geometry" on one rank -> the other rank hangs under mpirun
```

### 16s. Interfaces between submeshes: cohesive laws, the '+' side, opening control

```python
subs = [mesh.create_submesh(msh, tdim, ct.find(i)) for i in (1, 2)]             # (submesh, EntityMap, vertex map, geometry map)
V = [fem.functionspace(s[0], ("Lagrange", deg, (2,))) for s in subs]; em = [s[1] for s in subs]
# interface facets with the cell of subdomain 1 FIRST ('+'): plain meshtags order the two cells arbitrarily (mesh- and partition-dependent)
msh.topology.create_connectivity(fdim, tdim); msh.topology.create_connectivity(tdim, fdim)
f2c, c2f = msh.topology.connectivity(fdim, tdim), msh.topology.connectivity(tdim, fdim); data = []
for f in range(msh.topology.index_map(fdim).size_local):
    cs = f2c.links(f)
    if len(cs) == 2 and {tag[cs[0]], tag[cs[1]]} == {1, 2}:
        c0, c1 = (cs[0], cs[1]) if tag[cs[0]] == 1 else (cs[1], cs[0])
        data += [c0, int(np.where(c2f.links(c0) == f)[0][0]), c1, int(np.where(c2f.links(c1) == f)[0][0])]
dS = ufl.Measure("dS", domain=msh, subdomain_data=[(1, np.array(data, dtype=np.int32))])(1)
dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)                            # all integrals on the PARENT mesh
j = u[1]("-") - u[0]("+"); d = ufl.sqrt(ufl.dot(j, j) + 1e-30)
T = Gc / d0**2 * ufl.exp(-d / d0) * j                                            # exponential law, d0 = Gc / (e sc)
F = [inner(sig(u[0]), eps(w[0])) * dx(1) - dot(T, w[0]("+")) * dS, inner(sig(u[1]), eps(w[1])) * dx(2) + dot(T, w[1]("-")) * dS]
pb = NonlinearProblem(F, u, bcs=bcs, entity_maps=em, kind="mpi", petsc_options_prefix="czm_", petsc_options={...})
g = fem.petsc.assemble_vector([fem.form(-dot(w[0]("+"), n("+")) * dS, entity_maps=em), fem.form(dot(w[1]("-"), n("+")) * dS, entity_maps=em)], kind="mpi")
# opening control (bordered Newton): pb.solver.computeFunction(x, R); pb.solver.computeJacobian(x, A, A); K a = -R, K b = f, dlam = (target - g.x - g.a)/(g.b)
# Gmsh meshes: model_to_mesh(..., partitioner=mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, 2)) — else interior facets on partition boundaries are skipped
# problems re-created in a loop under MPI: gc.disable() + gc.collect() in the constructor (PETSc destructors are collective; random garbage collection hangs)
# never stop exactly at the peak (zero tangent: singular K); displacement control through a snap-back "converges" on the far branch
```

### 16t. Coupled flow and temperature in one mixed space (Boussinesq): continuation, consistent fluxes

```python
c = msh.basix_cell()
W = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", c, 2, shape=(2,)),   # u: P2
                                                     basix.ufl.element("Lagrange", c, 1),                # p: P1 (Taylor-Hood)
                                                     basix.ufl.element("Lagrange", c, 2)]))              # T: P2
w = fem.Function(W); u, p, T = ufl.split(w); v, q, s = ufl.TestFunctions(W)
Ra, Pr = fem.Constant(msh, 1e3), fem.Constant(msh, 0.71)                                # scaling kappa/L (the benchmark's); nu/L gives u / Pr
F = (ufl.inner(ufl.grad(u) * u, v) + Pr * ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - q * ufl.div(u)
     - Pr * Ra * T * v[1] + ufl.dot(u, ufl.grad(T)) * s + ufl.dot(ufl.grad(T), ufl.grad(s))) * ufl.dx   # buoyancy +Pr Ra T e_y on the right side
V, _ = W.sub(0).collapse(); Q, _ = W.sub(1).collapse(); S, mapS = W.sub(2).collapse()
bcs = [fem.dirichletbc(fem.Function(V), fem.locate_dofs_topological((W.sub(0), V), 1, walls), W.sub(0)),          # no slip
       fem.dirichletbc(fem.Function(Q), fem.locate_dofs_geometrical((W.sub(1), Q), corner), W.sub(1)),        # p only up to a constant: pin it
       fem.dirichletbc(Thot, fem.locate_dofs_topological((W.sub(2), S), 1, hot), W.sub(2)), ...]              # adiabatic walls: natural
pb = NonlinearProblem(F, w, bcs=bcs, petsc_options_prefix="bq_", petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "none",
                      "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
for Ra_ in (1e3, 1e4, 1e5, 1e6):                                                         # continuation: each solve starts from the last solution
    Ra.value = Ra_; pb.solve()
    assert pb.solver.getConvergedReason() > 0                                            # a failed solve can leave a plausible state
# consistent (residual) flux: the weak energy equation tested with z = 1 at the hot-wall dofs, 0 elsewhere — far more accurate than -dT/dx on the wall
z = fem.Function(S); z.x.array[fem.locate_dofs_topological(S, 1, hot)] = 1.0
Nu = comm.allreduce(fem.assemble_scalar(fem.form((ufl.dot(u, ufl.grad(T)) * z + ufl.dot(ufl.grad(T), ufl.grad(z))) * ufl.dx)), op=MPI.SUM) / H
# Nu_hot = Nu_cold follows from the cavity's symmetry for ANY solution: not a test. Velocity maxima: along x = 0.5 / y = 0.5, with positions.
# heated from below: u = 0, T linear solves the problem at every Ra; Newton from any u = 0 start returns it in one iteration ("converged", Nu = 1).
#   start from a roll (u AND T perturbed), continue along the branch in small steps (large steps can jump to the mirror roll)
# segregated (Picard) iteration: contraction grows with Ra (0.155 at 1e3, ~0.9 at 1e4, stalls at 3e4); relaxation T <- w T_new + (1 - w) T_old
```

### 16u. Axisymmetric advection–diffusion–reaction: a tubular reactor (Danckwerts, SUPG, mixing cup)

```python
z, r = ufl.SpatialCoordinate(msh)                                                  # x[0] = z (axial), x[1] = r (radial, axis at r = 0)
c = fem.Function(S); w = ufl.TestFunction(S)                                       # S = P2; u: a P2 vector Function (u_z, u_r), mean 1
F = (ufl.dot(u, ufl.grad(c)) * w + 1 / Pe * ufl.dot(ufl.grad(c), ufl.grad(w)) + Da * c**n * w) * r * ufl.dx   # weight r in EVERY integral
F += u[0] * (c - 1.0) * w * r * ds(INLET)                                          # Danckwerts: u c - (1/Pe) dc/dz = u * 1 (no Dirichlet at the inlet)
# axis r = 0: nothing (the boundary term carries r = 0); outlet: natural; reactive wall: dirichletbc(zero, wall dofs)
un = ufl.sqrt(ufl.dot(u, u) + 1e-30); Ku = ufl.JacobianInverse(msh) * u; h = un / ufl.sqrt(ufl.dot(Ku, Ku))   # cell length ALONG u
Peh = un * h * Pe / (2 * deg); tau = h / (2 * deg * un) * (1 / ufl.tanh(Peh) - 1 / Peh)
R = ufl.dot(u, ufl.grad(c)) - 1 / Pe * (ufl.div(ufl.grad(c)) + c.dx(1) / r) + Da * c**n   # the FULL strong residual (consistency)
F += tau * R * ufl.dot(u, ufl.grad(w)) * r * ufl.dx
NonlinearProblem(F, c, bcs=bcs, petsc_options_prefix="rx_", petsc_options={...}).solve()   # Newton (n = 2)
I = lambda e: comm.allreduce(fem.assemble_scalar(fem.form(e)), op=MPI.SUM)
c_mix = I(u[0] * c * r * ds(OUTLET)) / I(u[0] * r * ds(OUTLET))                    # the outlet concentration: flux-weighted, not the area mean
# axisymmetric (Navier-)Stokes: viscous term + u[1] * v[1] / r**2, div = u[0].dx(0) + u[1].dx(1) + u[1] / r, u_r = 0 on the axis; all times r
# exact limits: plug + dispersion (Danckwerts formula), segregated laminar flow int u exp(-Da_L/u) 2r dr, Taylor-Aris D(1 + Pe^2/48), Graetz Sh_inf = 3.657
```

### 16v. Brinkman in one domain; fields between a submesh and its parent; conjugate heat transfer

```python
invK = fem.Function(fem.functionspace(msh, ("DG", 0))); invK.x.array[ct.find(POROUS)] = 1 / K      # 1/K per cell, 0 in the free fluid
F = (mue * inner(grad(u), grad(v)) + mu * invK * dot(u, v) + beta * sqrt(dot(u, u) + 1e-20) * dot(u, v)
     - p * div(v) - q * div(u) - dot(f, v)) * dx                                   # Taylor-Hood; interface: u and shear continuous (natural)
# resolve the Brinkman layer sqrt(mue K / mu) at walls and interfaces; put the material interface ON mesh lines
fsub, em, _, _ = mesh.create_submesh(msh, tdim, ct.find(FLUID))                    # flow only where the fluid is
sub_cells = np.arange(fsub.topology.index_map(tdim).size_local + fsub.topology.index_map(tdim).num_ghosts, dtype=np.int32)
u_parent.interpolate(u_sub, cells0=sub_cells, cells1=em.sub_topology_to_topology(sub_cells, inverse=False))   # cells0: SOURCE, cells1: TARGET
# perfect contact: ONE temperature space on the parent, k per cell, advection Pe * dot(u_parent, grad(T)) * w (u = 0 in the solid)
# contact resistance: one space per submesh + hc * (Ts("-") - Tf("+")) * (ws("-") - wf("+")) * dS with explicit '+' = fluid facet data (16s);
#   NonlinearProblem([Ff, Fs], [Tf, Ts], entity_maps=[em_f, em_s], kind="mpi"); a huge hc as "perfect contact" ruins the conditioning (SNES -5)
# heat through a Dirichlet boundary: the residual of the weak form (no bcs) summed over its dofs — not -k grad(T).n
```

### 16w. Cahn–Hilliard schemes; a reused factorisation; a matrix-free operator; timing and ghosts

```python
W = fem.functionspace(msh, basix.ufl.mixed_element([P1, P1])); w, w0 = fem.Function(W), fem.Function(W)
V0, dofs = W.sub(0).collapse()                  # dofs is a LIST holding one index array: len(dofs) == 1
c, mu = ufl.split(w); c0, _ = ufl.split(w0); q, v = ufl.TestFunctions(W)
F = ((c - c0) / dt * q + M * inner(grad(mu), grad(q)) + (mu - (c**3 - c0)) * v - eps**2 * inner(grad(c), grad(v))) * dx   # convex splitting
#   backward Euler: c**3 - c (solvable only for dt*sigma < 1); stabilised: c0**3 - c0 + S*(c - c0) -> LINEAR, constant matrix
# initial noise as a FUNCTION OF x (cf.interpolate(lambda x: ...)): the same for every partition; check the SNES reason and the energy EVERY step
A = fem.petsc.assemble_matrix(fem.form(a)); A.assemble(); ksp.setOperators(A)          # constant matrix: assemble + factorise once
for n in range(nsteps):
    with b.localForm() as bl: bl.set(0.0)
    fem.petsc.assemble_vector(b, fem.form(L)); b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE); ksp.solve(b, w.x.petsc_vec); w.x.scatter_forward()
# matrix-free: PETSc.Mat().createPython([[nloc, N], [nloc, N]], Shell()); Shell.mult assembles form(action(a, wf)) into a GHOSTED work vector
# timing: comm.Barrier(); t = MPI.Wtime() ...; report the max over ranks; the first call includes JIT compilation
# ghosts: msh.topology.index_map(0).num_ghosts / .size_local; neighbours: set(index_map.owners)
```

### 16x. Verification for the project: errors against the exact expression, a QoI without an exact value, -np comparison

```python
x = ufl.SpatialCoordinate(msh); u_ex = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])       # the exact solution as a UFL EXPRESSION
dxq = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2 * p + 6})        # raised quadrature for the error integral
eL2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((u_h - u_ex)**2 * dxq)), op=MPI.SUM))
eH1 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_h - u_ex), ufl.grad(u_h - u_ex)) * dxq)), op=MPI.SUM))
# NOT: Iu = fem.Function(V); Iu.interpolate(...); norm(u_h - Iu) -> superconvergent, one order too good, 10-100x too small
# (if you must interpolate, use a space of degree >= p + 2)
reason = problem.solver.getConvergedReason()        # NonlinearProblem: SNES reason (> 0 converged); LinearProblem: problem.solver is the KSP
# projkit.py (Week 15, NumPy only): rate_table(N, {"L2": errs}, {"L2": p + 1}); richardson([Q1, Q2, Q3]) -> p_obs, Q_ext, GCI;
#   record("tag", comm, Q=Q) in the same cell with -np 1 and -np 2, then compare_np("tag") in an ordinary cell; budget(n, t, n_target)
```

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
