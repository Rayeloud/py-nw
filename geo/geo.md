# Input file

The solver requires a `.geo` file which defines the physical domain to be solved.

The geometry allows for the creation of a coarse mesh which will be used to create the computation grid used by the solver.

An input file has the following structure:

```cpp
// A good practice is to include an external file
// which will store all the parameters for an easy access
// when using the Cahn Hilliard solver.
Include "<your_parameters>.pro";

...

// GEOMETRY DEFINITION (see gmsh documentation)
// ----------------------------------------------------------------
// for example:
Box(1) = {0, 0, 0, L, L, H}; // domain

Cylinder(2) = {0, L/2, H/2, L, 0, 0, R, 2*Pi}; // sub-set of the domain to be set

// trick to separate the box from the cylinder
vols[] = BooleanFragments{Volume{1}; Delete;}{Volume{2}; Delete;}; 

...

// Important to define 1 physical volume to perform the voxelisation.
// The solver sets the initial conditions the domain by discretizing 
// the volume with voxels and setting each voxels to its appropriate 
// value.
// Physical Volume("<your_physical_volume_name>") = {<volume_ids>};
// for example:
Physical Volume("Comp1") = {vols[1]}; // supposing vols[1] is the cylinder and the rest of the domain is to be set to Comp2

...

// MESH DEFINITION (it is recommended to define Transfinite Volume)
// ----------------------------------------------------------------
// Ideally in the given example, the cylinder should have been split 
// in four to define a transfinite surface.

...

```
