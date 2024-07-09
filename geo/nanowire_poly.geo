Include "nanowire_data.pro";

//+
SetFactory("OpenCASCADE");

// Parameters
_dX = DefineNumber(dX, Name "Parameters/dX", Visible 0);
_offset = DefineNumber(offset, Name "Parameters/Center offset", Visible 0);

_R_1 = DefineNumber(R_1, Name "Parameters/r_1", Visible 0);
_R_2 = DefineNumber(R_2, Name "Parameters/r_2", Visible 0);

eps = DefineNumber(0.1, Name "Paramerters/eps", Visible 1);

SHAPE_FLAG = DefineNumber(SHAPE, Choices{
    0="Cylindrical",
    1="Pentagonal"}, 
    Name "Parameters/Shape");

TYPE_FLAG = DefineNumber(0, Choices{
    0, 1}, 
    Name "Parameters/In Barycenters?", Visible SHAPE_FLAG>0);

NB_NW_PARAM = DefineNumber(NB_NW-1, Choices{
    0="1",
    1="2"}, 
    Name "Parameters/Nanowire number");

/*
SHAPE = DefineNumber(0, Choices{
0="Cylinder",
1="Pentagonal"}, Name "Parameters/Nanowire/Shape");
*/

R1 = DefineNumber(R_1, Name "Parameters/Nanowire/NW_0/R");
angle1 = 0;
R2 = DefineNumber(R_2, Name "Parameters/Nanowire/NW_1/R", Visible NB_NW_PARAM>0);
dist2 = DefineNumber(distance, Name "Parameters/Nanowire/NW_1/dist", Visible NB_NW_PARAM>0);
angle2 = DefineNumber(angle, Min 0, Max 90, Name "Parameters/Nanowire/NW_1/angle", Visible NB_NW_PARAM>0);
angle_param = DefineNumber(angle, Name "Parameters/angle", Visible 0);

// WIP
R3 = DefineNumber(12, Name "Parameters/Nanowire/NW_2/R", Visible NB_NW_PARAM>1);
angle3 = DefineNumber(45, Name "Parameters/Nanowire/NW_2/angle", Visible NB_NW_PARAM>1);

N = DefineNumber(96, Name "Parameters/N");

L = Lx;
H = Ly;

Lx = DefineNumber(Lx, Name "Parameters/Lx", ReadOnly 1);
Ly = DefineNumber(Ly, Name "Parameters/Ly", ReadOnly 1);
Lz = DefineNumber(Lz, Name "Parameters/Lz", ReadOnly 1);

Nx = DefineNumber(N*L/H, Name "Parameters/Nx", ReadOnly 1);
Ny = DefineNumber(N, Name "Parameters/Ny", ReadOnly 1);
Nz = DefineNumber(N*L/H, Name "Parameters/Nz", ReadOnly 1);

// Define domain
Point(1) = {0, 0, 0, 10.0}; Point(2) = {L, 0, 0, 10.0};
Point(3) = {L, H, 0, 10.0}; Point(4) = {0, H, 0, 10.0};

Line(1) = {1, 2}; Line(2) = {2, 3};
Line(3) = {3, 4}; Line(4) = {4, 1};

// Define first NW
If (SHAPE_FLAG == 0)
    center_x = Lx/2;
    center_y = Ly/2 - offset;
    Point(5) = {center_x, center_y, 0, 1.0};
    Point(6) = {center_x, center_y - R1, 0, 1.0}; Point(7) = {center_x, center_y + R1, 0, 1.0};
    Point(8) = {center_x - R1, center_y, 0, 1.0}; Point(9) = {center_x + R1, center_y, 0, 1.0};

    Circle(5) = {7, 5, 9}; Circle(6) = {9, 5, 6};
    Circle(7) = {6, 5, 8}; Circle(8) = {8, 5, 7};

    // Define curve loops
    Curve Loop(10) = {5, 6, 7, 8}; Plane Surface(11) = {10};
    Curve Loop(12) = {1, 2, 3, 4}; Plane Surface(13) = {12, 10};

    // Extrude domain and NW
    Extrude {0, 0, L} {
        Surface{13}; Surface{11};
    }
Else
    // Pentagonal shape 
    // circles at vertex of pentagone inscribe in a circle of radius R1
    // center of the pentagone is at (center_x, center_y)
    If (TYPE_FLAG == 0)
        center_x = Lx/2;
        center_y = Ly/2 - offset;
        theta = 2*Pi/5;
        theta_offset = Pi / 2 + theta; //offset to start from the top
        R_pent = R1 * Sin(theta/2)-eps;
        // Define points
        x_0 = center_x;
        y_0 = center_y;
        For i In {1:5}
            x_i = x_0 + R1*Cos(i*theta+theta_offset);
            y_i = y_0 + R1*Sin(i*theta+theta_offset);
            idx = i-1;
            Printf("idx = %g", idx);
            Point(5+5*idx) = {x_i, y_i, 0, 1.0};
            Point(5+5*idx+1) = {x_i         , y_i - R_pent, 0, 1.0}; Point(5+5*idx+2) = {x_i         , y_i + R_pent, 0, 1.0};
            Point(5+5*idx+3) = {x_i - R_pent, y_i         , 0, 1.0}; Point(5+5*idx+4) = {x_i + R_pent, y_i         , 0, 1.0};

            Circle(5+4*idx) = {5+5*idx+2, 5+5*idx, 5+5*idx+4}; Circle(5+4*idx+1) = {5+5*idx+4, 5+5*idx, 5+5*idx+1};
            Circle(5+4*idx+2) = {5+5*idx+1, 5+5*idx, 5+5*idx+3}; Circle(5+4*idx+3) = {5+5*idx+3, 5+5*idx, 5+5*idx+2};

            // Define curve loops
            Curve Loop(10+2*idx) = {5+4*idx, 5+4*idx+1, 5+4*idx+2, 5+4*idx+3}; Plane Surface(10+2*idx+1) = {10+2*idx};
            ids(idx) = 10+2*idx;
            surfs(idx) = 10+2*idx+1;
            Printf("surfs(%g) = %g", idx, surfs(idx));
        EndFor

        Curve Loop(10+2*5+2) = {1, 2, 3, 4}; Plane Surface(10+2*5+3) = {10+2*5+2, ids(0), ids(1), ids(2), ids(3), ids(4)};
        // Extrude domain and NW
        Extrude {0, 0, L} {
            Surface{10+2*5+3};
        }
        For i In {0:5-1}
            Extrude {0, 0, L} {
                Surface{surfs(i)};
            }
        EndFor
    Else
        center_x = Lx/2;
        center_y = Ly/2 - offset;
        theta = 2*Pi/5;
        theta_offset = Pi / 2 + theta; //offset to start from the top
        R_pent = R1 * Sin(theta/2)-eps;
        // Define points
        x_0 = center_x;
        y_0 = center_y;
        For i In {1:5}
            x(i-1) = x_0 + R1*Cos(i*theta+theta_offset);
            y(i-1) = y_0 + R1*Sin(i*theta+theta_offset);
        EndFor
        // Find all barycenters in pentagon triangles
        For i In {0:5-1}
            x_bary(i) = (x(i) + x((i+1)%5) + x_0) / 3;
            y_bary(i) = (y(i) + y((i+1)%5) + y_0) / 3;
            Printf("x_bary(%g) = %g", i, x_bary(i));
            Printf("y_bary(%g) = %g", i, y_bary(i));
            Point(5+i) = {x_bary(i), y_bary(i), 0, 1.0};
        EndFor
        Point(100) = {x_0, y_0, 0, 1.0}; Point(101) = {x_0+R1, y_0, 0, 1.0};
        Point(102) = {x_0, y_0+R1, 0, 1.0}; Point(103) = {x_0-R1, y_0, 0, 1.0};
        Point(104) = {x_0, y_0-R1, 0, 1.0};
        Circle(1000) = {101, 100, 102}; Circle(1001) = {102, 100, 103};
        Circle(1002) = {103, 100, 104}; Circle(1003) = {104, 100, 101};
    EndIf
EndIf

// rotation of the NW
factor = DefineNumber(0.9, Name "Parameters/Nanowire/NW_1/factor", Visible NB_NW_PARAM>0);
Physical Volume("Nanowire") = {2, 3, 4, 5, 6};

// Meshing
If (SHAPE_FLAG == 0)
    // First NW (one NW config)
    Transfinite Curve {19, 21, 23, 24, 5, 6, 7, 8} = 30 Using Progression 1;
    Transfinite Curve {17, 22, 18, 20} = 1 Using Progression 1;
    Transfinite Surface {18}; Transfinite Surface {21}; Transfinite Surface {19}; Transfinite Surface {20};
    Transfinite Surface {11}; Transfinite Surface {23};
    Transfinite Volume{2};

    // domain
    // Horizontal edges
    Transfinite Curve {15, 11, 14, 9, 3, 1, 12, 10} = 5 Using Progression 1;
    // Vertical edges
    Transfinite Curve {4, 16, 2, 13} = 3 Using Progression 1;
EndIf


//+
