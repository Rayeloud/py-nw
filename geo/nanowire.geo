Include "nanowire_data.pro";

//+
SetFactory("OpenCASCADE");

// Parameters
READONLY = DefineNumber(1, Choices{0, 1}, Name "Parameters/0Read Only");
_dX = DefineNumber(dX, Name "Parameters/dX", Visible 0);
_offset = DefineNumber(offset, Name "Parameters/Center offset", Visible 1);

_R_1 = DefineNumber(R_1/dX, Name "Parameters/r_1", Visible 0);
_R_2 = DefineNumber(R_2/dX, Name "Parameters/r_2", Visible 0);

SHAPE_FLAG = DefineNumber(SHAPE, Choices{
    0="Cylindrical",
    1="Pentagonal"},
    Name "Parameters/Shape");

NB_NW_PARAM = DefineNumber(NB_NW-1, Choices{
0="1",
1="2"}, Name "Parameters/Nanowire number");

Lx = DefineNumber(Lx, Name "Parameters/Lx", ReadOnly READONLY);
Ly = DefineNumber(Ly, Name "Parameters/Ly", ReadOnly READONLY);
Lz = DefineNumber(Lz, Name "Parameters/Lz", ReadOnly READONLY);

R1 = DefineNumber(R_1, Name "Parameters/Nanowire/NW_0/R");
angle1 = 0;
R2 = DefineNumber(R_2, Name "Parameters/Nanowire/NW_1/R", Visible NB_NW_PARAM>0);
dist2 = DefineNumber(distance, Name "Parameters/Nanowire/NW_1/dist", Visible NB_NW_PARAM>0);
angle2 = DefineNumber(angle, Min 0, Max 90, Name "Parameters/Nanowire/NW_1/angle", Visible NB_NW_PARAM>0);
angle_param = DefineNumber(angle, Name "Parameters/angle", Visible 0);

N = DefineNumber(96, Name "Parameters/N");

Nx = DefineNumber(Lx/dX, Name "Parameters/Nx", ReadOnly READONLY);
Ny = DefineNumber(Ly/dX, Name "Parameters/Ny", ReadOnly READONLY);
Nz = DefineNumber(Lz/dX, Name "Parameters/Nz", ReadOnly READONLY);

// Define domain
Point(1) = {0, 0, 0, 100.0}; Point(2) = {Lx, 0, 0, 100.0};
Point(3) = {Lx, Ly, 0, 100.0}; Point(4) = {0, Ly, 0, 100.0};

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
    Extrude {0, 0, Lz} {
        Surface{13}; Surface{11};
    }
EndIf

If (SHAPE_FLAG == 1)
    center_x = Lx/2;
    center_y = Ly/2 - offset;
    theta = 2*Pi/5;
    theta_offset = Pi / 2 + theta; //offset to start from the top
    R_pent = Sqrt(2*Pi/(5*Sin(theta)))*R1;
    // Define points
    For i In {1:5}
        x_i = center_x + R_pent*Cos(i*theta+theta_offset);
        y_i = center_y + R_pent*Sin(i*theta+theta_offset);
        idx = i-1;
        Printf("idx = %g", idx);
        Point(5+idx) = {x_i, y_i, 0, 1.0};
    EndFor
    // Define curves
    Line(5) = {5, 6}; Line(6) = {6, 7}; Line(7) = {7, 8}; Line(8) = {8, 9}; Line(9) = {9, 5};
    // Define curve loops
    Curve Loop(10) = {5, 6, 7, 8, 9}; Plane Surface(11) = {10};
    Curve Loop(12) = {1, 2, 3, 4}; Plane Surface(13) = {12, 10};

    // Extrude domain and NW
    Extrude {0, 0, Lz} {
        Surface{13}; Surface{11};
    }
EndIf

// Define second NW

// rotation of the NW
factor = DefineNumber(0.9, Name "Parameters/Nanowire/NW_1/factor", Visible NB_NW_PARAM>0);
If (NB_NW_PARAM == 1 && angle2 > 0)
    ALPHA = angle2*Pi/180;
    If (angle2 <= 45)
        TAN_ALPHA = Tan(ALPHA);
        COS_ALPHA = Cos(ALPHA);
        SIN_ALPHA = Sin(ALPHA);
        X = (Lz * TAN_ALPHA + Lx)/2;
        X = Lx - X;
        Z = 0;

        DL = Lz/COS_ALPHA;
        DL_SCALED = DL*factor;
        D_X = (DL-DL_SCALED)/2*SIN_ALPHA;
        D_Z = (DL-DL_SCALED)/2*COS_ALPHA;
        DL = DL_SCALED;
    Else
        TAN_ALPHA = Tan(Pi/2 - ALPHA);
        COS_ALPHA = Cos(Pi/2 - ALPHA);
        SIN_ALPHA = Sin(Pi/2 - ALPHA);
        X = 0;
        Z = (Lx * TAN_ALPHA + Lz)/2;
        Z = Lz - Z;
        If (angle2 < 90)
            DL = Lx/COS_ALPHA;
            DL_SCALED = DL*factor;
            D_X = (DL-DL_SCALED)/2*COS_ALPHA;
            D_Z = (DL-DL_SCALED)/2*SIN_ALPHA;
            DL = DL_SCALED;
        Else
            DL = Lx;
            D_X = 0;
            D_Z = 0;
        EndIf
    EndIf

    X = X + D_X;
    Z = Z + D_Z;

    DX = DL*Sin(ALPHA);
    DZ = DL*Cos(ALPHA);

    y_offset = R1 + R2 + dist2;

    If (SHAPE_FLAG == 0)
        Point(20) = {X, center_y + y_offset, Z, 1.0};
        Point(21) = {X + R2, center_y + y_offset, Z, 1.0}; Point(22) = {X - R2, center_y + y_offset, Z, 1.0};
        Point(23) = {X, center_y + y_offset + R2, Z, 1.0}; Point(24) = {X, center_y + y_offset - R2, Z, 1.0};
        Rotate {{0, 1, 0}, {X, center_y + y_offset, Z}, angle2*Pi/180} {
            Point{24}; Point{22}; Point{23}; Point{21};
          }

        // Define curves
        Circle(100) = {23, 20, 21}; Circle(101) = {21, 20, 24};
        Circle(102) = {24, 20, 22}; Circle(103) = {22, 20, 23};
        
        // Define curve loops
        Curve Loop(104) = {100, 101, 102, 103}; 
    EndIf

    If (SHAPE_FLAG == 1)
        theta = 2*Pi/5;
        theta_offset = Pi / 2 + theta; //offset to start from the top
        R_pent = Sqrt(2*Pi/(5*Sin(theta)))*R2;
        // Define points
        For i In {1:5}
            x_i = X + R_pent*Cos(i*theta+theta_offset);
            y_i = center_y + y_offset + R_pent*Sin(i*theta+theta_offset);
            idx = i-1;
            Printf("idx = %g", idx);
            Point(20+idx) = {x_i, y_i, Z, 1.0};
        EndFor
        Rotate {{0, 1, 0}, {X, center_y + y_offset, Z}, angle2*Pi/180} {
            Point{24}; Point{22}; Point{23}; Point{21};Point{20};
            }
        // Define curves
        Line(99) = {20, 21}; Line(100) = {21, 22}; 
        Line(101) = {22, 23}; Line(102) = {23, 24}; Line(103) = {24, 20};
        
        // Define curve loops
        Curve Loop(104) = {99, 100, 101, 102, 103};
    EndIf

    Plane Surface(105) = {104};

    //+
    Extrude {DX, 0, DZ} {
      Surface{105};
    }

    vol_ids[] = BooleanFragments{Volume{1}; Delete; }{Volume{2, 3}; Delete; };
    For i In {0:#vol_ids()-1}
        Printf("Volume ids: %g\n", vol_ids[i]);
    EndFor

    Physical Volume("Nanowire") = {vol_ids[0], vol_ids[1]};
    // Meshing

    // first NW
    If (SHAPE_FLAG == 0)
        // face 1
        Transfinite Curve {19, 24, 23, 21, 5, 8, 6, 7} = 10 Using Progression 1;
        // side length 1
        Transfinite Curve {17, 18, 20, 22} = 10 Using Progression 1;

        // face surface 1 
        Transfinite Surface {23}; Transfinite Surface {11};
        // side surface 1
        Transfinite Surface {18}; Transfinite Surface {21}; Transfinite Surface {19}; Transfinite Surface {20};
        // volume 1
        Transfinite Volume{2};

        // face 2
        Transfinite Curve {106, 111, 110, 108, 102, 101, 103, 100} = 10 Using Progression 1;
        // side length 2
        Transfinite Curve {104, 105, 107, 109} = 10 Using Progression 1;

        // face surface 2
        Transfinite Surface {110}; Transfinite Surface {105};

        // side surface 2
        Transfinite Surface {109}; Transfinite Surface {106}; Transfinite Surface {107}; Transfinite Surface {108};
        // volume 2
        Transfinite Volume{3};

        // domain
        Transfinite Curve {123, 121, 122, 116, 115, 114, 113, 112} = 10 Using Progression 1;
        Transfinite Curve {117, 120, 119, 118} = 10 Using Progression 1;

        // Transfinite Curve {128, 131, 130, 129} = 10 Using Progression 1; Transfinite Curve {123, 126, 125, 124} = 10 Using Progression 1;
        // Transfinite Curve {132, 135, 134, 133} = 1 Using Progression 1;
        
        // Transfinite Surface {23}; Transfinite Surface {11};
        // Transfinite Surface {117}; Transfinite Surface {118}; Transfinite Surface {119}; Transfinite Surface {120};
        // Transfinite Volume{vol_ids[0]};
        
        // // second NW
        // Transfinite Curve {106, 111, 110, 108} = 10 Using Progression 1; Transfinite Curve {100, 103, 102, 101} = 10 Using Progression 1;
        // Transfinite Curve {105, 104, 109, 107} = 1 Using Progression 1;

        // Transfinite Surface {110}; Transfinite Surface {105};
        // Transfinite Surface {106}; Transfinite Surface {109}; Transfinite Surface {108}; Transfinite Surface {107};
        // Transfinite Volume{vol_ids[1]};

        // // domain
        // // Horizontal edges
        // Transfinite Curve {122, 116, 127, 119, 114, 112, 115, 113} = 5 Using Progression 1;
        // // Vertical edges
        // Transfinite Curve {120, 117, 118, 121} = 3 Using Progression 1;
    EndIf
    If (SHAPE_FLAG == 1)
        // NW 1
        Transfinite Curve {133, 132, 131, 135, 134, 127, 128, 126, 125, 129} = 10 Using Progression 1;
        Transfinite Curve {139, 140, 138, 137, 136} = 1 Using Progression 1;
        // NW2
        Transfinite Curve {112, 110, 108, 106, 113, 101, 102, 103, 99, 100} = 10 Using Progression 1;
        Transfinite Curve {109, 107, 111, 105, 104} = 1 Using Progression 1;
        // NW1
        //Transfinite Surface {24}; Transfinite Surface {11}; // FACES
        Transfinite Surface {120}; Transfinite Surface {121}; Transfinite Surface {119};
        Transfinite Surface {122}; Transfinite Surface {118};
        // NW2
        //Transfinite Surface {111}; Transfinite Surface {105}; // FACES
        Transfinite Surface {108}; Transfinite Surface {109}; Transfinite Surface {107};
        Transfinite Surface {110}; Transfinite Surface {106};

        // domain
        // Horizontal edges
        Transfinite Curve {115, 121, 116, 124, 118, 130, 117, 114} = 5 Using Progression 1;
        // Vertical edges
        Transfinite Curve {120, 119, 123, 122} = 3 Using Progression 1;
    EndIf
Else
    Physical Volume("Nanowire") = {2};

    // Meshing

    // First NW (one NW config)
    If (SHAPE_FLAG == 0)
        Transfinite Curve {19, 21, 23, 24, 5, 6, 7, 8} = 15 Using Progression 1;
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
    If (SHAPE_FLAG == 1)
        Transfinite Curve {24, 22, 20, 27, 26, 7, 6, 5, 9, 8} = 10 Using Progression 1;
        Transfinite Curve {23, 21, 19, 18, 25} = 1 Using Progression 1;
        Transfinite Surface {20}; Transfinite Surface {19}; Transfinite Surface {18};
        Transfinite Surface {22}; Transfinite Surface {21};

        // domain
        // Horizontal edges
        Transfinite Curve {12, 11, 1, 10, 16, 13, 3, 15} = 5 Using Progression 1;
        // Vertical edges
        Transfinite Curve {14, 2, 4, 17} = 3 Using Progression 1;
    EndIf
EndIf

