Include "nanowire_common.pro";

//+
SetFactory("OpenCASCADE");

// Parameters
_dX = DefineNumber(dX, Name "Parameters/dX", Visible 0);
_offset = DefineNumber(offset, Name "Parameters/Center offset", Visible 0);

_R_1 = DefineNumber(R_1, Name "Parameters/r_1", Visible 0);
_R_2 = DefineNumber(R_2, Name "Parameters/r_2", Visible 0);

NB_NW_PARAM = DefineNumber(NB_NW-1, Choices{
0="1",
1="2"}, Name "Parameters/Nanowire number");

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

// Define second NW

// rotation of the NW
factor = DefineNumber(0.9, Name "Parameters/Nanowire/NW_1/factor", Visible NB_NW_PARAM>0);
If (NB_NW_PARAM == 1 && angle2 > 0)
    /*
    TAN_ALPHA = Tan((90-angle2)*Pi/180);
    TAN_45 = Tan(45*Pi/180);

    DZ = TAN_ALPHA*L;
    DX = L;

    Z = L/2 - DZ/2;
    X = L/2 - DX/2;

    If (angle2 == 45)
        a = L - Z;
        L_max = Sqrt(a*a + L*L);
        factor = 0.9;
    Else
        b = L - 2*X;
        L_max = Sqrt(L*L + b*b);
        factor = L/L_max;
    EndIf

    D = L_max/Sqrt(1+TAN_ALPHA*TAN_ALPHA)*factor;

    DZ = TAN_ALPHA*D;
    DX = D;

    Z = L/2 - DZ/2;
    X = L/2 - DX/2;

    y_offset = R1 + R2 + dist2;

    Point(20) = {X, H/2 + y_offset, Z, 1.0};
    Point(21) = {X + R2, H/2 + y_offset, Z, 1.0}; Point(22) = {X - R2, H/2 + y_offset, Z, 1.0};
    Point(23) = {X, H/2 + y_offset + R2, Z, 1.0}; Point(24) = {X, H/2 + y_offset - R2, Z, 1.0};

    // Rotate points to direction of second NW
    Rotate {{0, 1, 0}, {X, H/2 + y_offset, Z}, angle2*Pi/180} {
      Point{24}; Point{22}; Point{23}; Point{21};
    }

    Circle(100) = {23, 20, 21}; Circle(101) = {21, 20, 24};
    Circle(102) = {24, 20, 22}; Circle(103) = {22, 20, 23};

    Curve Loop(104) = {100, 101, 102, 103}; Plane Surface(105) = {104};

    //+
    Extrude {DX, 0, DZ} {
      Surface{105};
    }

    vol_ids[] = BooleanFragments{Volume{1}; Delete; }{Volume{2, 3}; Delete; };

    For i In {0:#vol_ids()-1}
        Printf("Volume ids: %g\n", vol_ids[i]);
    EndFor

    Physical Volume("Nanowire") = {vol_ids[0], vol_ids[1]};

    // get NW curve ids
    tol = 1e-3;
    curve_ids() = Curve In BoundingBox{0-tol, 0-tol, 0-tol, L+tol, H+tol, L+tol}; // all the curve ids
    NW_curve_ids() = Curve In BoundingBox{0-tol, 0+tol, 0-tol, L+tol, H-tol, L+tol}; // curve ids associated with the inside of the domain
    */

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
        D_X = (DL-DL_SCALED)/2*Sin(ALPHA);
        D_Z = (DL-DL_SCALED)/2*Cos(ALPHA);
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
            D_X = (DL-DL_SCALED)/2*Sin(ALPHA);
            D_Z = (DL-DL_SCALED)/2*Cos(ALPHA);
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

    Point(20) = {X, center_y + y_offset, Z, 1.0};
    Point(21) = {X + R2, center_y + y_offset, Z, 1.0}; Point(22) = {X - R2, center_y + y_offset, Z, 1.0};
    Point(23) = {X, center_y + y_offset + R2, Z, 1.0}; Point(24) = {X, center_y + y_offset - R2, Z, 1.0};

    Rotate {{0, 1, 0}, {X, center_y + y_offset, Z}, angle2*Pi/180} {
      Point{24}; Point{22}; Point{23}; Point{21};
    }

    Circle(100) = {23, 20, 21}; Circle(101) = {21, 20, 24};
    Circle(102) = {24, 20, 22}; Circle(103) = {22, 20, 23};

    Curve Loop(104) = {100, 101, 102, 103}; Plane Surface(105) = {104};

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
    Transfinite Curve {128, 131, 130, 129} = 10 Using Progression 1; Transfinite Curve {123, 126, 125, 124} = 10 Using Progression 1;
    Transfinite Curve {132, 135, 134, 133} = 1 Using Progression 1;
    
    Transfinite Surface {23}; Transfinite Surface {11};
    Transfinite Surface {117}; Transfinite Surface {118}; Transfinite Surface {119}; Transfinite Surface {120};
    Transfinite Volume{vol_ids[0]};
    // second NW
    Transfinite Curve {106, 111, 110, 108} = 10 Using Progression 1; Transfinite Curve {100, 103, 102, 101} = 10 Using Progression 1;
    Transfinite Curve {105, 104, 109, 107} = 1 Using Progression 1;

    Transfinite Surface {110}; Transfinite Surface {105};
    Transfinite Surface {106}; Transfinite Surface {109}; Transfinite Surface {108}; Transfinite Surface {107};
    Transfinite Volume{vol_ids[1]};

    // domain
    // Horizontal edges
    Transfinite Curve {122, 116, 127, 119, 114, 112, 115, 113} = 5 Using Progression 1;
    // Vertical edges
    Transfinite Curve {120, 117, 118, 121} = 3 Using Progression 1;
Else
    Physical Volume("Nanowire") = {2};

    // Meshing

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

