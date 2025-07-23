

struct Pt2d {
    int x, y;

    // add in an assignment op???
    pos operator+(const pos& a) const { return Pt2d(x+a.x, y+a.y); }
    pos operator-(const pos& a) const { return Pt2d(x-a.x, y-a.y); }
    pos operator*(const int a) const { return Pt2d(x * a, y * a); }
    pos operator/(const int a) const { return Pt2d(x / a, y / a); }
    bool operator==(const pos& a) const { return x == a.x && y == a.y; }
    bool operator<(const pos& a) const { return (x == a.x)?  y < a.y: x < a.x; }
    // add in the round function
    // add in the str/io overload?
    // add in the hash func 
}

struct Pt3d{
    int x, y, z;
    // need to fix this so that z is acocunted for
 
    // add in an assignment op???
    pos operator+(const pos& a) const { return Pt2d(x+a.x, y+a.y); }
    pos operator-(const pos& a) const { return Pt2d(x-a.x, y-a.y); }
    pos operator*(const int a) const { return Pt2d(x * a, y * a); }
    pos operator/(const int a) const { return Pt2d(x / a, y / a); }
    bool operator==(const pos& a) const { return x == a.x && y == a.y; }
    bool operator<(const pos& a) const { return (x == a.x)?  y < a.y: x < a.x; }
    // add in the round function
    // add in the str/io overload?
    // add in the hash func 
}


class GeometryAlgorithms {
    // add in the code here
    double dot_product(Pt2d left_vec, Pt2d right_vec) {
        return left_vec.x * right_vec.x + left_vec.y * right_vec.y;
    }

};
