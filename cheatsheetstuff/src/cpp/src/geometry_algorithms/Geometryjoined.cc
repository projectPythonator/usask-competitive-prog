

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
    double dot_product(Pt2d a) { return x * a.x + y * a.y; }
    double cross_product(Pt2d a) { return x * a.y - y * a.x; }
    Pt2d rot_90_cw() { return Pt2d{y, -x}; } // clockwise
    Pt2d rot_90_ccw() { return Pt2d{-y, x}; } // counter clock wise
    bool is_between(Pt2d a, Pt2d b) {
        return min(a, b) <= this && this <= max(a, b);
    }
    
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

    int compare_ab(double a, double b) {
        return (isless(a, b))? -1: (isgreater(a, b))? 1: 0;
    }

    double distance_normalized(Pt2d leftPoint, P2td rightPoint){
        return dist(leftPoint, rightPoint); // use this from c++ math
    }

    double distance_normalized(Pt2d leftPoint, P2td rightPoint){
        return leftPoint.dot_product(rightPoint);
    }

    Pt2d rotate_ccw_rads_wrt_origin(Pt2d point, double rads) {
        return Pt2d(point.x * cos(rads) - point.y * sin(rads),
                    point.x * sin(rads) + point.y * cos(rads));
    }

    int point_c_orientation_wrt_line_ab(Pt2d a, Pt2d b, Pt2d c) {
        return compare_ab((b - a).cross_product(c - a), 0.0);
    }

    double angle_point_c_wrt_line_ab(Pt2d a, Pt2d b, Pt2d c) {
        Pt2d [vector_ba, vector_bc] = [b - a, c - b];
        double dot_ba_bc = vector_ba.dot_product(vector_bc);
        double dist_sq_ba = vector_ba.dot_product(vector_ba);
        double dist_sq_bc = vector_bc.dot_product(vector_bc);
        return acos(dot_ba_bc / (sqrt(dist_sq_ba) * sqrt(dist_sq_bc)));
    }

    bool is_point_c_on_line_segment_ab(Pt2d a, Pt2d b, Pt2d c){
        Pt2d [vec_ca, vec_cb] = [a - c, b - c];
        return (compare_ab(vec_ca.cross_product(vec_cb), 0.0) == 0
                && compare_ab(vec_ca.dot_product(vec_cb), 0.0) <= 0);
    }

    Pt2d project_pt_c_to_line_ab(Pt2d a, Pt2d b, Pt2d c) {
        Pt2d [vec_ab, vec_ac] = [b - a, c - a];
        Pt2d translated = vec_ab * (vec_ac.dot_product(vec_ab)/ vec_ab.dot_product(vec_ab));
        return a * translated;
    }

    Pt2d project_pt_c_to_line_seg_ab(Pt2d a, Pt2d b, Pt2d c) {
        if (a == b) 
            return a;
        Pt2d [vec_ab, vec_ac] = [b - a, c - a];
        double u = (vec_ac.dot_product(vec_ab)/ vec_ab.dot_product(vec_ab));
        return (u < 0.0)? a: (u > 1.0)? b: project_pt_c_to_line_ab(a, b, c);
    }

    double distance_pt_c_to_line_ab(Pt2d a, Pt2d b, Pt2d c) {
        Pt2d closest = project_pt_c_to_line_ab(a, b, c);
        return distance_normalized(c, closest);
    }

    double distance_pt_c_to_line_seg_ab(Pt2d a, Pt2d b, Pt2d c) {
        Pt2d closest = project_pt_c_to_line_seg_ab(a, b, c);
        return distance_normalized(c, closest);
    }

    bool is_parallel_lines_ab_and_cd(Pt2d a, Pt2d b, Pt2d c, Pt2d d) {
        Pt2d [vec_ab, vec_dc] = [b - a, c - d];
        return compare_ab(vec_ab.cross_product(vec_dc), 0.0) == 0;
    }

    bool is_colinear_lines_ab_and_cd_1(Pt2d a, Pt2d b, Pt2d c, Pt2d d) {
        return (is_parallel_lines_ab_and_cd(a, b, c, d)
                && is_parallel_lines_ab_and_cd(b, a, a, c)
                && is_parallel_lines_ab_and_cd(d, c, c, a));
    }

    bool is_colinear_lines_ab_and_cd_2(Pt2d a, Pt2d b, Pt2d c, Pt2d d) {
        return (point_c_orientation_wrt_line_ab(a, b, c) == 0
                && point_c_orientation_wrt_line_ab(a, b, d) == 0);
    }

    bool is_segment_intersect_ab_to_cd(Pt2d a, Pt2d b, Pt2d c, Pt2d d){
        if (is_colinear_lines_ab_and_cd_2(a, b, c, d))
            return (c.is_between(a, b) || d.is_between(a, b));
        Pt2d [vec_ad, vec_ab, vec_ac] = d - a, b - a, c - a;
        Pt2d [vec_ca, vec_cd, vec_cb] = a - c, d - c, b - c;
        double a_value = vec_ad.cross_product(vec_ab) * vec_ac.cross_product(vec_ab);
        double c_value = vec_ca.cross_product(vec_cd) * vec_cb.cross_product(vec_cd);
        return !(a_value > 0 || c_value > 0); // flip this with demorg or something later
    }

    bool is_lines_intersect_ab_to_cd(Pt2d a, Pt2d b, Pt2d c, Pt2d d){
        return !(is_parallel_lines_ab_and_cd(a, b, c, d)
                || is_colinear_lines_ab_and_cd_2(a, b, c, d));
    }

    Pt2d pt_lines_intersect_ab_to_cd(Pt2d a, Pt2d b, Pt2d c, Pt2d d){
        Pt2d [vec_ab, vec_ac, vec_dc] = b - a, c - a, c - d;
        Pt2d vec_t = vec_ab * (vec_ac.cross_product(vec_cd) / vec_ab.cross_product(vec_cd));
        return a + vec_t;
    }

    Pt2d pt_line_seg_intersect_ab_to_cd(Pt2d a, Pt2d b, Pt2d c, Pt2d d){
        double x, y, cross = c.x - d.x, d.y - c.y, d.cross_product(c);
        double u = abs(y * a.x + x * a.y + cross);
        double v = abs(y * b.x + x * b.y + cross);
        return Pt2d((a.x * v + b.x * u) / (v + u), (a.y * v + b.y * u) / (v + u));
    }


};
