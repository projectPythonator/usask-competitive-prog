

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

    bool is_point_in_radius_of_circle(Pt2d point, Pt2d centerPoint, double radius) {
        return compare_ab(distance_normalized(point, center_point), radius) < 0;
    }

    Pt2d pt_circle_center_given_pt_abc(Pt2d a, Pt2d b, Pt2d c) {
        ab, ac = (a + b) / 2, (a + c) / 2;
        ab_rotated = (a-ab).rotate_cw_90() + ab;
        ac_rotated = (a-ac).rotate_cw_90() + ac;
        return pt_lines_intersect_ab_to_cd(ab, ab_rotated, ac, ac_rotated);
    }
    
    Tuple<2, Pt2d> pts_line_ab_intersects_circle_cr(Pt2d a, Pt2d b, Pt2d c, double radius) {
        vec_ba, vec_ac = b-a, a-c;
        double dist_sq_ba = vec_ba.dot_product(vec_ba);
        double dist_sq_ac = vec_ac.dot_product(vec_ac);
        double dist_sq_ac_ba = vec_ac.dot_product(vec_ba);
        double dist_sq = dist_sq_ac_ba * dist_sq_ac_ba - dist_ba * dist_ac; 
        int result = compare_ab(dist_sq, 0.0);
        if (result >= 0) {
            auto first_int = c + vec_ac + vec_ba*(-dist_sq_ac_ba + sqrt(dist_sq + EPS))/dist_sq_ba;
            auto second_int = c + vec_ac + vec_ba*(-dist_sq_ac_ba - sqrt(dist_sq))/dist_sq_ba;
            return (result == 0)? {first_int}, {first_int, second_int};
        }
        return {Pt2d(Inf, Inf), Pt2d(Inf, Inf)};
    }

    Tuple<2, Pt2d> pts_two_circles_intersect_cr1_cr2(Pt2d c1, Pt2d c2, double r1, double r2) {
        center_dist = distance_normalized(c1, c2);
        if (compare_ab(center_dist, r1+r2) <= 0 && compare_ab(center_dist + min(r1, r2), max(r1,  r2))) {
            double x = (center_dist * center_dist - r2*r2 + r1*r1)/(2*center_dist);
            double y = sqrt(r1*r1 - x*x);
            Pt2d v = (c2-c1)/center_dist;
            Pt2d pt1, pt2 = x1+v*x, v.rotate_ccw_90() * y;
            return (compare_ab(y, 0.0))? {pt1 , pt2}: {pt1+pt2, pt1-pt2};
        }
        return {Pt2d(Inf, Inf), Pt2d(Inf, Inf)};
    }

    vector<Pt2d> pt_tangent_to_circle_cr(Pt2d centerPoint, double radius, Pt2d pt) {
        Pt2d vec_pc = pt - center_point;
        double x = vec_pc.dot_product(vec_pc);
        double dist_sq = x - radius * radius;
        int result = compare_ab(dist_sq, 0.0);
        if (result >= 0) {
            dist_sq = (result)? dist_sq: 0;
            Pt2d q1 = vec_pc * (radius * radius / x);
            Pt2d q2 = (vec_pc * -radius * sqrt(dist_sq) / x).rotate_ccw_90();
            return {center_point + q1 -q2, center_point + q1 + q2};
        }
        return vector<Pt2d>();
    }

    
    vector<Pt2d> tangents_between_2_circles(Pt2d c1, double r1, Pt2d c2, double r2) {
        vector<Pt2d> r_tangents;
        if (!compare_ab(r1, r2)) {
            c2c1 = c2-c1;
            double multiplier = r1/sqrt(c2c1.dot_product(c2c1));
            tangent = (c2c1*multiplier).rotate_ccw_90();
            r_tangents = {(c1+tangent, c2+tangent), (c1-tangent, c2-tangent)};
        }else {
            Pt2d ref_pt = ((c1 * -r2) + (c2*r1)) / (r1-r2);
            vector<Pt2d> ps = pt_tangent_to_circle_cr(c1, r1, ref_pt);
            vector<Pt2d> qs = pt_tangent_to_circle_cr(c2, r2, ref_pt);
            for (const auto [a, b]: ranges::zip(ps, qs)) // is better way to do this ?;?????
                r_tangents.push_back({a, b});
        }
        Pt2d ref_pt = ((c1 *r2) + (c2*r1)) / (r1+r2);
        Pt2d ps = pt_tangent_to_circle_cr(c1, r1, ref_pt);
        Pt2d qs = pt_tangent_to_circle_cr(c2, r2, ref_pt);
        for (const auto [a, b]: ranges::zip(ps, qs)) // is better way to do this ?;?????
            r_tangents.push_back({a, b});
    }

    tuple<3, double> sides_of_triangle_abc(Pt2d a, Pt2d b, Pt2d c){
        return {distance_normalized(a,b), distance_normalized(b,c), distance_normalized(c,a)};
    }

    bool pt_p_in_trangle_abc(Pt2d a, Pt2d b, Pt2d c, Pt2d p){
        return (point_c_rotation_wrt_line_ab(a, b, p) >= 0
                && point_c_rotation_wrt_line_ab(b, c, p) >= 0
                && point_c_rotation_wrt_line_ab(c, a, p) >= 0);
    }

    double perimeter_of_triangle_abc(double ab, double bc, double ca) {
        return ab+bc+ca;
    }

    double triangle_area_bh(double base, double height){
        return base*height/2;
    }

    double triangle_area_from_heron_abc(double ab, double bc, double ca) {
        double s = perimeter_of_triangle_abc(ab, bc, ca) / 2;
        return sqrt(s * (s - ab) * (s - bc) * (s - ca));
    }

    double triangle_area_from_cross_product_abc(Pt2d a, Pt2d b, Pt2d c){
        return (a.cross_product(b) + b.cross_product(c) + c.cross_product(a))/2;
    }

    double incircle_radius_of_triangle_abc(Pt2d a, Pt2d b, Pt2d c){
        double [ab, bc, ca] = sides_of_triangle_abc(a, b, c);
        double area = triangle_area_from_heron_abc(ab, bc, ca);
        double perimeter = perimeter_of_trangle_abc(ab, bc, ca);
        return area / perimeter;
    }

    double circumcircule_radius_of_triangle_abc(Pt2d a, Pt2d b, Pt2d c) {
        double [ab, bc, ca] = sides_of_triangle_abc(a, b, c);
        double area = triangle_area_from_heron_abc(ab, bc, ca);
        return (ab * bc * ca) / (4 * area);
    }

    tuple<3, double> incircle_pt_for_triangle_abc(Pt2d a, Pt2d b, Pt2d c) {
        double radius = incircle_radius_of_triangle_abc(a, b, c);
        if (compare_ab(radius, 0.0) == 0)
            return {0, 0, 0};
        double [ab, bc, ca] = sides_of_triangle_abc(a, b, c);
        double ratio1 = ab/ca;
        double ratio2 = ab/bc;
        Pt2d pt1 = b + (c-b) * (ratio1 / (ratio1 + 1.0));
        Pt2d pt2 = a + (c-a) * (ratio2 / (ratio2 + 1.0));
        if (is_lines_intersect_ab_cd(a, pt1, b, pt2)) {
            Pt2d intersection_pt pt_lines_intersect_ab_to_cd(a, pt1, b, pt2);
            return {1, radius, round(intersection_pt, 12)};
        }
        return {0, 0, 0};
    }

    Pt2d triangle_circle_center_pt_abcd(Pt2d a, Pt2d b, Pt2d c, Pt2d d) {
        Pt2d pt1 = (b-a).rotate_cw_90();
        Pt2d pt2 = (d-a).rotate_cw_90();
        double cross_prod_1_2 = pt1.cross_product(pt2);
        if (compare_ab(cross_prod_1_2, 0.0) == 0)
            return NULL;  // wtf to do when I used to return None

        Pt2d pt3 = Pt2d(a.dot_product(pt1), c.dot_product(pt2));
        double x = ((pt3.x * pt2.y) - (pt3.y * pt1.y)) / cross_prod_1_2;
        double y = ((pt3.x * pt2.x) - (pt3.y * pt1.x)) / -cross_prod_1_2;
        return round(Pt2d(x, y), 12);
    }

    Pt2d angle_bisector_for_triangle_abc(Pt2d a, Pt2d b, Pt2d c){
        double  dist_ba = distance_normalized(b, a);
        double  dist_ca = distance_normalized(c, a);
        Pt2d ref_pt = (b-a)/dist_ba*dist_ca;
        return ref_pt + (c-a)+a;
    }

};
