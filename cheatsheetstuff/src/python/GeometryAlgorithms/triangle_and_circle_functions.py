from point2d import Pt2d, Numeric, EPS
from geometry_utility_2d import GeometryAlgorithms as basicPointFunctions
from point_to_line_functions import GeometryAlgorithms as pointToLineFunctions
from typing import List
from math import sqrt


class GeometryAlgorithms(basicPointFunctions, pointToLineFunctions):
  def is_point_in_radius_of_circle(self, point: Pt2d, center_point: Pt2d, radius: float) -> bool:
    """True if point p in circle False otherwise. Use <= for circumference inclusion."""
    return self.compare_ab(self.distance_normalized(point, center_point), radius) < 0

  def pt_circle_center_given_pt_abc(self, a_point: Pt2d, b_point: Pt2d, c_point: Pt2d) -> Pt2d:
    """Find the center of a circle based of 3 distinct points
    TODO add in the formula
    =# TODO RETEST
    """
    ab, ac = (a_point + b_point) / 2, (a_point + c_point) / 2
    ab_rotated: Pt2d = self.rotate_cw_90_wrt_origin(a_point - ab) + ab
    ac_rotated: Pt2d = self.rotate_cw_90_wrt_origin(a_point - ac) + ac
    return self.pt_lines_intersect_ab_to_cd(ab, ab_rotated, ac, ac_rotated)

  def pts_line_ab_intersects_circle_cr(self, a, b, c, r):  # TODO RETEST
    """Compute the point(s) that line ab intersects circle c radius r. from stanford 2016
    TODO add in the formula
    """
    vec_ba, vec_ac = b-a, a-c
    dist_sq_ba = self.dot_product(vec_ba, vec_ba)
    dist_sq_ac_ba = self.dot_product(vec_ac, vec_ba)
    dist_sq_ac = self.dot_product(vec_ac, vec_ac) - r * r
    dist_sq = dist_sq_ac_ba*dist_sq_ac_ba - dist_sq_ba*dist_sq_ac
    result = self.compare_ab(dist_sq, 0.0)
    if result >= 0:
      first_intersect = c + vec_ac + vec_ba*(-dist_sq_ac_ba + sqrt(dist_sq + EPS))/dist_sq_ba
      second_intersect = c + vec_ac + vec_ba*(-dist_sq_ac_ba - sqrt(dist_sq))/dist_sq_ba
      return first_intersect if result == 0 else first_intersect, second_intersect
    return None  # no intersect

  def pts_two_circles_intersect_cr1_cr2(self, c1: Pt2d, c2: Pt2d, r1, r2):  # TODO RETEST
    """I think this is the points on the circumference but not fully sure. from stanford 2016
    TODO add in teh formula
    """
    center_dist = self.distance_normalized(c1, c2)
    if (self.compare_ab(center_dist, r1 + r2) <= 0 <= self.compare_ab(center_dist + min(r1, r2),
                                                                      max(r1, r2))):
      x = (center_dist*center_dist - r2*r2 + r1*r1)/(2*center_dist)
      y = sqrt(r1*r1 - x*x)
      v = (c2-c1)/center_dist
      pt1, pt2 = c1 + v * x, self.rotate_ccw_90_wrt_origin(v) * y
      return (pt1 + pt2) if self.compare_ab(y, 0.0) <= 0 else (pt1+pt2, pt1-pt2)
    return None  # no overlap

  def pt_tangent_to_circle_cr(self, center_point: Pt2d, radius: Numeric, pt: Pt2d) -> List[Pt2d]:
    """Find the two points that create tangent lines from p to the circumference.
    TODO add in teh formula
    # TODO RETEST
    """
    vec_pc = pt - center_point
    x = self.dot_product(vec_pc, vec_pc)
    dist_sq = x - radius * radius
    result = self.compare_ab(dist_sq, 0.0)
    if result >= 0:
      dist_sq = dist_sq if result else 0
      q1 = vec_pc * (radius * radius / x)
      q2 = self.rotate_ccw_90_wrt_origin(vec_pc * (-radius * sqrt(dist_sq) / x))
      return [center_point + q1 - q2, center_point + q1 + q2]
    return []

  def tangents_between_2_circles(self, c1, r1, c2, r2):  # TODO RETEST
    """Between two circles there should be at least 4 points that make two tangent lines.
    TODO add in teh formula
    """
    if self.compare_ab(r1, r2) == 0:
      c2c1 = c2 - c1
      multiplier = r1/sqrt(self.dot_product(c2c1, c2c1))
      tangent = self.rotate_ccw_90_wrt_origin(c2c1 * multiplier)  # need better name
      r_tangents = [(c1+tangent, c2+tangent), (c1-tangent, c2-tangent)]
    else:
      ref_pt = ((c1 * -r2) + (c2 * r1)) / (r1 - r2)
      ps = self.pt_tangent_to_circle_cr(c1, r1, ref_pt)
      qs = self.pt_tangent_to_circle_cr(c2, r2, ref_pt)
      r_tangents = [(ps[i], qs[i]) for i in range(min(len(ps), len(qs)))]
    ref_pt = ((c1 * r2) + (c2 * r1)) / (r1 + r2)
    ps = self.pt_tangent_to_circle_cr(c1, r1, ref_pt)
    qs = self.pt_tangent_to_circle_cr(c2, r2, ref_pt)
    for i in range(min(len(ps), len(qs))):
      r_tangents.append((ps[i], qs[i]))
    return r_tangents

  def sides_of_triangle_abc(self, a, b, c):  # TODO RETEST
    """Compute the side lengths of a triangle."""
    dist_ab = self.distance_normalized(a, b)
    dist_bc = self.distance_normalized(b, c)
    dist_ca = self.distance_normalized(c, a)
    return dist_ab, dist_bc, dist_ca

  def pt_p_in_triangle_abc(self, a, b, c, p):  # TODO RETEST
    """Compute if a point is in or on a triangle. If all edges return the same orientation this
    should return true and the point should be in or on the triangle."""
    return (self.point_c_rotation_wrt_line_ab(a, b, p) >= 0
            and self.point_c_rotation_wrt_line_ab(b, c, p) >= 0
            and self.point_c_rotation_wrt_line_ab(c, a, p) >= 0)

  def perimeter_of_triangle_abc(self, side_ab, side_bc, side_ca):  # TODO RETEST
    """Computes the perimeter of triangle given the side lengths."""
    return side_ab + side_bc + side_ca

  def triangle_area_bh(self, base, height):  # TODO RETEST
    """Simple triangle area formula: area = b*h/2."""
    return base*height/2

  def triangle_area_from_heron_abc(self, side_ab, side_bc, side_ca):  # TODO RETEST
    """Compute heron's formula which gives us the area of a triangle given the side lengths."""
    s = self.perimeter_of_triangle_abc(side_ab, side_bc, side_ca) / 2
    return sqrt(s * (s-side_ab) * (s-side_bc) * (s-side_ca))

  def triangle_area_from_cross_product_abc(self, a, b, c):  # TODO RETEST
    """Compute triangle area, via cross-products of the pairwise sides ab, bc, ca."""
    return (self.cross_product(a, b) + self.cross_product(b, c) + self.cross_product(c, a))/2

  # def incircle_radis_of_triangle_abc_helper(self, ab, bc, ca):
  #   area = self.triangle_area_from_heron_abc(ab, bc, ca)
  #   perimeter = self.perimeter_of_triangle_abc(ab, bc, ca) / 2
  #   return area/perimeter

  def incircle_radius_of_triangle_abc(self, a, b, c):  # TODO RETEST
    """Computes the radius of the incircle, achieved by computing the side lengths then finding
    the area and perimeter to use in this Formula: r = area/(perimeter/2) Author: TODO Author
    """
    side_ab, side_bc, side_ca = self.sides_of_triangle_abc(a, b, c)
    area = self.triangle_area_from_heron_abc(side_ab, side_bc, side_ca)
    perimeter = self.perimeter_of_triangle_abc(side_ab, side_bc, side_ca) / 2
    return area / perimeter

  # def circumcircle_radis_of_triangle_abc_helper(self, ab, bc, ca):
  #   area = self.triangle_area_from_heron_abc(ab, bc, ca)
  #   return (ab*bc*ca) / (4*area)

  def circumcircle_radius_of_triangle_abc(self, a, b, c):  # TODO RETEST
    """Computes the radius of the circum-circle, achieved by computing the side lengths then
    gets the area for Formula: r = (ab * bc * ca) / (4 * area) Author: TODO Author
    """
    side_ab, side_bc, side_ca = self.sides_of_triangle_abc(a, b, c)
    area = self.triangle_area_from_heron_abc(side_ab, side_bc, side_ca)
    return (side_ab * side_bc * side_ca) / (4 * area)

  def incircle_pt_for_triangle_abc_1(self, a, b, c):  # TODO RETEST
    """Get the circle center of an incircle.

    Complexity per call: Time: lots of ops but still O(1), Space O(1)
    Formula: TODO add in the formula
    Optimization: get sides individually instead of through another call
    """
    radius = self.incircle_radius_of_triangle_abc(a, b, c)
    if self.compare_ab(radius, 0.0) == 0:  # if the radius was 0 we don't have a point
      return False, 0, 0
    side_ab, side_bc, side_ca = self.sides_of_triangle_abc(a, b, c)
    ratio_1 = side_ab/side_ca
    ratio_2 = side_ab/side_bc
    pt_1 = b + (c-b) * (ratio_1/(ratio_1 + 1.0))
    pt_2 = a + (c-a) * (ratio_2/(ratio_2 + 1.0))

    if self.is_lines_intersect_ab_to_cd(a, pt_1, b, pt_2):
      intersection_pt = self.pt_lines_intersect_ab_to_cd(a, pt_1, b, pt_2)
      return True, radius, round(intersection_pt, 12)  # can remove the round function
    return False, 0, 0

  def triangle_circle_center_pt_abcd(self, a, b, c, d):  # TODO RETEST
    """A 2 in one method that can get the middle point of both incircle circumcenter.
    Method: TODO add in the formula

    Complexity per call: Time: lots of ops but still O(1), Space O(1)
    Optimization: paste rotation code instead of function call
    """
    pt_1 = self.rotate_cw_90_wrt_origin(b-a)  # rotation on the vector b-a
    pt_2 = self.rotate_cw_90_wrt_origin(d-c)  # rotation on the vector d-c
    cross_product_1_2 = self.cross_product(pt_1, pt_2)
    # cross_product_2_1 = -cross_product_1_2  # self.cross_product(pt_2, pt_1)
    if self.compare_ab(cross_product_1_2, 0.0) == 0:
      return None
    pt_3 = Pt2d(self.dot_product(a, pt_1), self.dot_product(c, pt_2))
    x = ((pt_3.x * pt_2.y) - (pt_3.y * pt_1.y)) / cross_product_1_2
    y = ((pt_3.x * pt_2.x) - (pt_3.y * pt_1.x)) / -cross_product_1_2  # cross(pt_2, pt_1)
    return round(Pt2d(x, y), 12)

  def angle_bisector_for_triangle_abc(self, a, b, c):  # TODO RETEST
    """Compute the angle bisector point.
    Method: TODO add in the formula
    """
    dist_ba = self.distance_normalized(b, a)
    dist_ca = self.distance_normalized(c, a)
    ref_pt = (b-a) / dist_ba * dist_ca
    return ref_pt + (c-a) + a

  def perpendicular_bisector_for_triangle_ab(self, a, b):  # TODO RETEST
    """Compute the perpendicular bisector point.
    Method: TODO add in the formula
    """
    rotated_vector_ba = self.rotate_ccw_90_wrt_origin(b-a)  # code is a ccw turn. check formula
    return rotated_vector_ba + (a+b)/2

  def incircle_pt_for_triangle_abc_2(self, a, b, c):  # TODO RETEST
    """An alternative way to compute incircle. This one uses bisectors
    Method: TODO add in the formula
    """
    bisector_abc = self.angle_bisector_for_triangle_abc(a, b, c)
    bisector_bca = self.angle_bisector_for_triangle_abc(b, c, a)
    return self.triangle_circle_center_pt_abcd(a, bisector_abc, b, bisector_bca)

  def circumcenter_pt_of_triangle_abc_2(self, a, b, c):  # TODO RETEST
    """An alternative way to compute circumcenter. This one uses bisectors
    Method: TODO add in the formula
    """
    bisector_ab = self.perpendicular_bisector_for_triangle_ab(a, b)
    bisector_bc = self.perpendicular_bisector_for_triangle_ab(b, c)
    ab2, bc2 = (a+b)/2, (b+c)/2
    return self.triangle_circle_center_pt_abcd(ab2, bisector_ab, bc2, bisector_bc)

  def orthocenter_pt_of_triangle_abc_v2(self, a, b, c):  # TODO RETEST
    """Compute the orthogonal center of triangle abc.Z
    Method: TODO add in the formula
    """
    return a + b + c - self.circumcenter_pt_of_triangle_abc_2(a, b, c) * 2
