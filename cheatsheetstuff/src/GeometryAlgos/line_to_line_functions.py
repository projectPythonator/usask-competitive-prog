from point2d import Pt2d
from geometry_utility_2d import GeometryAlgorithms as basicPointFunctions
from point_to_line_functions import GeometryAlgorithms as pointToLineFunctions
from math import sqrt, acos


class GeometryAlgorithms(basicPointFunctions, pointToLineFunctions):
  def is_parallel_lines_ab_and_cd(self, endpoint_a: Pt2d, endpoint_b: Pt2d,
                                  endpoint_c: Pt2d, endpoint_d: Pt2d) -> bool:  # TODO RETEST
    """Two lines are parallel if the cross_product between vec_ab and vec_dc is 0."""
    vec_ab, vec_dc = endpoint_b - endpoint_a, endpoint_c - endpoint_d
    return self.compare_ab(self.cross_product(vec_ab, vec_dc), 0.0) == 0

  def is_collinear_lines_ab_and_cd_1(self, end_pt_a: Pt2d, end_pt_b: Pt2d,
                                     end_pt_c: Pt2d, end_pt_d: Pt2d) -> bool:  # TODO RETEST
    """Old function. a!=b and c!=d and then returns correctly"""
    return (self.is_parallel_lines_ab_and_cd(end_pt_a, end_pt_b, end_pt_c, end_pt_d)
            and self.is_parallel_lines_ab_and_cd(end_pt_b, end_pt_a, end_pt_a, end_pt_c)
            and self.is_parallel_lines_ab_and_cd(end_pt_d, end_pt_c, end_pt_c, end_pt_a))

  def is_collinear_lines_ab_and_cd_2(self, end_point_a: Pt2d, endpoint_b: Pt2d,
                                     endpoint_c: Pt2d, endpoint_d: Pt2d) -> bool:  # TODO RETEST
    """Two lines are collinear iff a!=b and c!=d, and both c and d are collinear to line ab."""
    return (self.point_c_rotation_wrt_line_ab(end_point_a, endpoint_b, endpoint_c) == 0
            and self.point_c_rotation_wrt_line_ab(end_point_a, endpoint_b, endpoint_d) == 0)

  def is_segments_intersect_ab_to_cd(self, end_pt_a: Pt2d, end_pt_b: Pt2d,
                                     end_pt_c: Pt2d, end_pt_d: Pt2d) -> bool:
    """4 distinct points as two lines intersect if they are collinear and at least one of the
     end points c or d are in between a and b otherwise, need to compute cross products."""
    if self.is_collinear_lines_ab_and_cd_2(end_pt_a, end_pt_b, end_pt_c, end_pt_d):
      lo, hi = (end_pt_a, end_pt_b) if end_pt_a < end_pt_b else (end_pt_b, end_pt_a)
      return lo <= end_pt_c <= hi or lo <= end_pt_d <= hi
    vec_ad, vec_ab, vec_ac = end_pt_d - end_pt_a, end_pt_b - end_pt_a, end_pt_c - end_pt_a
    vec_ca, vec_cd, vec_cb = end_pt_a - end_pt_c, end_pt_d - end_pt_c, end_pt_b - end_pt_c
    point_a_value = self.cross_product(vec_ad, vec_ab) * self.cross_product(vec_ac, vec_ab)
    point_c_value = self.cross_product(vec_ca, vec_cd) * self.cross_product(vec_cb, vec_cd)
    return not (point_a_value > 0 or point_c_value > 0)

  def is_lines_intersect_ab_to_cd(self, end_pt_a: Pt2d, end_pt_b: Pt2d,
                                  end_pt_c: Pt2d, end_pt_d: Pt2d) -> bool:
    """Two lines intersect if they aren't parallel or if they collinear."""
    return (not self.is_parallel_lines_ab_and_cd(end_pt_a, end_pt_b, end_pt_c, end_pt_d)
            or self.is_collinear_lines_ab_and_cd_2(end_pt_a, end_pt_b, end_pt_c, end_pt_d))

  def pt_lines_intersect_ab_to_cd(self, end_pt_a: Pt2d, end_pt_b: Pt2d,
                                  end_pt_c: Pt2d, end_pt_d: Pt2d) -> Pt2d:  # TODO RETEST
    """Compute the intersection point between two lines via cross products of the vectors."""
    vec_ab, vec_ac, vec_dc = end_pt_b - end_pt_a, end_pt_c - end_pt_a, end_pt_c - end_pt_d
    vec_t = vec_ab * (self.cross_product(vec_ac, vec_dc) / self.cross_product(vec_ab, vec_dc))
    return end_pt_a + vec_t

  def pt_line_seg_intersect_ab_to_cd(self, a: Pt2d, b: Pt2d, c: Pt2d, d: Pt2d) -> Pt2d:
    """Same as for line intersect but this time we need to use a specific formula.
    Formula: # TODO RETEST"""
    x, y, cross_prod = c.x - d.x, d.y - c.y, self.cross_product(d, c)
    u = abs(y * a.x + x * a.y + cross_prod)
    v = abs(y * b.x + x * b.y + cross_prod)
    return Pt2d((a.x * v + b.x * u) / (v + u), (a.y * v + b.y * u) / (v + u))
