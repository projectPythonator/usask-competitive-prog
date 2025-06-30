from point2d import Pt2d, CW
from geometry_utility_2d import GeometryAlgorithms as basicPointFunctions
from point_to_line_functions import GeometryAlgorithms as pointToLineFunctions
from typing import List


class GeometryAlgorithms(basicPointFunctions, pointToLineFunctions):
  def convex_hull_monotone_chain(self, pts: List[Pt2d]) -> List[Pt2d]:
    """Compute convex hull of a list of points via Monotone Chain method. CCW ordering returned.

    Complexity per call: Time: O(nlog n), Space: final O(n), aux O(nlog n)
    Optimizations: can use heapsort for Space: O(n)[for set + heap] or O(1) [if we consume pts],
    Can also optimize out the append and pop with using a stack like index.
    """
    def func(points, cur_hull, min_size):
      for p in points:
        while (len(cur_hull) > min_size
               and self.point_c_rotation_wrt_line_ab(cur_hull[-2], cur_hull[-1], p) == CW):
          cur_hull.pop()
        cur_hull.append(p)
      cur_hull.pop()
    unique_points, convex_hull = sorted(set(pts)), []
    if len(unique_points) > 1:
      func(unique_points, convex_hull, 1)
      func(unique_points[::-1], convex_hull, 1 + len(convex_hull))
      return convex_hull + [convex_hull[0]]  # add the first point because lots of our methods
    return unique_points             # require the first and last point be the same
