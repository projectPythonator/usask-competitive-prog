from point2d import Pt2d, ClosestPair
from geometry_utility_2d import GeometryAlgorithms as basicPointFunctions
from typing import List
from itertools import combinations


class GeometryAlgorithms(basicPointFunctions):
  def closest_pair_helper(self, lo: int, hi: int, x_ord: List[Pt2d]) -> ClosestPair:
    """brute force function, for small range will brute force find the closet pair. O(n^2)"""
    closest_pair = (self.distance(x_ord[lo], x_ord[lo + 1]), x_ord[lo], x_ord[lo + 1])
    for pt_a, pt_b in combinations(x_ord[lo:hi], 2):  # 2 for every pair of combinations
      distance_ij = self.distance(pt_a, pt_b)
      if distance_ij < closest_pair[0]:
        closest_pair = (distance_ij, pt_a, pt_b)
    return closest_pair

  def closest_pair_recursive(self, lo: int, hi: int,
                             x_ord: List[Pt2d], y_ord: List[Pt2d]) -> ClosestPair:
    """Recursive part of computing the closest pair. Divide by y recurse then do a special check

    Complexity per call T(n/2) halves each time, T(n/2) halves each call, O(n) at max tho
    Optimizations and notes: If not working use y_part_left and y_part_right again. I did add in
    the optimization of using y_partition 3 times over rather than having 3 separate lists
    also can remove compare_ab for direct compare
    """
    n = hi - lo
    if n < 32:  # base case, just brute force: powers of 2 between with 32 working the fastest.
      return self.closest_pair_helper(lo, hi, x_ord)
    left_len, right_len = lo + n - n // 2, lo + n // 2
    mid = round((x_ord[left_len].x + x_ord[right_len].x) / 2)
    partition_left, partition_right = [], []
    append_left, append_right = partition_left.append, partition_right.append
    for pt in y_ord:
      append_right(pt) if pt.x > mid else append_left(pt)
    best_left = self.closest_pair_recursive(lo, left_len, x_ord, partition_left)
    best_right = self.closest_pair_recursive(left_len, hi, x_ord, partition_right)
    best_pair = best_left if best_left[0] <= best_right[0] else best_right
    if self.compare_ab(best_pair[0], 0) == 0:
      return best_pair
    y_partition = [pt for pt in y_ord if best_pair[0] > (pt.x - mid) ** 2]
    for i, pt_a in enumerate(y_partition):
      a_y = pt_a.y
      for pt_b in y_partition[i+1:]:  # slicing seemed to run the fastest, range was second.
        dist_ij = a_y - pt_b.y
        if dist_ij ** 2 >= best_pair[0]:
          break
        dist_ij = self.distance(pt_a, pt_b)
        if dist_ij < best_pair[0]:
          best_pair = (dist_ij, pt_a, pt_b)
    return best_pair

  def compute_closest_pair(self, pts: List[Pt2d]) -> ClosestPair:
    """Compute the closest pair of points in a set of points. method is divide and conqur

    Complexity per call Time: O(nlog n), Space O(nlog n)
    Optimizations: use c++ if too much memory, haven't found the way to do it without nlog n
    """
    x_ord = sorted(pts, key=lambda pt: pt.x)
    y_ord = sorted(pts, key=lambda pt: pt.y)
    return self.closest_pair_recursive(0, len(pts), x_ord, y_ord)
