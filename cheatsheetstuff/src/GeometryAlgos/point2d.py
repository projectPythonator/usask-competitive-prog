from math import isclose


class Pt2d_int:  # need to think when to use this
  __slots__ = ("x", "y")
  def __init__(self, x_val, y_val): self.x, self.y = x_val, y_val
  def __add__(self, other): return Pt2d_int(self.x + other.x, self.y + other.y)
  def __sub__(self, other): return Pt2d_int(self.x - other.x, self.y - other.y)
  def __mul__(self, scale): return Pt2d_int(self.x * scale, self.y * scale)
  def __floordiv__(self, scale): return Pt2d_int(self.x // scale, self.y // scale)
  def __eq__(self, other): return self.x == other.x and self.y == other.y
  def __lt__(self, other): return self.x < other.x if self.x != other.x else self.y < other.y
  def __str__(self): return "{} {}".format(self.x, self.y)
  def __hash__(self): return hash((self.x, self.y))
  def get_tup(self): return self.x, self.y


class Pt2d:  # float default version
  __slots__ = ("x", "y")
  def __init__(self, x_val, y_val): self.x, self.y = map(float, (x_val, y_val))
  def __add__(self, other): return Pt2d(self.x + other.x, self.y + other.y)
  def __sub__(self, other): return Pt2d(self.x - other.x, self.y - other.y)
  def __mul__(self, scale): return Pt2d(self.x * scale, self.y * scale)
  def __truediv__(self, scale): return Pt2d(self.x / scale, self.y / scale)
  def __floordiv__(self, scale): return Pt2d(self.x // scale, self.y // scale)
  def __eq__(self, other): return isclose(self.x, other.x) and isclose(self.y, other.y)
  def __lt__(self, other): return self.y < other.y if isclose(self.x, other.x) else self.x < other.x
  def __str__(self): return "{} {}".format(self.x, self.y)
  def __hash__(self): return hash((self.x, self.y))
  def get_tup(self): return self.x, self.y
