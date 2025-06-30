#tested on following kattis problems given by ids
# amoebas, countingstars, island(modified), coast(modified)
dr=[1,0,-1,0]
dc=[0,1,0,-1]
def ff(r, c, c1, c2):
  global R, C, grid, dc, dr
  stk=[(r, c)]
  while stk:
    r,c=stk.pop()
    if c<0 or r<0 or r>=R or c>=C:
      continue
    if grid[r][c]!=c1:
      continue
    grid[r][c]=c2
    for i in range(4):
      stk.append((r+dr[i],c+dc[i]))
