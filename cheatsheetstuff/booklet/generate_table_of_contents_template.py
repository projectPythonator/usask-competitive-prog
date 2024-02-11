

def main():
  contents = []
  with open('booklet.py') as f:
    for i, line in enumerate(f.readlines()):
      tmp = line.strip()
      tmp = tmp.rstrip()
      if 'def' in tmp and '(self' in tmp and '__' not in tmp:
        contents.append((1, tmp, i))
      elif 'class' in tmp[:5]:
        contents.append((0, tmp, i))
  out = []
  max_outline = 0
  mod1 = 79
  mod2 = mod1*2
  for i, (typ, line, line_num) in enumerate(contents):
    if typ == 1:
      line = line[line.find('def ')+4:line.find('(self')]
      line = ' '+line
    else:
      line = line[6:line.find(':')]
    line = line + ':'
    begin = '{}'.format(line_num//mod2 + 1)
    end = ('end' if i+1 == len(contents) else str(contents[i+1][2]//mod2 + 1))
    both = '{}-{}'.format(begin, end)
    out_line = '{}:{}'.format(line.ljust(35, '-' if typ == 1 else ' '), both.ljust(5))
    out.append(out_line)
  with open('table_of_contents.txt',mode='w') as f:
    f.write('\n'.join(out))
  output = []
  for i in range(len(out)//2):
    output.append('{} {}'.format(out[i].ljust(59), out[i+len(out)//2].ljust(59)))
  if len(out)%2 == 1:
    output.append(out[-1].rjust(60))
  for line in output:
    max_outline = max(max_outline, len(line))
    print(line)
  print(len(contents))
  print(max_outline)
main()