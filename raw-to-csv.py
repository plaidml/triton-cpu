#!/usr/bin/env python

import io
import sys
import re
from pprint import pprint

def convert_all(in_file):
  metadata = None
  csv_lines = []
  count_new_line_sep = 0
  for line in in_file:
    if line.startswith('sys '):
      if metadata:
          yield (csv_lines, metadata)
          metadata = None
          csv_lines = []
    if line in {"", "\n"}:
      count_new_line_sep += 1
      if count_new_line_sep >= 2:
        if metadata:
            yield (csv_lines, metadata)
            metadata = None
            csv_lines = []
            count_new_line_sep = 0
    if line.startswith("RUN") or line.startswith("BENCHMARK"):
      metadata = {'line': line}
    if m := re.match(r" +M.*", line): # header row
      csv_line = re.subn("[, ][, ]+", ",", line)
      csv_lines += [csv_line[0][1:]]
    if m := re.match(r"\d+ +(\d.*)", line): # data row
      csv_line = re.subn("[, ]+", ",", m[1])
      csv_lines += [csv_line[0]+"\n"]


if __name__ == "__main__":
  in_file = sys.stdin if (len(sys.argv) < 2 or sys.argv[1] == '-') else open(sys.argv[1])

  converted = convert_all(in_file)

  for csv_lines, metadata in converted:
    print(metadata['line'], end='')
    for csv_line in csv_lines:
      print(csv_line, end='') # csv_line includes its own newline


