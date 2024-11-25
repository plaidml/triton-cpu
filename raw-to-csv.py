#!/usr/bin/env python

import sys
import re

def convert_all(in_file):
  metadata = None
  csv_lines = []

  for line in in_file:
    if line.startswith("RUN") or line.startswith("BENCHMARK"):
      # If this is not the first RUN line, yield the previous bundle
      if metadata and csv_lines:
        yield (csv_lines, metadata)
        metadata = None
        csv_lines = []
      metadata = {'line': line}
    if m := re.match(r" +M.*", line): # header row
      csv_line = re.subn("[, ][, ]+", ",", line)
      l = csv_line[0][1:].split(",")
      k_onwards = l[2:]
      csv_lines += [",".join(k_onwards)]
    if m := re.match(r"\d+ +(\d.*)", line): # data row
      csv_line = re.subn("[, ]+", ",", m[1])
      l = csv_line[0][1:].split(",")
      k_onwards = l[2:]
      csv_lines += [",".join(k_onwards)+"\n"]

  # The last bundle gets yielded here
  if metadata and csv_lines:
    yield (csv_lines, metadata)


if __name__ == "__main__":
  in_file = sys.stdin if (len(sys.argv) < 2 or sys.argv[1] == '-') else open(sys.argv[1])

  converted = convert_all(in_file)

  for csv_lines, metadata in converted:
    print(metadata['line'], end='')
    for csv_line in csv_lines:
      print(csv_line, end='')
