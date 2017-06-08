# Tools to build the magnatagatune dataset.
# 
# Copyright (C) 2009 Olivier Gillet (ol.gillet@gmail.com).
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Helper function for joining csv files."""

import csv


def LeftJoin(source, target, extra_columns_names, callback,
             limit=0, delimiter='\t'):
  """Helper function for left joins.
  
  Invokes callback on every row of source. If callback returns data,
  add this data to the row. If not, discard the entire row. Output is written
  to target."""
  header = True
  output = csv.writer(file(target, 'wb'), delimiter=delimiter,
                      quoting=csv.QUOTE_NONNUMERIC)
  for count, row in enumerate(csv.reader(file(source, 'rb'),
                                         delimiter=delimiter)):
    if limit and count >= limit:
      break
    if not row[-1]:
      row = row[:-1]
    for i, element in enumerate(row):
      row[i] = element.strip()
    if header:
      row.extend(extra_columns_names)
      header = False
    else:
      extra = callback(row)
      if extra is None:
        row = None
      else:
        row.extend(extra)
    if row:
      output.writerow(row)
