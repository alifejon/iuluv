#!/usr/bin/python
#
# Tools to build the magnatagatune dataset.
#
# usage:
#   ./join_with_clip_info.py [-j|-f] -i -a clip_info_final.csv \
#       annotations.csv \
#       annotations_final.csv
#
# OR
#   ./join_with_clip_info.py [-j|-f] -i -c clip_info_final.csv \
#       comparisons.csv \
#       comparisons_final.csv
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

"""Script for joining the raw annotation/comparisons CSV with the clip info."""

import csv
import join
import logging
from optparse import OptionParser
import os


def JoinComparisons(source, target, clip_info, options):
  """Removes entries in the comparison file with no matching clip."""
  def JoinWithClipInfo(row):
    id_1 = row[0]
    id_2 = row[1]
    id_3 = row[2]
    if id_1 in clip_info and id_2 in clip_info and id_3 in clip_info:
      if options.join:
        return [clip_info.get(id_1), clip_info.get(id_2), clip_info.get(id_3)]
      else:
        return []
    else:
      return None

  extra_columns = []
  if options.join:
    extra_columns = ['clip1_mp3_path', 'clip2_mp3_path', 'clip3_mp3_path']
  join.LeftJoin(source, target, extra_columns, JoinWithClipInfo)


def JoinAnnotations(source, target, clip_info, options):
  """Removes entries in the annotation file with no matching clip."""
  def JoinWithClipInfo(row):
    clip_id = row[0]
    if clip_id in clip_info:
      if options.join:
        return [clip_info.get(clip_id)]
      else:
        return []
    else:
      return None

  extra_columns = []
  if options.join:
    extra_columns = ['mp3_path']
  join.LeftJoin(source, target, extra_columns, JoinWithClipInfo)


def LoadClipInfo(path):
  """Loads the clip information."""
  clip_info = {}
  for row in csv.reader(file(path), delimiter='\t'):
    if row[-1]:
      clip_info[row[0]] = row[-1]
  return clip_info


if __name__ == '__main__':
  LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

  parser = OptionParser()
  parser.add_option('-a', '--annotations', dest='is_annotation',
                    action='store_true',
                    default=True,
                    help='whether the file to process contains annotations')
  parser.add_option('-c', '--comparisons', dest='is_annotation',
                    action='store_false',
                    default=True,
                    help='whether the file to process contains comparisons')
  parser.add_option('-i', '--clip_info', dest='clip_info',
                    default='/mnt/magnatagatune/clip_info_final.csv',
                    help='path to final clip information file')
  parser.add_option('-j', '--join', dest='join', action='store_true',
                    default=True,
                    help='add mp3 path to CSV file (join)')
  parser.add_option('-f', '--filter', dest='join', action='store_false',
                    default=True,
                    help='only filter rows with no matching clip')

  options, args = parser.parse_args()
  if options.is_annotation:
    default_name = 'annotations'
    join_function = JoinAnnotations
  else:
    default_name = 'comparisons'
    join_function = JoinComparisons
  if len(args) < 1:
    args.append('data/%s.csv' % default_name)
  if len(args) < 2:
    args.append('/mnt/magnatagatune/%s_final.csv' % default_name)
  clip_info = LoadClipInfo(options.clip_info)
  join_function(args[0], args[1], clip_info, options)
