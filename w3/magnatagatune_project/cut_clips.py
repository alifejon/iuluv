#!/usr/bin/python
#
# Tools to build the magnatagatune dataset.
#
# usage:
#  ./cut_clips.py clip_info.csv clip_info_final.csv
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

"""Script for fetching and cutting the audio files in clip_info.csv."""

import commands
import logging
from optparse import OptionParser
import os
import urllib
import random
import string
import sys
import time
from xml.dom import minidom

import join


def CreateDirs(cut, uncut):
  """Create directories for mp3 files."""
  # Creates the directories for the mp3 file shards.
  for c in list('0123456789abcdef'):
    try:
      os.mkdir(os.path.join(cut, c))
    except:
      pass
  # Creates the directory for the full mp3 files.
  try:
    os.mkdir(uncut)
  except:
    pass


def Canonicalize(parts):
  """Returns a canonical name (lower case, no funky chars)."""
  in_chr = ''.join(map(chr, range(256)))
  out_chr = range(256)
  for i in range(256):
    if chr(i) in string.uppercase:
      out_chr[i] = ord(chr(i).lower())
    elif chr(i) in string.lowercase:
      out_chr[i] = i
    elif chr(i) in string.digits:
      out_chr[i] = i
    else:
      out_chr[i] = ord('_')
  table = string.maketrans(in_chr, ''.join(map(chr, out_chr)))
  bad_chars = '\t\n\r-_:()[]"\',;+'
  return '-'.join([x.strip().translate(table, bad_chars) for x in parts])


def FetchUrl(url):
  """Fetches the content of a url."""
  logging.info('Fetching %s...' % url)
  f = urllib.urlopen(url)
  s = f.read()
  f.close()
  return s


def CutClip(source, start, end, destination, song, artist, album, url, options):
  """Cuts and encodes a segment of a mp3 file."""
  comment = 'This excerpt (range [%d, %d] from the original) is for ' + \
    'research purposes only. It is part of the tagatune dataset: ' + \
    'http://tagatune.org/Datasets.html - The original track is released ' + \
    'under a Creative Commons v1.0 by-nc-sa license ' + \
    'http://creativecommons.org/licenses/by-nc-sa/1.0/ by Magnatune.com ' + \
    'and is available here for purchase and commercial licensing: %s'
  comment = comment % (start, end, url)
  tags = {
    '--tt': song.encode('string-escape') + ' (EXCERPT)',
    '--ta': artist.encode('string-escape'),
    '--tl': album.encode('string-escape'),
    '--tc': comment.encode('string-escape'),
  }
  tags_flags = ' '.join(['%s "%s"' % kv for kv in tags.items()])
  lame_args = (options.lame_path, tags_flags, options.lame_flags, destination)
  lame_command = '%s %s %s tmp.wav %s' % lame_args
  sox_args = (options.sox_path, source, start, end - start)
  sox_command = '%s %s tmp.wav trim %d %d' % sox_args
  error, output = commands.getstatusoutput(sox_command)
  if not error:
    error, output = commands.getstatusoutput(lame_command)
  if error:
    logging.error('Cut error: %s', output)
    raise RuntimeError('Encoding error')
  return destination


def LoadMagnatuneCatalog(path):
  """Gets a dictionary mapping an (album sku, song title) pair to a mp3 url."""
  logging.info('Loading catalog XML...')
  xml = file(path).read()
  # Fix 3 invalid characters in Magnatune's XML feed.
  xml = xml.translate(''.join([chr(i) for i in xrange(256)]), '\x19\x13')
  doc = minidom.parseString(xml)
  tracks = {}
  for track in doc.getElementsByTagName('Track'):
    # A join using artist / album as a key is not always possible because some
    # artist names appear differently in both datasets, for example:
    # St. Eliyah Childrens Choir vs Saint Elijah Childrens Choir.
    # The album sku is more reliable.
    title = Canonicalize([track.getAttribute('title').encode('utf-8')])
    album_sku = track.getAttribute('albumsku').encode('utf-8')
    url = track.getAttribute('url').encode('utf-8')
    try:
      track_number = int(os.path.split(url)[-1][:2])
    except:
      logging.info('Cannot parse track number: %s' % url)
      continue
    tracks.setdefault((album_sku, track_number, title), []).append(url)

  # Do not include in the catalog tracks that map to more than 1 mp3 url.
  deduped_tracks = {}
  for k, v in tracks.items():
    if len(v) > 1:
      logging.error('Duplicate entry found for key: %s', str(k))
    else:
      deduped_tracks[k] = v[0]
  return deduped_tracks


def Retry(callback, max_num_retries):
  """Invokes callback max_num_retries until it succeeds."""
  num_failures = 0
  while num_failures < max_num_retries:
    try:
      return callback()
    except KeyboardInterrupt:
      raise KeyboardInterrupt()
    except:
      logging.error('Failure in %s' % str(callback))
      num_failures += 1
      time.sleep(1 << num_failures)  # Exponential backoff
  return False


def CutClips(source, target, catalog, options):
  """Cuts clips and adds url/mp3 path to the CSV file."""
  def DownloadAndCut(row):
    title = Canonicalize([row[2]])
    track_number = int(row[1])
    album_sku = row[5].split('/')[-2]
    # Uses the last nibble of the hash of the album_sku as a shard index.
    shard = ('%x' % hash(album_sku))[-1]
    if options.shards and not shard in options.shards:
      return None
    url = catalog.get((album_sku, track_number, title), '')
    if not url:
      logging.info('No matching url for track %s' % '-'.join(row[1:4]))
      return ['', '']
    mp3_name = Canonicalize([row[3], row[4], '%02d' % track_number, row[2]])
    mp3_path = os.path.join(options.uncut_mp3_path, mp3_name + '.mp3')

    start = int(row[6])
    end = int(row[7])
    suffix = '-%d-%d' % (start, end)
    cut_path = os.path.join(options.cut_mp3_path,
                            shard, mp3_name + suffix + '.mp3')

    if not os.path.exists(mp3_path) and not os.path.exists(cut_path):
      def Fetch():
        return FetchUrl(url)
      content = Retry(Fetch, options.max_num_retries)
      if not content:
        logging.error('Could not download %s' % url)
        return [url, '']
      else:
        file(mp3_path, 'w').write(content)
        time.sleep(random.randint(options.sleep  / 2, options.sleep * 3 / 2))

    if not os.path.exists(cut_path):
      def Cut():
        return CutClip(mp3_path, start, end, cut_path, row[0], row[1],
                       row[2], row[3], options)
      target = Retry(Cut, options.max_num_retries)
      if not target:
        logging.error('Could not cut %s' % mp3_path)
        return [url, '']

    logging.info('Cut: %s cut to %s' % (mp3_path, cut_path))
    return [url, os.path.join(shard, mp3_name + suffix + '.mp3')]

  join.LeftJoin(source, target, ['original_url', 'mp3_path'],
                DownloadAndCut, limit=options.limit)


if __name__ == '__main__':
  LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

  parser = OptionParser()
  parser.add_option('-c', '--cut_mp3_path', dest='cut_mp3_path',
                    default='/mnt/magnatagatune/mp3',
                    help='write cut mp3 files to PATH', metavar='PATH')
  parser.add_option('-u', '--uncut_mp3_path', dest='uncut_mp3_path',
                    default='/tmp/mp3',
                    help='write original mp3 files to PATH', metavar='PATH')
  parser.add_option('-i', '--song_info', dest='song_info',
                    default='data/song_info2_xml',
                    help='Path to the tagatune catalog XML file')
  parser.add_option('-s', '--sleep', dest='sleep', type=int,
                    default=5,
                    help='sleep N seconds between downloads', metavar='N')
  parser.add_option('-x', '--sox_path', dest='sox_path',
                    default='/usr/local/bin/sox',
                    help='path to sox binary')
  parser.add_option('-l', '--lame_path', dest='lame_path',
                    default='/usr/local/bin/lame',
                    help='path to lame binary')
  parser.add_option('-f', '--lame_flags', dest='lame_flags',
                    default=' -m m -b 32 --resample 16 -o ',
                    help='flags used for mp3 encoding')
  parser.add_option('-n', '--max_num_retries', dest='max_num_retries',
                    type=int, default=8,
                    help='maximum number of retries before ignoring')
  parser.add_option('-d', '--shards', dest='shards',
                    default='',
                    help='shards to generate, empty for all shards')
  parser.add_option('-L', '--limit', dest='limit',
                    type=int, default=0,
                    help='use only the first N rows of the input', metavar='N')

  options, args = parser.parse_args()
  if len(args) < 1:
    args.append('data/clip_info.csv')
  if len(args) < 2:
    default_name = '/mnt/magnatagatune/clip_info_final.csv'
    if options.shards:
      default_name = '%s.shards_%s' % (default_name, options.shards)
    if options.limit:
      default_name = '%s.limit_%d' % (default_name, options.limit)
    args.append(default_name)

  CreateDirs(options.cut_mp3_path, options.uncut_mp3_path)
  catalog = LoadMagnatuneCatalog(options.song_info)
  CutClips(args[0], args[1], catalog, options)
