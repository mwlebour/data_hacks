#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2010 Bitly
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Generate a text format histogram

This is a loose port to python of the Perl version at
http://www.pandamatak.com/people/anand/xfer/histo

https://github.com/bitly/data_hacks
"""

import sys
from decimal import Decimal
import logging
import math
from optparse import OptionParser
from collections import namedtuple

def first_bucket_size(k, n):
    """Logarithmic buckets means, the size of bucket i+1 is twice
    the size of bucket i.
    For k+1 buckets whose sum is n, we have
    (note, k+1 buckets, since 0 is counted as well):
        \sum_{i=0}^{k} x*2^i   = n
        x * \sum_{i=0}^{k} 2^i = n
        x * (2^{k+1} - 1)      = n
        x = n/(2^{k+1} - 1)
    """
    return n/(2**(k+1)-1)

def log_steps(k, n):
    "k logarithmic steps whose sum is n"
    x = first_bucket_size(k-1, n)
    sum = 0
    for i in range(k):
        sum += 2**i * x
        yield sum



class MVSD(object):
    "A class that calculates a running Mean / Variance / Standard Deviation"
    def __init__(self):
        self.is_started = False
        self.ss = Decimal(0)  # (running) sum of square deviations from mean
        self.m = Decimal(0)  # (running) mean
        self.total_w = Decimal(0)  # weight of items seen

    def add(self, x, w=1):
        "add another datapoint to the Mean / Variance / Standard Deviation"
        if not isinstance(x, Decimal):
            x = Decimal(x)
        if not self.is_started:
            self.m = x
            self.ss = Decimal(0)
            self.total_w = w
            self.is_started = True
        else:
            temp_w = self.total_w + w
            self.ss += (self.total_w * w * (x - self.m) *
                        (x - self.m)) / temp_w
            self.m += (x - self.m) / temp_w
            self.total_w = temp_w

    def var(self):
        return self.ss / self.total_w

    def sd(self):
        return math.sqrt(self.var())

    def mean(self):
        return self.m

DataPoint = namedtuple('DataPoint', ['value', 'count'])


def test_mvsd():
    mvsd = MVSD()
    for x in range(10):
        mvsd.add(x)

    assert '%.2f' % mvsd.mean() == "4.50"
    assert '%.2f' % mvsd.var() == "8.25"
    assert '%.14f' % mvsd.sd() == "2.87228132326901"


def load_stream(input_stream, agg_value_key, agg_key_value):
    for line in input_stream:
        clean_line = line.strip()
        if not clean_line:
            # skip empty lines (ie: newlines)
            continue
        if clean_line[0] in ['"', "'"]:
            clean_line = clean_line.strip("\"'")
        try:
            if agg_key_value:
                key, value = clean_line.rstrip().rsplit(None, 1)
                yield DataPoint(Decimal(key), int(value))
            elif agg_value_key:
                value, key = clean_line.lstrip().split(None, 1)
                yield DataPoint(Decimal(key), int(value))
            else:
                yield DataPoint(Decimal(clean_line), 1)
        except:
            logging.exception('failed %r', line)
            print >>sys.stderr, "invalid line %r" % line


def median(values, key=None):
    if not key:
        key = None  # map and sort accept None as identity
    length = len(values)
    if length % 2:
        median_indeces = [length/2]
    else:
        median_indeces = [length/2-1, length/2]

    values = sorted(values, key=key)
    return sum(map(key,
                   [values[i] for i in median_indeces])) / len(median_indeces)


def test_median():
    assert 6 == median([8, 7, 9, 1, 2, 6, 3])  # odd-sized list
    assert 4 == median([4, 5, 2, 1, 9, 10])  # even-sized int list. (4+5)/2 = 4
    # even-sized float list. (4.0+5)/2 = 4.5
    assert "4.50" == "%.2f" % median([4.0, 5, 2, 1, 9, 10])


def histogram(stream, options):
    """
    Loop over the stream and add each entry to the dataset, printing out at the
    end.

    stream yields Decimal()
    """

    boundaries = []
    bucket_counts = []
    buckets = 0
    bucket_scale = 1

    # always track underflow
    boundaries.append(Decimal("-Infinity"))

    # calculate boundaries first
    if options.custbuckets:

        # assume bucket list is each edge, i.e.,
        # there is one less bucket than specified values
        # e.g., 0,1,2,3,4 has 4 buckets (one at each comma)
        bound = options.custbuckets.split(',')
        bound_sort = sorted(map(Decimal, bound))

        # iterate through the sorted list and append to boundaries
        for x in bound_sort:
            boundaries.append(x)

    else:
        # need to determin min, max and step

        # glob the iterator here so we can do min/max on it
        if not options.min or not options.max:
            stream = list(stream)

        min_v = ( Decimal(options.min) if options.min 
                    else min(stream, key=lambda x: x.value).value)
        max_v = ( Decimal(options.max) if options.max 
                    else max(stream, key=lambda x: x.value).value)

            # check to make sure all values aren't the same
        if not options.min or not options.max:
            if max_v <= min_v:
                max_v = min_v + 1;


        # should be fine if we entered if above, but if user specified
        # need to make sure max is greater than min (though theoretically we
        # could have a decreasing x-axis).
        if not max_v > min_v:
            raise ValueError('max must be > min. max:%s min:%s' % (max_v, min_v))

        diff = max_v - min_v
        buckets = 10 if not options.buckets else int(options.buckets)

        if buckets <= 0:
            raise ValueError('# of buckets must be > 0')

        # for nBuckets, we need nBuckets+1 boundaries
        boundaries.append(min_v)

        if options.logscale:
            for step in log_steps(buckets, diff):
                boundaries.append(min_v + step)
        else:
            step = diff / buckets
            for x in range(buckets):
                boundaries.append(min_v + (step * (x + 1)))

    # always track overflow
    boundaries.append(Decimal("Infinity"))

    buckets = len(boundaries) - 1
    bucket_counts = [0 for x in range(buckets)]

    mvsd = MVSD()
    accepted_data = []
    max_v = Decimal("-Infinity")
    min_v = Decimal("Infinity")
    samples = 0

    # let's fill
    for record in stream:

        samples += 1

        # track min,max value
        if record.value > max_v:
            max_v = record.value
        if record.value > min_v:
            min_v = record.value

        if options.mvsd:
            mvsd.add(record.value, record.count)
            accepted_data.append(record)

        for bucket_postion, boundary in enumerate(boundaries):
            # boundaries are ordered so if rhs of bucket is greater than
            # value, fill the bucket between current boundary and previous
            # boundary
            if boundary > record.value:
                bucket_counts[bucket_postion-1] += record.count
                break


    # auto-pick the hash scale
    if max(bucket_counts) > 75:
        bucket_scale = int(max(bucket_counts) / 75)

    nSkipped = 0
    if not options.overunder:
        nSkipped = bucket_counts[0] + bucket_counts[-1]

    area = sum(bucket_counts) - nSkipped

    # header
    if not options.quiet:
        print("# NumSamples = %d; Area = %0.2f; Min = %0.2f; Max = %0.2f" %
              (samples,area,min_v,max_v))

        if not options.overunder and nSkipped > 0:
            print("# %d value%s outside of min/max" %
                  (nSkipped, nSkipped > 1 and 's' or ''))

        if options.mvsd:
            print("# Mean = %f; Variance = %f; SD = %f; Median %f" %
                  (mvsd.mean(), mvsd.var(), mvsd.sd(),
                   median(accepted_data, key=lambda x: x.value)))

        print "# each " + options.dot + " represents a count of %d" % bucket_scale

    percentage = ""
    format_string = options.format + ' - ' + options.format + ' [%6d]: %s%s'

    for bucket in range(buckets):
        if not options.overunder and (bucket == 0 or bucket == buckets-1):
            continue
        bucket_min = boundaries[bucket]
        bucket_max = boundaries[bucket+1]
        bucket_count = bucket_counts[bucket]
        star_count = bucket_count / bucket_scale
        if options.percentage:
            percentage = " (%0.2f%%)" % (100 * Decimal(bucket_count) /
                                         Decimal(area))
        print format_string % (bucket_min, bucket_max, bucket_count, 
                                options.dot * star_count, percentage)


if __name__ == "__main__":
    parser = OptionParser()
    parser.usage = "cat data | %prog [options]"
    parser.add_option("-a", "--agg", dest="agg_value_key", default=False,
                      action="store_true", help="Two column input format, " +
                      "space seperated with value<space>key")
    parser.add_option("-A", "--agg-key-value", dest="agg_key_value",
                      default=False, action="store_true", help="Two column " +
                      "input format, space seperated with key<space>value")
    parser.add_option("-m", "--min", dest="min",
                      help="minimum value for graph; not specifying both min"
                      " and max disables streaming, all data will be loaded"
                      " into memory before histogramming")
    parser.add_option("-x", "--max", dest="max",
                      help="maximum value for graph; not specifying both min"
                      " and max disables streaming, all data will be loaded"
                      " into memory before histogramming")
    parser.add_option("-b", "--buckets", dest="buckets",
                      help="Number of buckets to use for the histogram")
    parser.add_option("-l", "--logscale", dest="logscale", default=False,
                      action="store_true",
                      help="Buckets grow in logarithmic scale")
    parser.add_option("-B", "--custom-buckets", dest="custbuckets",
                      help="Comma seperated list of bucket " +
                      "edges for the histogram")
    parser.add_option("--no-mvsd", dest="mvsd", action="store_false",
                      default=True, help="Disable the calculation of Mean, " +
                      "Variance and SD (improves performance)")
    parser.add_option("-f", "--bucket-format", dest="format", default="%10.4f",
                      help="format for bucket numbers")
    parser.add_option("-p", "--percentage", dest="percentage", default=False,
                      action="store_true", help="List percentage for each bar")
    parser.add_option("-d","--dot", dest="dot", default='∎', help="Dot representation")
    parser.add_option("-q","--quiet", dest="quiet", action="store_true", help="ssssh, no header")
    parser.add_option("-o","--overunder", dest="overunder", action="store_true", 
            help="Add overflow and underflow bins")

    (options, args) = parser.parse_args()

    if (options.min or options.max or options.buckets) and options.custbuckets:
        print ("Cannot specify min, max or number of buckets " 
              "as well as customer buckets\n")
        parser.print_usage()
        sys.exit(1)

    if sys.stdin.isatty():
        # if isatty() that means it's run without anything piped into it
        parser.print_usage()
        print "for more help use --help"
        sys.exit(1)

    histogram(load_stream(sys.stdin, options.agg_value_key,
                          options.agg_key_value), options)

