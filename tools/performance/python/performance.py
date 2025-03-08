# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: performance python line chart

# !/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import optparse
import numpy as np
import matplotlib.pyplot as plt


def process_log(file):
    seq_data = []
    batch_data = []
    with open(file) as fp:
        line = fp.readline()
        while line and 'Benchmark Batch' not in line:
            if 'TFLOPS' in line:
                seq_data.append(line)

            line = fp.readline()

        while line:
            if 'TFLOPS' in line:
                batch_data.append(line)

            line = fp.readline()

    return seq_data, batch_data


def draw_line_chart(methods, dims, data, figure_name, y_step, x_label, y_label, title):
    fig = plt.figure(figsize=(32, 24), dpi=100)

    dims_str = list(map(str, dims))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']

    for i in range(len(methods)):
        plt.plot(dims_str, data[i], color=colors[i % len(colors)],
                 linestyle=linestyles[(i // len(colors)) % len(linestyles)], marker='o', markersize=6)

    # plt.xticks(dims)
    plt.ylim(bottom=0)
    plt.yticks(
        range(0, round(np.max(np.max(data, axis=0)) + 0.5) + y_step, y_step))
    plt.tick_params(labelsize=25)

    # plt.hlines(y=100, xmin=dims_str[0], xmax=dims_str[-1], colors='r', linestyles='-.')
    plt.grid(True, linestyle='-.')

    plt.xlabel(x_label, fontdict={'size': '30'})
    plt.ylabel(y_label, fontdict={'size': '30'})
    plt.title(title, fontdict={'size': '30'})
    plt.legend(methods, loc='best', prop={'size': '30'})

    plt.savefig(figure_name, dpi=fig.dpi)
    # plt.show()


def analyze_data(data, dim_idx, data_path, data_type, x_label):
    methods = []
    dims = []
    for it in data:
        iterms = it.split(' ')

        method = iterms[0]
        if method not in methods:
            methods.append(method)

        params = iterms[1].split('-')
        dim = int(params[dim_idx])
        if dim not in dims:
            dims.append(dim)

    dims.sort()

    throughputs = np.zeros((len(methods), len(dims)), np.float64)
    bandwidths = np.zeros((len(methods), len(dims)), np.float64)
    for it in data:
        iterms = it.split(' ')
        method = iterms[0]
        params = iterms[1].split('-')
        dim = int(params[dim_idx])

        throughputs[methods.index(
            method)][dims.index(dim)] = float(iterms[7])

        bandwidths[methods.index(
            method)][dims.index(dim)] = float(iterms[10])

    draw_line_chart(methods, dims, throughputs, data_path + data_type +
                    '_throughput.png', 20, x_label, 'Throughput / TFLOPS', 'Decoding Attention Throughput')

    draw_line_chart(methods, dims, bandwidths, data_path + data_type +
                    '_bandwidth.png', 100, x_label, 'Bandwidth / GB/s', 'Decoding Attention Bandwidth')


def main():
    usage = "python3 performance.py -f log/benchmark_decoding_attn.log"
    parser = optparse.OptionParser(usage)
    parser.add_option('-f', '--file', dest='file',
                      type='string', help='file name', default='log/benchmark_decoding_attn.log')

    options, args = parser.parse_args()
    file = options.file

    seq_data, batch_data = process_log(file)
    data_path = os.path.dirname(file)
    analyze_data(seq_data, 1, data_path, '/seq', 'Seq Len')
    analyze_data(batch_data, 0, data_path, '/batch', 'Batch Size')


if __name__ == "__main__":
    main()
