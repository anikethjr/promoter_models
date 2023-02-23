# functions used to create sequence importance plots
# derived from https://github.com/kundajelab/deeplift/blob/master/deeplift/visualization/viz_sequence.py

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
      np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.2, 0.0]]),
      np.array([[1.0, 0.0], [0.5, 1.0], [0.5, 0.8], [0.8, 0.0]]),
      np.array([[0.225, 0.45], [0.775, 0.45], [0.85, 0.3], [0.15, 0.3]])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(
            matplotlib.patches.Polygon(
                (np.array([1, height])[None, :] * polygon_coords + np.array(
                    [left_edge, base])[None, :]),
                facecolor=color,
                edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=1.3,
          height=height,
          facecolor=color,
          edgecolor=color))
    ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=0.7 * 1.3,
          height=0.7 * height,
          facecolor='white',
          edgecolor='white'))
    ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 1, base],
          width=1.0,
          height=height,
          facecolor='white',
          edgecolor='white',
          fill=True))

def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=1.3,
          height=height,
          facecolor=color,
          edgecolor=color))
    ax.add_patch(
      matplotlib.patches.Ellipse(
          xy=[left_edge + 0.65, base + 0.5 * height],
          width=0.7 * 1.3,
          height=0.7 * height,
          facecolor='white',
          edgecolor='white'))
    ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 1, base],
          width=1.0,
          height=height,
          facecolor='white',
          edgecolor='white',
          fill=True))
    ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 0.825, base + 0.085 * height],
          width=0.174,
          height=0.415 * height,
          facecolor=color,
          edgecolor=color,
          fill=True))
    ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 0.625, base + 0.35 * height],
          width=0.374,
          height=0.15 * height,
          facecolor=color,
          edgecolor=color,
          fill=True))

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge + 0.4, base],
          width=0.2,
          height=height,
          facecolor=color,
          edgecolor=color,
          fill=True))
    ax.add_patch(
      matplotlib.patches.Rectangle(
          xy=[left_edge, base + 0.8 * height],
          width=1.0,
          height=0.2 * height,
          facecolor=color,
          edgecolor=color,
          fill=True))

def seqlogo(array, height_padding_factor=0.5, length_padding=1, subticks_frequency=50, ax=None):
    if ax is None:
        ax = plt.gca()

    default_colors = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green'}
    default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}
    
    colors = ['red', 'blue', 'orange', 'green']
    plot_funcs = [plot_a, plot_c, plot_g, plot_t]
    
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)