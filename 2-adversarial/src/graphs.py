# this file does the following:


# compare the number of nodes extended during the search for 
    # • minimax,
    # • minimax avec votre meilleure fonction d’évaluation,
    # • alpha_beta,
    # • alpha_beta avec la meilleure fonction d’évaluation.
    # • expectimax
# when searching
# on different maps,



import os
import numpy as np
from typing import List, Tuple
from lle import World
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


map2023_10_29_13_48_15 = """.   G   S1 
L0E S0  .  
X   @   X  
"""
results2023_10_29_13_48_15 = {
"depth":  3 ,
"minimax":  27 ,
"minimax_with_better_value_function":  27 ,
"alpha_beta":  22 ,
"alpha_beta_with_better_value_function":  16 ,
"expectimax":  27 ,
"action_minimax":  " North " ,
"action_minimax_with_better_value_function":  " North " ,
"action_alpha_beta":  " North " ,
"action_alpha_beta_with_better_value_function":  " North " ,
"action_expectimax":  " North " ,
}

map2023_10_29_13_49_35 = """.  S0 S2 X 
.  .  G  . 
.  @  .  G 
.  .  S1 . 
.  .  X  X 
"""
results2023_10_29_13_49_35 = {
"depth":  4 ,
"minimax":  2706 ,
"minimax_with_better_value_function":  2706 ,
"alpha_beta":  312 ,
"alpha_beta_with_better_value_function":  366 ,
"expectimax":  253 ,
"action_minimax":  " Stay " ,
"action_minimax_with_better_value_function":  " Stay " ,
"action_alpha_beta":  " Stay " ,
"action_alpha_beta_with_better_value_function":  " South " ,
"action_expectimax":  " South " ,
}

map2023_10_29_13_53_25 = """G  .  .  .  .  G 
X  .  G  S2 S0 G 
.  .  G  X  S1 X 
.  .  G  .  .  . 
"""
results2023_10_29_13_53_25 = {
"depth":  4 ,
"minimax":  3160 ,
"minimax_with_better_value_function":  3160 ,
"alpha_beta":  929 ,
"alpha_beta_with_better_value_function":  436 ,
"expectimax":  303 ,
"action_minimax":  " East " ,
"action_minimax_with_better_value_function":  " East " ,
"action_alpha_beta":  " East " ,
"action_alpha_beta_with_better_value_function":  " East " ,
"action_expectimax":  " East " ,
}

map2023_10_29_13_55_47 = """.   .   L0S .   .   S2 
X   L1S .   S1  .   .  
X   .   .   @   S0  X  
"""
results2023_10_29_13_55_47 = {
"depth":  4 ,
"minimax":  701 ,
"minimax_with_better_value_function":  701 ,
"alpha_beta":  297 ,
"alpha_beta_with_better_value_function":  385 ,
"expectimax":  94 ,
"action_minimax":  " Stay " ,
"action_minimax_with_better_value_function":  " Stay " ,
"action_alpha_beta":  " Stay " ,
"action_alpha_beta_with_better_value_function":  " North " ,
"action_expectimax":  " East " ,
}

map2023_10_29_13_56_42 = """.  X  G 
@  @  S0
.  .  . 
.  .  . 
.  X  S1
"""
results2023_10_29_13_56_42 = {
"depth":  3 ,
"minimax":  41 ,
"minimax_with_better_value_function":  41 ,
"alpha_beta":  30 ,
"alpha_beta_with_better_value_function":  20 ,
"expectimax":  41 ,
"action_minimax":  " North " ,
"action_minimax_with_better_value_function":  " North " ,
"action_alpha_beta":  " North " ,
"action_alpha_beta_with_better_value_function":  " North " ,
"action_expectimax":  " North " ,
}


def find_var_name(var):
    for name, value in globals().items():
        if var is value:
            return name

# for each map, create a bar graph to compare the number of nodes expanded and action by each algorithm

def transcript_to_tex(map):
    """Transcribes map_name to its LaTeX equivalent for use in matplotlib titles."""
    # Dictionary mapping special characters to their LaTeX-safe versions
    special_chars = {
        ' ': r'\;',  # Space
        '.': r'\textperiodcentered',  # Period
        'G': r'G',  # G doesn't need escaping
        'S': r'S',  # S doesn't need escaping
        '1': r'1',  # 1 doesn't need escaping
        'L': r'L',  # L doesn't need escaping
        '0': r'0',  # 0 doesn't need escaping
        'E': r'E',  # E doesn't need escaping
        'X': r'X',  # X doesn't need escaping
        '@': r'\textcircled{a}',  # @ symbol
    }
    
    # Replace each character in map_name with its LaTeX-safe version
    map_name_title = ''.join(special_chars.get(c, c) for c in map)
    
    return map_name_title


def create_bar_graph(map_name, results):
    # Extract only the node expansion counts for each algorithm
    node_expanded_keys = ['minimax', 'minimax_with_better_value_function', 'alpha_beta', 'alpha_beta_with_better_value_function', 'expectimax']
    node_expansions = {k: results[k] for k in ['minimax', 'minimax_with_better_value_function', 'alpha_beta', 'alpha_beta_with_better_value_function', 'expectimax']}
    
    labels = ['minimax', 'minimax \n with_better_value_function', 'alpha_beta', 'alpha_beta \n with_better_value_function', 'expectimax']
    values = list(node_expansions.values())
    
    # Create the bar graph
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['red', 'green', 'blue', 'yellow', 'purple'])
    
    # Annotate the bars with the corresponding action
    for bar, label in zip(bars, node_expanded_keys):
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() - 0.1, str(results[label]) + "\n" + results[f'action_{label.lower()}'], va='center', ha='center')
    
    ax.set_ylabel('Nodes Expanded with depth = ' + str(results["depth"]))
    ax.set_xlabel('Algorithms')
    print("map_name: ", map_name)
    map_name_title = transcript_to_tex(map_name)

    print("map_name_title: ", map_name_title)
    # plt.title(r'$\text{' + map_name_title + '}$', usetex=True)
    # plt.title(r'{map_name_title}', usetex=True)
    plt.title(map_name_title, usetex=True)

    plt.tight_layout()
    # plt.show()
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    map_name_variable_name = find_var_name(map_name)
    plt.savefig(f'graphs/{map_name_variable_name}.png')

for map, results in zip([map2023_10_29_13_48_15
                        , map2023_10_29_13_49_35
                        , map2023_10_29_13_53_25
                        , map2023_10_29_13_55_47
                        , map2023_10_29_13_56_42
                        ]
                        , [results2023_10_29_13_48_15
                        , results2023_10_29_13_49_35
                        , results2023_10_29_13_53_25
                        , results2023_10_29_13_55_47
                        , results2023_10_29_13_56_42
                        ]):
    create_bar_graph(map, results)

# create_bar_graph(map2023_10_29_13_48_15, results2023_10_29_13_48_15)


a = 1
# print(find_var_name(a))  # Output: "a"
# print(find_var_name(1))  # Output: "a"
