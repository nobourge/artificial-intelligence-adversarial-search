import copy
import datetime
import os
import sys
from typing import List, Tuple

from lle import Action, World
from mdp import MDP, S, A

from world_mdp import BetterValueFunction, WorldMDP
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter


def stock_tree(mdp: MDP[A, S]
               , algorithm: str
                ) -> None:
    """Stocks the tree in a png file"""
    if isinstance(mdp, WorldMDP):
        if not os.path.exists('tree/current/'+algorithm):
            os.makedirs('tree/current/'+algorithm)
        UniqueDotExporter(mdp.root).to_picture("tree/current/"+algorithm+".png")
        print("tree stocked in tree/current/"+algorithm+".png")

        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('tree/'+algorithm):
            os.makedirs('tree/'+algorithm)
        UniqueDotExporter(mdp.root).to_picture("tree/"+algorithm+"/"+date_time+".png")
        print("tree stocked in tree/"+algorithm+"/"+date_time+".png")
