import cv2
from lle import World


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
"expectimax":  2706 ,
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
"expectimax":  3160 ,
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
"expectimax":  701 ,
"action_minimax":  " Stay " ,
"action_minimax_with_better_value_function":  " North " ,
"action_alpha_beta":  " Stay " ,
"action_alpha_beta_with_better_value_function":  " North " ,
"action_expectimax":  " Stay " ,
}

map2023_10_29_13_56_42 = """.  X  G 
@  @  S0
.  .  . 
.  .  . 
.  X  S1
"""

def display(map_str, map_name):
    world_instance = World(map_str)
    img = world_instance.get_image()
    cv2.imshow("Visualisation", img)
    cv2.waitKey(0) # Attend que l'utilisateur appuie sur 'enter'
    cv2.waitKey(1) # continue l'exécution du code
    #save img
    cv2.imwrite(map_name, img)

maps_names = ["map2023_10_29_13_48_15", "map2023_10_29_13_49_35", "map2023_10_29_13_53_25", "map2023_10_29_13_55_47", "map2023_10_29_13_56_42"]
map_names_png = ["map2023_10_29_13_48_15.png", "map2023_10_29_13_49_35.png", "map2023_10_29_13_53_25.png", "map2023_10_29_13_55_47.png", "map2023_10_29_13_56_42.png"]
maps = [map2023_10_29_13_48_15, map2023_10_29_13_49_35, map2023_10_29_13_53_25, map2023_10_29_13_55_47, map2023_10_29_13_56_42]
for map_str, map_name in zip(maps, map_names_png):
    display(map_str, map_name)