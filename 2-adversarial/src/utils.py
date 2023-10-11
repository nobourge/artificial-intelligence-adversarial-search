# def min_distance_position(position, positions):
#     # Create a cost vector
#     cost_vector = np.zeros(len(positions))
    
#     for i, point in enumerate(positions):
#         cost_vector[i] = ((position[0] - point[0]) ** 2 + (position[1] - point[1]) ** 2) ** 0.5

#     # Find the index of the minimum distance
#     min_index = np.argmin(cost_vector)

#     # Extract the closest position and its distance
#     closest_position = positions[min_index]
#     min_distance = cost_vector[min_index]

#     return closest_position, min_distance

from typing import List, Tuple


def get_distance(coord1, coord2):
    """Returns the distance between two coordinates"""
    x1, y1 = coord1
    x2, y2 = coord2
    return abs(x1 - x2) + abs(y1 - y2)

def min_distance_position(position : Tuple[int, int]
                          , positions: List[Tuple[int, int]] 
                            ) -> Tuple[Tuple[int, int], float]:
    """Returns the position in positions that is closest to position"""
    min_distance = float("inf")
    min_position = None
    for pos in positions:
        distance = 0
        distance = get_distance(position, pos)
        # print(distance)
        if distance < min_distance:
            min_distance = distance
            min_position = pos
    return min_position, min_distance

# def order_items()
# function to print visited set or stack items in terminal
def print_items(title, items, transform=None) -> None:
    """Prints items in terminal
    Args:
        items: items to print
    T is a generic type variable
    possible types for T:
    set, list, tuple, dict, etc."""
    print(title)
    i = 0
    for item in items:
        i += 1
        print(i, item)
    print("")


