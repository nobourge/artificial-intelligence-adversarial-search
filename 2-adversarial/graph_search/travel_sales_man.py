# Adjusted code based on user input
import heapq
from typing import List, Tuple
import numpy as np
from priority_queue import PriorityQueueOptimized
from utils import min_distance_position

# Adapted Greedy Algorithm with Balanced Total Distances
def balanced_multi_salesmen_greedy_tsp(remaining_cities: List[Tuple[int, int]]
                                       , num_salesmen: int
                                       , start_cities: List[Tuple[int, int]]
                                       , finish_cities: List[Tuple[int, int]]):
    """Given a list of cities coordinates, returns a list of cities visited by each salesman
    in the order that minimizes the total distance traveled.
    """
    
    routes = {f"Salesman_{i+1}": [start_cities[i]] for i in range(num_salesmen)}
    distances = {f"Salesman_{i+1}": 0.0 for i in range(num_salesmen)}

    while remaining_cities:
        for salesman in routes.keys():
            if not remaining_cities:
                break
            
            current_city = routes[salesman][-1]
            nearest_city, nearest_distance = min_distance_position(routes[salesman][-1], remaining_cities)
            distances[salesman] += nearest_distance
            routes[salesman].append(nearest_city)
            remaining_cities.remove(nearest_city)

    for salesman in routes.keys():
        current_city = routes[salesman][-1]
        finish_city, final_distance = min_distance_position(current_city, finish_cities)
        distances[salesman] += final_distance
        routes[salesman].append(finish_city)
        
    total_distance = sum(distances.values())
    return routes, distances, total_distance


# Greedy Algorithm with Min Heap (Priority Queue)
def greedy_tsp_heap(cities_to_visit
                    , start_city=None
                    , finish_city=None):
    """Given a dict of cities and their coordinates, returns a list of cities
    visited in the order that minimizes the total distance traveled.
    Route: A→C→B→D→E→A
    Total Distance: 12.35
    Elapsed Time: 0.0000703 seconds"""

    road = [start_city]
    current_city = start_city
    total_distance = 0.0

    def distance(city1, city2):
        x1, y1 = cities_to_visit[city1]
        x2, y2 = cities_to_visit[city2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    while cities_to_visit:
        heap = [(distance(current_city, city), city) for city in cities_to_visit]
        heapq.heapify(heap)
        nearest_distance, nearest_city = heapq.heappop(heap)
        total_distance += nearest_distance
        road.append(nearest_city)
        current_city = nearest_city
        cities_to_visit.remove(nearest_city)

    total_distance += distance(current_city, finish_city)
    road.append(finish_city)

    return road, total_distance


def greedy_tsp_optimized_pq(cities):
    """Given a dict of cities and their coordinates, returns a list of cities
    visited in the order that minimizes the total distance traveled.
    Elapsed Time: 0.0002046 seconds"""
    start_city = list(cities.keys())[0]
    road = [start_city]
    cities_to_visit = list(set(cities.keys()) - set(road))
    current_city = start_city
    total_distance = 0.0

    def distance(city1, city2):
        x1, y1 = cities[city1]
        x2, y2 = cities[city2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    pq = PriorityQueueOptimized()

    while cities_to_visit:
        for city in cities_to_visit:
            pq.push(city, distance(current_city, city))
        
        nearest_city = pq.pop()
        nearest_distance = distance(current_city, nearest_city)
        total_distance += nearest_distance
        road.append(nearest_city)
        current_city = nearest_city
        cities_to_visit.remove(nearest_city)

    total_distance += distance(current_city, start_city)
    road.append(start_city)

    return road, total_distance