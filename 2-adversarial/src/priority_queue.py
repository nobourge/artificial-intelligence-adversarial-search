import heapq
import json
from typing import Dict, Generic, TypeVar
# from typing_extensions import deprecated

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.

    Credits: Berkley AI Pacman Project
    """

    def __init__(self):
        self.heap: list[T] = []
        self.count = 0

    def push(self, item: T, priority: float):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self) -> T:
        """pop the item with the minimum priority"""
        (_, _, item) = heapq.heappop(self.heap)
        return item
    
    def pop_maximum_priority(self) -> T:
        """pop the item with the maximum priority"""
        (_, _, item) = heapq.heappop(self.heap)
        return item

    # @deprecated
    # def isEmpty(self):
    #     return len(self.heap) == 0

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item: T, priority: float):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class PriorityQueueOptimized(Generic[T]): #todo: 2x slower than PriorityQueue at least
    """heapq default minheap
    
    heapify to rebuild the heap, which is an 
    O(n) operation optimized using a dictionary 
    to keep track of the heap indices for each item, allowing to 
    update the heap in 
    O(logn) time."""
    def __init__(self):
        self.heap = []
        # self.entry_finder: Dict[T, int] = {}  
        self.entry_finder: Dict[str, int] = {}  # String keys for unhashable types
        self.count = 0

    # PriorityQueueOptimized iterable form
    def __iter__(self):
        return iter(self.heap)
    
    def _stringify(self, item: T) -> str:
        # return json.dumps(item)
        return str(id(item))
    
    def serialize(self, world_state: T) -> tuple:
        return (tuple(world_state.agents_positions), tuple(world_state.gems_collected))

    def push(self, item: T, priority: float):
        # serialized_item = self.serialize(item)
        # entry = (priority, self.count, serialized_item)
        # self.entry_finder[serialized_item] = len(self.heap)  # Keep track of the index
        entry = (priority, self.count, item)
        key = self._stringify(item)
        self.entry_finder[key] = len(self.heap)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self) -> T:
        while self.heap:
            _, _, item = heapq.heappop(self.heap)
            key = self._stringify(item)
            # key = self.serialize(item)
            if key in self.entry_finder:
                del self.entry_finder[key]
                return item
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item: T, priority: float):
        key = self._stringify(item)
        # key = self.serialize(item)

        if key in self.entry_finder:
            index = self.entry_finder[key]
            _, _, existing_item = self.heap[index]
            # if existing_item is not key:
            if existing_item is not item:
                return
            self.heap[index] = (priority, self.count, item)
            heapq._siftup(self.heap, index)
            heapq._siftdown(self.heap, 0, index)
            self.count += 1
        else:
            self.push(item, priority)

