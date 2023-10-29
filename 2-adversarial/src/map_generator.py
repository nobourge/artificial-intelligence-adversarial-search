import random


class MapGenerator:
    def __init__(self
                 , map_width = 9
                 , map_height = 9
                 , map_type = "random"
                 ):
        self.map_width = map_width
        self.map_height = map_height
        self.map_type = map_type
        self.map = []
        self.map = self.generate_map_str()
    

    def generate_map_str(self):
        # Generate map based on map_type
        random_params = self.generate_random_params()
        random_map = self.generate_random_map(**random_params)
        # random_map_str = self.map_to_str(random_map)
        matrix_layout = self.matrix_to_layout(random_map)
        return matrix_layout        
        
    def generate_random_map(self
                               , rows=9
                               , cols=9
                               , num_agents=1
                               , num_gems=0
                               , num_lasers=0
                               ):
        """
        Generate a random map of dimensions (rows x cols) with given elements.
        
        Parameters:
        - rows: Number of rows
        - cols: Number of columns
        - num_agents: Number of agents (default is 1)
        - num_gems: Number of gems (default is 0)
        - num_lasers: Number of lasers (default is 0)
        
        Returns:
        - A list of lists representing the map
        """
        # if rows*cols < num_agents*2 + num_gems + num_lasers:
        #     rows = cols = num_agents*2 + num_gems + num_lasers
        # Initialize the map with floor tiles '.' and walls '@'
        map_grid = [['.' for _ in range(cols)] for _ in range(rows)]
        
        
        
        # Place agents' start positions
        for i in range(num_agents):
            placed = False
            while not placed:
                r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                if map_grid[r][c] == '.':
                    map_grid[r][c] = f'S{i}'
                    placed = True
        
        # Place exits (same number as agents)
        for _ in range(num_agents):
            while True:
                r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                if map_grid[r][c] == '.':
                    map_grid[r][c] = 'X'
                    break
        
        # Place gems
        for _ in range(num_gems):
            for _ in range(3):
                r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                if map_grid[r][c] == '.':
                    map_grid[r][c] = 'G'
                    break
        
        # Place lasers
        for i in range(num_lasers):
            for _ in range(3):
                r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                if map_grid[r][c] == '.':
                    direction = random.choice(['N', 'S', 'E', 'W'])
                    map_grid[r][c] = f'L{i}{direction}'
                    break
        # Randomly place walls (20% of the map)
        for _ in range((rows * cols) // 5):
            r, c = random.randint(0, rows-1), random.randint(0, cols-1)
            map_grid[r][c] = '@'
        
        return map_grid

    # Function to generate random but reasonable parameters for the map
    def generate_random_params(self
                               , max_rows=9
                            #    , max_rows=5
                               , max_cols=9
                            #    , max_cols=5
                               , max_agents=3
                               , max_gems=6
                               , max_lasers=3):
        """
        Generate random but reasonable parameters for the map.
        
        Parameters:
        - max_rows: Maximum number of rows (default is 9)
        - max_cols: Maximum number of columns (default is 9)
        - max_agents: Maximum number of agents (default is 4)
        - max_gems: Maximum number of gems (default is 6)
        - max_lasers: Maximum number of lasers (default is 3)
        
        Returns:
        - A dictionary containing the generated parameters
        """
        rows = random.randint(2, max_rows)
        cols = random.randint(2, max_cols)
        max_agents = min(max_agents, rows*cols//2)
        num_agents = random.randint(2, max_agents)
        print("rows: ", rows)
        print("cols: ", cols)
        print("num_agents: ", num_agents)

        params = {
            "rows": rows,
            "cols": cols,
            "num_agents": num_agents,
            "num_gems": random.randint(0, max_gems),
            "num_lasers": random.randint(0, max_lasers)
        }
        
        return params

    def map_to_string(self
                      , map_grid):
        """
        Convert a given map into a string.
        
        Parameters:
        - map_grid: A list of lists representing the map.
        
        Returns:
        - A string representing the map.
        """
        return "\n".join(" ".join(row) for row in map_grid)
    
    def matrix_to_layout(self
                         ,matrix):
        """
        Convert a given matrix into a layout.
        
        Parameters:
        matrix (list of list of str): A matrix representing the layout.

        Returns:
        list of str: Each string represents a row in the layout.
        """
        # Determine the maximum length of any element in the matrix for alignment
        max_len = max(len(str(item)) for row in matrix for item in row)
        
        layout = """"""
        # layout += "\n"

        for row in matrix:
            # Align the elements by padding with spaces
            aligned_row = " ".join(str(item).ljust(max_len) for item in row)
            layout += aligned_row + "\n"
            
        return layout

