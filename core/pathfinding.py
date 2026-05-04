import heapq
import numpy as np

class PathFinder:
    """
    A* pathfinding utilities.

    This module includes:
    - astar(start, goal, mask)
    - astar_pathedgoal(start, goal_path, mask)
    """
    def __init__(self, state):
        self.state = state

    @staticmethod
    def heuristic(a, b):
        """ Calculates the Euclidean distance between two points.

            Args:
                a, b (tuples (x,y))
                    2D points as x, y coordinates

            Returns:
                Euclidean distance between x and y
        """
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def astar(self, start, goal, mask):
        """ A* directly from start to a single goal point in root skeletons (walkable pixels == 255) using
            Euclidean distance as a heuristic.

            Args:
                mask (2D numpy array)
                    Root skeleton where walkable pixels == 255

                start, goal (tuples (x,y))
                    2D points as x, y coordinates

            Returns:
                path (list(tuples (x,y)))
                    list of 2D points (x,y) along the path found by A*

            Raises:
                RuntimeError
                    When start or goal pixel != 255 in mask
        """
        rows, cols = mask.shape
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        if mask[start[1], start[0]] != 255 or mask[goal[1], goal[0]] != 255:
            raise RuntimeError("Start or goal is not walkable.")

        # neighbor offsets
        neighbors = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Construct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for dx, dy in neighbors:
                nb = (current[0] + dx, current[1] + dy)

                if 0 <= nb[0] < cols and 0 <= nb[1] < rows and mask[nb[1], nb[0]] != 0:
                    cost = 1.414 if dx != 0 and dy != 0 else 1
                    tentative = gscore[current] + cost

                    if nb not in gscore or tentative < gscore[nb]:
                        came_from[nb] = current
                        gscore[nb] = tentative
                        fscore[nb] = tentative + self.heuristic(nb, goal)
                        heapq.heappush(open_set, (fscore[nb], nb))
        return None

    def astar_pathedgoal(self, start, goal_path, mask):
        """ A* from start to any point in a list of points (path of the main root axis) using Euclidean distance as a
            heuristic. In every step the Euclidean distance between the current point and any point in the goal set is
            calculated and the goal point with the smallest euclidean distance is used as the current goal. The path is
            returned when current == any point in the goal set. Walkable pixels in mask are pixels == 255. This is used
            to find paths for lateral roots once the main root axis is established.

            Args:
                mask (2D numpy array)
                    Root skeleton where walkable pixels == 255

                start (tuple (x,y))
                    2D points as x, y coordinates

                goal_path (list(tuples (x,y)))
                    list of 2D points (x,y)

            Returns:
                path (list(tuples (x,y)))
                    list of 2D points (x,y) along the path found by A*

            Raises:
                RuntimeError
                    When start pixel != 255 in mask

                RuntimeError
                    When no pixel in goal set == 255
        """
        start = tuple(map(int, start))
        rows, cols = mask.shape

        # Validate start
        if mask[start[1], start[0]] != 255:
            raise RuntimeError("Start is not walkable.")

        # Preprocess goal set
        # Convert to ints & remove those outside mask
        goal_set = []
        for (x, y) in goal_path:
            x = int(x); y = int(y)
            if 0 <= x < cols and 0 <= y < rows:
                if mask[y, x] != 0:
                    goal_set.append((x, y))

        if not goal_set:
            raise RuntimeError("No usable goal pixels in goal_path.")

        # Quick hit: if start already on the path
        if start in goal_set:
            return [start]

        # Precompute: map each walkable pixel to heuristic = distance to nearest goal pixel
        goal_array = np.array(goal_set)
        gx = goal_array[:, 0]
        gy = goal_array[:, 1]

        def dist_to_path(p):
            """Return min Euclidean distance from pixel p to any goal pixel."""
            return np.min(np.hypot(gx - p[0], gy - p[1]))

        neighbors = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        open_set = []
        came_from = {}
        gscore = {start: 0.0}
        fscore = {start: dist_to_path(start)}

        heapq.heappush(open_set, (fscore[start], start))

        # Use set for quick membership check
        goal_set_lookup = set(goal_set)

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if we hit the path
            if current in goal_set_lookup:
                # reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for dx, dy in neighbors:
                nb = (current[0] + dx, current[1] + dy)
                if 0 <= nb[0] < cols and 0 <= nb[1] < rows and mask[nb[1], nb[0]] != 0:
                    step_cost = 1.414 if dx != 0 and dy != 0 else 1
                    tentative = gscore[current] + step_cost

                    if nb not in gscore or tentative < gscore[nb]:
                        came_from[nb] = current
                        gscore[nb] = tentative
                        f_nb = tentative + dist_to_path(nb)
                        fscore[nb] = f_nb
                        heapq.heappush(open_set, (f_nb, nb))

        return None