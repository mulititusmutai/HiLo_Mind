import pygame
import math
import pymunk
import numpy as np
import heapq
from shapely.geometry import LineString, Point
from enum import Enum, auto

class FSM(Enum):
    scan      = auto()
    drive     = auto()
    backtrack = auto()
    solve     = auto()
    done      = auto()

def get_astar_path(nodes_dict, start_node, goal_node, treat_unknown_as_open=False):
    def heuristic(a, b):
        return abs(a.grid_pos[0] - b.grid_pos[0]) + abs(a.grid_pos[1] - b.grid_pos[1])

    def get_move_dir(from_node, to_node):
        return (to_node.grid_pos[0] - from_node.grid_pos[0], 
                to_node.grid_pos[1] - from_node.grid_pos[1])

    open_set = []
    heapq.heappush(open_set, (0, start_node.id, start_node))
    came_from = {}
    g_score = {start_node.id: 0.0}
    arrival_dir = {start_node.id: None}

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current.id == goal_node.id:            
            path = []
            final_cost = g_score[current.id]
            while current.id in came_from:
                path.append(current)
                current = came_from[current.id]
            return path[::-1], final_cost

        for d, is_wall in current.walls.items():            
            is_passable = (is_wall is False) or (treat_unknown_as_open and is_wall is None)         
            if is_passable:
                neighbor = current.grid_links.get(d)
                if not neighbor: continue
                move_dir = get_move_dir(current, neighbor)
                prev_dir = arrival_dir.get(current.id)                                
                turn_penalty = 0.5 if (prev_dir is not None and move_dir != prev_dir) else 0.0
                tentative_g = g_score[current.id] + 1.0 + turn_penalty

                if tentative_g < g_score.get(neighbor.id, float('inf')):
                    came_from[neighbor.id] = current
                    g_score[neighbor.id] = tentative_g
                    arrival_dir[neighbor.id] = move_dir
                    f = tentative_g + heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f, neighbor.id, neighbor))
                    
    return [], float('inf')

class Node:
    _id_counter = 0
    def __init__(self, grid_pos, world_pos):
        self.grid_pos   = grid_pos
        self.pos        = world_pos
        self.grid_links = {}
        self.passages   = set()
        self.walls      = {'n': None, 'e': None, 's': None, 'w': None}
        self.id         = Node._id_counter
        Node._id_counter += 1

class Grid:
    DIRS  = ('n', 'e', 's', 'w')
    DELTA = {'n': (0, 1), 'e': (1, 0), 's': (0, -1), 'w': (-1, 0)}

    def __init__(self, spacing: float, start_spec: dict):
        self.spacing = spacing
        self._nodes  = {}
        self.start   = None
        sx, sy = start_spec['x'], start_spec['y']
        HALF   = spacing / 2.0
        for gx in range(0, 18):
            for gy in range(0, 18):
                world = (gx * spacing - HALF, gy * spacing - HALF)
                node  = Node((gx, gy), world)
                self._nodes[(gx, gy)] = node
        for (gx, gy), node in self._nodes.items():
            for d, (ddx, ddy) in self.DELTA.items():
                nbr = self._nodes.get((gx + ddx, gy + ddy))
                node.grid_links[d] = nbr
        self.start = self._nodes[(sx, sy)]

    def node_at(self, grid_pos):
        return self._nodes.get(grid_pos)

class DFSExplorer:
    def __init__(self, start_node: Node, grid_nodes_dict: dict):
        self.visited = set()
        self._current = start_node
        self._grid_nodes = grid_nodes_dict

    def record_scan(self, node: Node, open_passages: set):
        node.passages = open_passages
        self.visited.add(node.id)
        self._current = node

    def decide_next(self, node: Node, goal_node: Node):
        best_nbr = None
        lowest_cost = float('inf')
        
        for d in node.passages:
            nbr = node.grid_links.get(d)
            if nbr and nbr.id not in self.visited:
                _, cost = get_astar_path(self._grid_nodes, nbr, goal_node, treat_unknown_as_open=True)
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_nbr = nbr

        if best_nbr:
            return ('drive', best_nbr)
        return self._backtrack_via_astar()

    def _backtrack_via_astar(self):
        frontiers = []
        for n_id in self.visited:
            candidate = next((n for n in self._grid_nodes.values() if n.id == n_id), None)
            if candidate:
                for d, is_wall in candidate.walls.items():
                    if is_wall is False:
                        nbr = candidate.grid_links.get(d)
                        if nbr and nbr.id not in self.visited:
                            dist = math.dist(self._current.pos, candidate.pos)
                            frontiers.append((dist, candidate, nbr))   
        if frontiers:
            frontiers.sort(key=lambda x: x[0])
            _, target_junction, next_unexplored = frontiers[0]
            path, _ = get_astar_path(self._grid_nodes, self._current, target_junction)            
            return ('backtrack', path, next_unexplored)    
        return ('done', None)

class Navigator:
    ARRIVAL_RADIUS = 11.0
    ANGLE_TOL      = 8.0
    def __init__(self):
        self.target = None

    def set_target(self, target: Node, from_node: Node = None):
        self.target = target

    def arrived(self, state: dict) -> bool:
        return math.dist(self.target.pos, state['position']) < self.ARRIVAL_RADIUS

    def steer(self, state: dict) -> str:
        dx = self.target.pos[0] - state['position'][0]
        dy = self.target.pos[1] - state['position'][1]
        desired = math.degrees(math.atan2(dy, dx)) % 360.0
        err = (desired - state['heading'] + 180) % 360 - 180
        if   err >  self.ANGLE_TOL: return 'r'
        elif err < -self.ANGLE_TOL: return 'l'
        else:                       return 'f'

_CARDINALS  = {90: 'n', 0: 'e', 270: 's', 180: 'w'}

def snap_to_cardinal(absolute_angle_deg: float):
    best, best_diff = None, 45.0
    for deg, label in _CARDINALS.items():
        diff = abs((absolute_angle_deg - deg + 180) % 360 - 180)
        if diff < best_diff:
            best_diff = diff
            best      = label
    return best

def read_wall_data(state: dict, mount_angles: list, open_threshold: float = 50.0) -> dict:
    wall_status = {}
    for reading, mount in zip(state['readings'], mount_angles):
        dist, _conf = reading
        abs_angle = (mount + state['heading']) % 360
        cardinal  = snap_to_cardinal(abs_angle)
        if cardinal:
            wall_status[cardinal] = dist <= open_threshold
    return wall_status

class Tree:
    def __init__(self, grid: Grid, mouse, goal_grid_pos=(1, 9)):
        self.grid  = grid
        self.mouse = mouse
        self.dfs = DFSExplorer(grid.start, grid._nodes)
        self.nav = Navigator()
        self._waypoints      = []
        self._post_bt_target = None
        self._at_node = grid.start
        self.FS       = FSM.scan
        self.goal_node = grid.node_at(goal_grid_pos)
        self.mission_phase = "exploring" 

    def mind(self):
        if not self.mouse.telemetry: return None
        state = self.mouse.telemetry[-1]

        if self.FS == FSM.scan:
            self._do_scan(state)
            return None
        elif self.FS == FSM.drive:
            if self.nav.arrived(state):
                self._at_node = self.nav.target
                self.FS = FSM.scan
                return None
            return self.nav.steer(state)
        elif self.FS == FSM.backtrack or self.FS == FSM.solve:
            return self._do_path_follow(state)
        elif self.FS == FSM.done:
            return None

    def _do_scan(self, state):
        node = self._at_node
        wall_data = read_wall_data(state, self.mouse.mount_angles)
        
        for direction, is_wall in wall_data.items():
            node.walls[direction] = is_wall
            
        open_passages = {d for d, wall in wall_data.items() if not wall}
        self.dfs.record_scan(node, open_passages)
        
        action = self.dfs.decide_next(node, self.goal_node)
        
        if action[0] == 'drive':
            self.nav.set_target(action[1], node)
            self.FS = FSM.drive
        elif action[0] == 'backtrack':
            self._start_path_sequence(action[1], action[2], FSM.backtrack)
        elif action[0] == 'done':
            self._handle_exploration_complete()

    def _handle_exploration_complete(self):
        if self.mission_phase == "exploring":
            path, _ = get_astar_path(self.grid._nodes, self._at_node, self.grid.start)
            if path:
                self.mission_phase = "going_home"
                self._start_path_sequence(path[:-1], path[-1], FSM.solve)
            else: self.FS = FSM.done
        elif self.mission_phase == "going_home":
            path, _ = get_astar_path(self.grid._nodes, self._at_node, self.goal_node)
            if path:
                self.mission_phase = "going_goal"
                self._start_path_sequence(path[:-1], path[-1], FSM.solve)
            else: self.FS = FSM.done
        else: self.FS = FSM.done

    def _start_path_sequence(self, waypoints, final_target, next_state):
        self._waypoints = list(waypoints)
        self._post_bt_target = final_target
        if self._waypoints:
            nxt = self._waypoints.pop(0)
            self.nav.set_target(nxt, self._at_node)
            self.FS = next_state
        else:
            self.nav.set_target(final_target, self._at_node)
            self.FS = FSM.drive

    def _do_path_follow(self, state):
        if self.nav.arrived(state):
            self._at_node = self.nav.target
            if self._waypoints:
                nxt = self._waypoints.pop(0)
                self.nav.set_target(nxt, self._at_node)
            else:
                self.nav.set_target(self._post_bt_target, self._at_node)
                self.FS = FSM.drive
            return None
        return self.nav.steer(state)

class Space:
    def __init__(self, width=400, height=400, fps=60, bg_color=(15,15,15), gravity=(0,0), caption="HiLo Explorer"):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.fps, self.dt = fps, 1/fps
        self.bg_color = bg_color
        self.physics = pymunk.Space()
        self.physics.gravity = gravity
        self.dynamic_objects, self.static_objects = [], []
        self.running, self.camera_offset, self.camera_target = True, [0, 0], None

    def set_camera_target(self, obj): self.camera_target = obj

    def _update_camera(self):
        if self.camera_target and self.camera_target.body:
            t_pos = self.camera_target.body.position
            target_x = (self.width / 2) - t_pos.x
            target_y = (self.height / 2) - t_pos.y
            self.camera_offset[0] += (target_x - self.camera_offset[0]) * 0.1
            self.camera_offset[1] += (target_y - self.camera_offset[1]) * 0.1

    def add_dynamic(self, obj): self.dynamic_objects.append(obj)
    def add_static(self, obj): self.static_objects.append(obj)

    def step(self):
        for obj in self.dynamic_objects: obj._update_forces(self.dt)
        self.physics.step(self.dt)
        self._update_camera()

    def draw(self):
        self.screen.fill(self.bg_color)
        for obj in self.static_objects: obj.update(self.screen, self.camera_offset)
        for obj in self.dynamic_objects: obj.update(self.screen, self.camera_offset)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self, update_handler=None):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            if update_handler: update_handler(self)
            self.step()
            self.draw()

class StatSegment:
    def __init__(self, position, length, angle, thickness, space, friction=0.8, elasticity=0.1, color=(45,15,15)):
        self.body = space.physics.static_body
        x, y = position
        dx, dy = length * math.cos(angle), length * math.sin(angle)
        self.shape = pymunk.Segment(self.body, (x, y), (x + dx, y + dy), thickness)
        self.shape.friction, self.shape.elasticity, self.color = friction, elasticity, color
        space.physics.add(self.shape)

    def update(self, screen, offset):
        pa = (self.shape.a[0] + offset[0], self.shape.a[1] + offset[1])
        pb = (self.shape.b[0] + offset[0], self.shape.b[1] + offset[1])
        pygame.draw.line(screen, self.color, pa, pb, int(self.shape.radius * 2))

class DiscoveredMapRenderer:
    def __init__(self, grid, goal_grid_pos):
        self.grid = grid
        self.goal_grid_pos = goal_grid_pos

    def update(self, screen, offset):
        COLOR, THICKNESS, HALF = (0, 255, 255), 3, self.grid.spacing / 2.0
        for node in self.grid._nodes.values():
            x, y = node.pos[0] + offset[0], node.pos[1] + offset[1]
            if node.walls['n'] is True: pygame.draw.line(screen, COLOR, (x - HALF, y + HALF), (x + HALF, y + HALF), THICKNESS)
            if node.walls['e'] is True: pygame.draw.line(screen, COLOR, (x + HALF, y - HALF), (x + HALF, y + HALF), THICKNESS)
            if node.walls['s'] is True: pygame.draw.line(screen, COLOR, (x - HALF, y - HALF), (x + HALF, y - HALF), THICKNESS)
            if node.walls['w'] is True: pygame.draw.line(screen, COLOR, (x - HALF, y - HALF), (x - HALF, y + HALF), THICKNESS)

        start_node = self.grid.start
        sx, sy = start_node.pos[0] + offset[0], start_node.pos[1] + offset[1]
        pygame.draw.circle(screen, (0, 255, 0), (int(sx), int(sy)), 15, 3)
        goal_node = self.grid.node_at(self.goal_grid_pos)
        if goal_node:
            gx, gy = goal_node.pos[0] + offset[0], goal_node.pos[1] + offset[1]
            pygame.draw.circle(screen, (255, 0, 0), (int(gx), int(gy)), 15, 3)

class DynamicObject:
    def __init__(self, mass, space, color, position):
        self.mass, self.space, self.color, self.position = mass, space, color, position
        self.body, self.shape, self.forces_to_apply = None, None, []
    def apply_force(self, force, offset=(0,0), duration=0.1):
        self.forces_to_apply.append({"force": force, "offset": offset, "duration": duration, "elapsed": 0.0})
    def _update_forces(self, dt):
        for f in self.forces_to_apply[:]:
            self.body.apply_force_at_world_point(f["force"], self.body.local_to_world(f["offset"]))
            f["elapsed"] += dt
            if f["elapsed"] >= f["duration"]: self.forces_to_apply.remove(f)

class DynCircle(DynamicObject):
    def __init__(self, mass, radius, friction, elasticity, space, color, position):
        super().__init__(mass, space, color, position)
        self.radius = radius
        self.body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        self.body.angle, self.body.position = -math.pi/2, position
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction, self.shape.elasticity = friction, elasticity
        self.space.physics.add(self.body, self.shape)
    def update(self, screen, offset):
        p, a = self.body.position, self.body.angle
        draw_pos = (int(p.x + offset[0]), int(p.y + offset[1]))
        pygame.draw.circle(screen, self.color, draw_pos, self.radius)
        pygame.draw.line(screen, (0,0,0), draw_pos, (draw_pos[0] + self.radius * math.cos(a), draw_pos[1] + self.radius * math.sin(a)), 2)

class SonarAgent:
    def __init__(self, rel_angle_deg, arm_length=0, fov=30, max_range=500):
        self.rel_angle_deg, self.arm_length, self.fov, self.max_range = rel_angle_deg, arm_length, fov, max_range
        self.origin, self.heading_rad, self.is_locked, self.best_hit_pos, self.best_conf = (0,0), 0, 0, None, 0.0

    def update_sync(self, body_pos, body_angle, radius, maze_lines):
        self.heading_rad = body_angle + math.radians(self.rel_angle_deg)
        self.origin = (body_pos.x + (radius + self.arm_length) * math.cos(self.heading_rad), 
                       body_pos.y + (radius + self.arm_length) * math.sin(self.heading_rad))
        self._perform_scan(maze_lines)

    def _perform_scan(self, walls):
        self.best_conf, self.best_hit_pos, self.is_locked = -1.0, None, 0
        base_deg = math.degrees(self.heading_rad)
        for rel_angle in range(-self.fov//2, self.fov//2 + 1, 5):
            angle_rad = math.radians(base_deg + rel_angle)
            ray = LineString([self.origin, (self.origin[0] + self.max_range * math.cos(angle_rad), self.origin[1] + self.max_range * math.sin(angle_rad))])
            closest = self.max_range
            hit = None
            for wall in walls:
                if ray.intersects(wall):
                    inter = ray.intersection(wall)
                    if isinstance(inter, Point):
                        d = math.dist(self.origin, inter.coords[0])
                        if d < closest: closest, hit = d, inter.coords[0]
            if hit:
                conf = 1.0 / (closest + 1.0)
                if conf > self.best_conf: self.best_conf, self.best_hit_pos, self.is_locked = conf, hit, 1

    def draw(self, screen, offset):
        if self.is_locked:
            dist = math.dist(self.origin, self.best_hit_pos)
            fake_hit = (self.origin[0] + dist * math.cos(self.heading_rad) + offset[0], self.origin[1] + dist * math.sin(self.heading_rad) + offset[1])
            pygame.draw.line(screen, (255, 0, 0), (self.origin[0]+offset[0], self.origin[1]+offset[1]), fake_hit, 1)
            pygame.draw.circle(screen, (255, 0, 0), (int(fake_hit[0]), int(fake_hit[1])), 3)

class Mouse(DynCircle):
    def __init__(self, space, position, radius, mass, color, maze_lines):
        super().__init__(mass, radius, 0.5, 0.1, space, color, position)
        self.maze_lines, self.mount_angles = maze_lines, [0, 180, 270, 90]
        self.sensors = [SonarAgent(rel_angle_deg=a) for a in self.mount_angles]
        self.telemetry = []

    def update(self, screen, offset):
        super().update(screen, offset)
        row = {'position': (self.body.position.x, self.body.position.y), 'heading': math.degrees(self.body.angle) % 360, 'readings': []}
        for s in self.sensors:
            s.update_sync(self.body.position, self.body.angle, self.radius, self.maze_lines)
            s.draw(screen, offset)
            row['readings'].append((math.dist(s.origin, s.best_hit_pos) if s.best_hit_pos else s.max_range, s.best_conf))
        self.telemetry.append(row)

    def apply_friction(self):
        self.body.velocity *= 0.85
        self.body.angular_velocity *= 0.65

    def turn(self, direction):
        pwr = 125
        if direction == 'f': 
            self._fire(180, 90, -pwr)
            self._fire(0, 270, -pwr)
        elif direction == 'r': 
            self._fire(180, 90, pwr/2)
            self._fire(0, 270, -pwr/2)
        elif direction == 'l': 
            self._fire(0, 270, pwr/2)
            self._fire(180, 90, -pwr/2)

    def _fire(self, m_angle, f_offset, pwr):
        theta = math.radians(m_angle - 90)
        f_angle = self.body.angle + theta + math.radians(f_offset)
        self.apply_force((pwr * math.cos(f_angle), pwr * math.sin(f_angle)), offset=(self.radius * math.cos(theta), self.radius * math.sin(theta)))

def build_synced_maze(space_wrapper, connections, spacing):
    lines = []
    for (r1, c1), (r2, c2) in connections:
        x1, y1, x2, y2 = c1*spacing, r1*spacing, c2*spacing, r2*spacing
        wall = StatSegment((x1, y1), math.dist((x1,y1), (x2,y2)), math.atan2(y2-y1, x2-x1), 2, space_wrapper)
        space_wrapper.add_static(wall)
        lines.append(LineString([(x1, y1), (x2, y2)]))
    return lines

def main():
    sim = Space(width=1000, height=800, fps=25)
    map_3 = [[[2, 3], [2, 4]], [[1, 2], [2, 2]], [[3, 2], [4, 2]], [[1, 1], [4, 1]], [[1, 1], [1, 4]],
             [[3, 2], [3, 3]], [[1, 4], [4, 4]], [[4, 1], [4, 4]]] # 3,4, 4,2
    map_9 = [[[4, 3], [4, 8]], [[2, 8], [2, 10]], [[7, 1], [7, 2]], [[5, 4], [5, 5]], [[5, 9], [6, 9]],
             [[9, 6], [10, 6]], [[6, 2], [6, 3]], [[1, 1], [10, 1]], [[3, 8], [4, 8]], [[1, 3], [2, 3]],
             [[2, 6], [5, 6]], [[6, 5], [6, 6]], [[9, 2], [9, 3]], [[5, 1], [5, 3]], [[5, 8], [5, 9]],
             [[8, 7], [10, 7]], [[2, 5], [2, 6]], [[6, 8], [8, 8]], [[1, 10], [10, 10]], [[7, 9], [7, 10]],
             [[9, 5], [9, 6]], [[4, 1], [4, 2]], [[6, 6], [7, 6]], [[3, 3], [4, 3]], [[7, 4], [9, 4]],
             [[7, 3], [7, 7]], [[6, 3], [7, 3]], [[1, 4], [3, 4]], [[3, 2], [3, 3]], [[9, 2], [10, 2]],
             [[5, 7], [7, 7]], [[4, 4], [7, 4]], [[1, 7], [3, 7]], [[8, 4], [8, 6]], [[8, 2], [8, 4]],
             [[8, 9], [10, 9]], [[9, 8], [9, 9]], [[10, 1], [10, 10]], [[1, 1], [1, 10]], [[3, 8], [3, 9]],
             [[4, 9], [4, 10]], [[2, 1], [2, 2]], [[6, 8], [6, 9]], [[3, 5], [4, 5]]] # 6,2, 6,10
    map_4 = [[[1, 5], [5, 5]], [[4, 1], [4, 2]], [[2, 2], [3, 2]], [[2, 4], [3, 4]], [[2, 4], [2, 5]],
             [[5, 1], [5, 5]], [[3, 2], [3, 3]], [[4, 3], [5, 3]], [[1, 3], [3, 3]], [[1, 1], [1, 5]],
             [[1, 1], [5, 1]], [[4, 3], [4, 4]]] # 5,2, 3,3
    
    spacing, sx, sy, gx, gy = 90, 6,2, 6,10
    maze_lines = build_synced_maze(sim, map_9, spacing)
    grid = Grid(spacing, {'x':sx, 'y':sy})
    player = Mouse(sim, grid.start.pos, 25, 2.0, (255, 255, 255), maze_lines)
    sim.add_dynamic(player)
    sim.set_camera_target(player)
    sim.add_static(DiscoveredMapRenderer(grid, (gx, gy)))
    tree = Tree(grid, player, (gx, gy))

    def update_loop(space_ref):
        player.apply_friction()
        if player.telemetry:
            move = tree.mind()
            if move: player.turn(move)

    sim.run(update_handler=update_loop)

if __name__ == "__main__":
    main()
