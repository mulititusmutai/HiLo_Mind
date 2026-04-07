import pygame
import math
import pymunk
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point
from enum import Enum, auto

def generate_hilo_map(csv_file='mouse_telemetry.csv', grid_res=60, wall_threshold=5, min_neighbors=1):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Plotter Error: {e}")
        return

    mount_angles = [0, 180, 270, 90]
    MOUSE_RADIUS = 50 
    map_pts_x, map_pts_y = [], []
    
    for _, row in df.iterrows():
        rx, ry = row['pos_x'], row['pos_y']
        r_heading = np.radians(row['heading'])
        for i, rel_angle in enumerate(mount_angles):
            dist = row[f's{i}_dist']
            if 10 < dist < 490:
                total_angle = r_heading + np.radians(rel_angle)
                wx = rx + (dist + MOUSE_RADIUS) * np.cos(total_angle)
                wy = ry + (dist + MOUSE_RADIUS) * np.sin(total_angle)
                map_pts_x.append(wx)
                map_pts_y.append(wy)

    counts, xedges, yedges = np.histogram2d(map_pts_x, map_pts_y, bins=grid_res)
    final_grid = np.zeros_like(counts)
    rows, cols = counts.shape
    
    for r in range(rows):
        for c in range(cols):
            if counts[r, c] >= wall_threshold:
                neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if counts[nr, nc] >= wall_threshold:
                                neighbors += 1
                
                if neighbors >= min_neighbors:
                    final_grid[r, c] = counts[r, c]

    plt.figure(figsize=(10, 8))
    plt.plot(df['pos_x'], df['pos_y'], color='cyan', alpha=0.3, label='Path')
    
    final_grid[final_grid == 0] = np.nan
    plt.imshow(final_grid.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
               cmap='magma', aspect='equal')
    
    plt.colorbar(label='Wall Confidence')
    plt.gca().invert_yaxis() 
    plt.axis('equal')
    plt.title(f"Filtered Map (Neighbors >= {min_neighbors})")
    plt.savefig('hilo_filtered_reconstruction.png', dpi=300)
    plt.show()

class FSM(Enum):
    scan      = auto()
    drive     = auto()
    backtrack = auto()
    done      = auto()

class Node:
    _id_counter = 0
    def __init__(self, grid_pos, world_pos):
        self.grid_pos   = grid_pos
        self.pos        = world_pos
        self.grid_links = {}
        self.passages   = set()
        self.id         = Node._id_counter
        Node._id_counter += 1

    def __repr__(self):
        return f"Node({self.grid_pos}, world={self.pos})"

class Grid:
    DIRS  = ('n', 'e', 's', 'w')
    DELTA = {'n': (0, 1), 'e': (1, 0), 's': (0, -1), 'w': (-1, 0)}

    def __init__(self, spacing: float, start_spec: dict):
        self.spacing = spacing
        self._nodes  = {}
        self.start   = None
        sx, sy = start_spec['x'], start_spec['y']
        HALF   = spacing / 2.0
        for gx in range(sx - 12, sx + 13):
            for gy in range(sy - 12, sy + 13):
                world = (gx * spacing - HALF, gy * spacing - HALF)
                node  = Node((gx, gy), world)
                self._nodes[(gx, gy)] = node
        for (gx, gy), node in self._nodes.items():
            for d, (ddx, ddy) in self.DELTA.items():
                nbr = self._nodes.get((gx + ddx, gy + ddy))
                node.grid_links[d] = nbr
        self.start = self._nodes[(sx, sy)]
        print(f"Grid ready. Start node: {self.start}")

    def node_at(self, grid_pos):
        return self._nodes.get(grid_pos)

class DFSExplorer:
    def __init__(self, start_node: Node):
        self.visited        = set()
        self._stack         = []
        self._came_from     = {}
        self._current       = start_node
        self._came_from[start_node.id] = None

    def record_scan(self, node: Node, open_passages: set):
        node.passages = open_passages
        self.visited.add(node.id)
        self._current = node

    def decide_next(self, node: Node):
        exits = []
        for d in node.passages:
            nbr = node.grid_links.get(d)
            if nbr is not None and nbr.id not in self.visited:
                exits.append(nbr)
        if exits:
            if len(exits) > 1:
                self._stack.append((node, exits[1:]))
            target = exits[0]
            self._came_from[target.id] = node
            return ('drive', target)
        return self._backtrack()

    def _backtrack(self):
        while self._stack:
            junction, remaining = self._stack[-1]
            fresh = [n for n in remaining if n.id not in self.visited]
            if fresh:
                self._stack[-1] = (junction, fresh[1:])
                if not self._stack[-1][1]:
                    self._stack.pop()
                target    = fresh[0]
                waypoints = self._path_to(junction)
                self._came_from[target.id] = junction
                return ('backtrack', waypoints, target)
            else:
                self._stack.pop()
        return ('done', None)

    def _path_to(self, destination: Node):
        path    = []
        cursor  = self._current
        seen    = set()
        while cursor is not destination and cursor is not None:
            if cursor.id in seen:
                break
            seen.add(cursor.id)
            parent = self._came_from.get(cursor.id)
            if parent is not None:
                path.append(parent)
            cursor = parent
        return path

class Navigator:
    ARRIVAL_RADIUS = 22.0
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
        if   err >  self.ANGLE_TOL:  return 'r'
        elif err < -self.ANGLE_TOL:  return 'l'
        else:                        return 'f'

_CARDINALS  = {90: 'n', 0: 'e', 270: 's', 180: 'w'}
_REAR_MOUNT = 180

def snap_to_cardinal(absolute_angle_deg: float):
    best, best_diff = None, 45.0
    for deg, label in _CARDINALS.items():
        diff = abs((absolute_angle_deg - deg + 180) % 360 - 180)
        if diff < best_diff:
            best_diff = diff
            best      = label
    return best

def read_open_passages(state: dict, mount_angles: list, open_threshold: float = 90.0) -> set:
    open_dirs = set()
    for reading, mount in zip(state['readings'], mount_angles):
        if mount == _REAR_MOUNT:
            continue
        dist, _conf = reading
        if dist > open_threshold:
            abs_angle = (mount + state['heading']) % 360
            cardinal  = snap_to_cardinal(abs_angle)
            if cardinal:
                open_dirs.add(cardinal)
    return open_dirs

class Tree:
    def __init__(self, grid: Grid, mouse):
        self.grid  = grid
        self.mouse = mouse
        self.dfs = DFSExplorer(grid.start)
        self.nav = Navigator()
        self._waypoints      = []
        self._post_bt_target = None
        self._at_node = grid.start
        self.FS       = FSM.scan
        print(f"Tree initialised. Start: {grid.start.pos}")

    def mind(self):
        if not self.mouse.telemetry:
            return None
        state = self.mouse.telemetry[-1]
        if self.FS == FSM.scan:
            self._do_scan(state)
            return None
        elif self.FS == FSM.drive:
            if self.nav.arrived(state):
                self._at_node = self.nav.target
                print(f"Arrived at {self._at_node.pos}")
                self.FS = FSM.scan
                return None
            return self.nav.steer(state)
        elif self.FS == FSM.backtrack:
            return self._do_backtrack(state)
        elif self.FS == FSM.done:
            return None

    def _do_scan(self, state):
        node          = self._at_node
        open_passages = read_open_passages(state, self.mouse.mount_angles)
        self.dfs.record_scan(node, open_passages)
        action = self.dfs.decide_next(node)
        if action[0] == 'drive':
            _, target = action
            print(f"Driving to {target.pos}")
            self.nav.set_target(target, node)
            self.FS = FSM.drive
        elif action[0] == 'backtrack':
            _, waypoints, final_target = action
            print(f"Dead end / exhausted. Backtracking {len(waypoints)} hops.")
            self._start_backtrack(waypoints, final_target)
        elif action[0] == 'done':
            print("DFS complete — all nodes explored.")
            self.FS = FSM.done

    def _start_backtrack(self, waypoints: list, final_target: Node):
        self._waypoints      = list(waypoints)
        self._post_bt_target = final_target
        if self._waypoints:
            nxt = self._waypoints.pop(0)
            self.nav.set_target(nxt, self._at_node)
            self.FS = FSM.backtrack
        else:
            self._finish_backtrack()

    def _do_backtrack(self, state):
        if self.nav.arrived(state):
            self._at_node = self.nav.target
            print(f"Backtrack hop reached {self._at_node.pos}")
            if self._waypoints:
                nxt = self._waypoints.pop(0)
                self.nav.set_target(nxt, self._at_node)
            else:
                self._finish_backtrack()
            return None
        return self.nav.steer(state)

    def _finish_backtrack(self):
        target = self._post_bt_target
        print(f"Backtrack done. Driving to fresh exit: {target.pos}")
        self.nav.set_target(target, self._at_node)
        self.FS = FSM.drive

class Space:
    def __init__(self, width=400, height=400, fps=60, bg_color=(0,0,0), gravity=(0,981), caption="Pymunk Example"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.dt = 1 / fps
        self.bg_color = bg_color
        self.physics = pymunk.Space()
        self.physics.gravity = gravity
        self.dynamic_objects = []
        self.static_objects = []
        self.running = True
        self.camera_offset = [0, 0]
        self.camera_target = None

    def set_camera_target(self, obj):
        if self.camera_target is None:
            self.camera_target = obj
        else:
            print("Action Denied: Camera target is already locked.")

    def _update_camera(self):
        if self.camera_target and self.camera_target.body:
            t_pos = self.camera_target.body.position
            target_x = (self.width / 2) - t_pos.x
            target_y = (self.height / 2) - t_pos.y
            self.camera_offset[0] += (target_x - self.camera_offset[0]) * 0.1
            self.camera_offset[1] += (target_y - self.camera_offset[1]) * 0.1

    def add_dynamic(self, obj):
        self.dynamic_objects.append(obj)

    def add_static(self, obj):
        self.static_objects.append(obj)

    def step(self):
        for obj in self.dynamic_objects:
            obj._update_forces(self.dt)
        self.physics.step(self.dt)
        self._update_camera()

    def draw(self):
        self.screen.fill(self.bg_color)
        for obj in self.static_objects:
            obj.update(self.screen, self.camera_offset)
        for obj in self.dynamic_objects:
            obj.update(self.screen, self.camera_offset)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self, event_handler=None, update_handler=None):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event_handler:
                    event_handler(event, self)
            if update_handler:
                update_handler(self)
            self.step()
            self.draw()

class StaticObject:
    def __init__(self, position, space, friction=0.8, elasticity=0.99, color=(255,255,255)):
        self.position = position
        self.space = space
        self.friction = friction
        self.elasticity = elasticity
        self.color = color
        self.body = space.physics.static_body
        self.shape = None

class StatSegment(StaticObject):
    def __init__(self, position, length, angle, thickness, space, friction=0.8, elasticity=0.99, color=(255,255,255)):
        super().__init__(position, space, friction, elasticity, color)
        x, y = self.position
        dx = length * math.cos(angle)
        dy = length * math.sin(angle)
        self.shape = pymunk.Segment(self.body, (x, y), (x + dx, y + dy), thickness)
        self.shape.friction = self.friction
        self.shape.elasticity = self.elasticity
        self.space.physics.add(self.shape)

    def update(self, screen, offset):
        pa = (self.shape.a[0] + offset[0], self.shape.a[1] + offset[1])
        pb = (self.shape.b[0] + offset[0], self.shape.b[1] + offset[1])
        pygame.draw.line(screen, self.color, pa, pb, int(self.shape.radius * 2))

class DynamicObject:
    def __init__(self, mass, friction, elasticity, space, color, position):
        self.mass = mass
        self.friction = friction
        self.elasticity = elasticity
        self.space = space
        self.color = color
        self.position = position
        self.body = None
        self.shape = None
        self.forces_to_apply = []
        self.impulses_to_apply = []

    def apply_force(self, force, offset=(0,0), duration=0.1):
        self.forces_to_apply.append({"force": force, "offset": offset, "duration": duration, "elapsed": 0.0})

    def apply_impulse(self, impulse, offset=(0,0)):
        self.impulses_to_apply.append({"impulse": impulse, "offset": offset})

    def _update_forces(self, dt):
        for f in self.forces_to_apply[:]:
            world_point = self.body.local_to_world(f["offset"])
            self.body.apply_force_at_world_point(f["force"], world_point)
            f["elapsed"] += dt
            if f["elapsed"] >= f["duration"]:
                self.forces_to_apply.remove(f)
        for imp in self.impulses_to_apply[:]:
            world_point = self.body.local_to_world(imp["offset"])
            self.body.apply_impulse_at_world_point(imp["impulse"], world_point)
        self.impulses_to_apply.clear()

    def update(self, screen, offset):
        pass

class DynCircle(DynamicObject):
    def __init__(self, mass, radius, friction, elasticity, space, color, position):
        super().__init__(mass, friction, elasticity, space, color, position)
        self.radius = radius
        moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.body = pymunk.Body(self.mass, moment)
        #self.body.angle -= math.pi/2
        self.body.position = self.position        
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = self.friction
        self.shape.elasticity = self.elasticity
        self.space.physics.add(self.body, self.shape)

    def update(self, screen, offset):
        pos = self.body.position        
        angle = self.body.angle
        draw_pos = (int(pos.x + offset[0]), int(pos.y + offset[1]))
        pygame.draw.circle(screen, self.color, draw_pos, self.radius)
        line_end_x = draw_pos[0] + self.radius * math.cos(angle)
        line_end_y = draw_pos[1] + self.radius * math.sin(angle)
        pygame.draw.line(screen, (0,0,0), draw_pos, (line_end_x, line_end_y), 2)

class SonarAgent:
    def __init__(self, rel_angle_deg, arm_length, fov=30, max_range=500):
        self.rel_angle_deg = rel_angle_deg
        self.arm_length = arm_length
        self.fov = fov
        self.max_range = max_range
        self.origin = (0, 0)
        self.heading_rad = 0
        self.is_locked = 0
        self.best_hit_pos = None
        self.best_conf = 0.0
        self.rejection_threshold = 0.001

    def update_sync(self, body_pos, body_angle, radius, maze_lines):
        self.heading_rad = body_angle + math.radians(self.rel_angle_deg)
        self.origin = (body_pos.x + (radius + self.arm_length) * math.cos(self.heading_rad), body_pos.y + (radius + self.arm_length) * math.sin(self.heading_rad))
        self._perform_scan(maze_lines)

    def _perform_scan(self, walls):
        self.best_conf = -1.0
        self.best_hit_pos = None
        self.is_locked = 0
        base_deg = math.degrees(self.heading_rad)
        for rel_angle in range(-self.fov//2, self.fov//2 + 1, 3):
            angle_rad = math.radians(base_deg + rel_angle)
            ray_vec = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            end_x = self.origin[0] + self.max_range * math.cos(angle_rad)
            end_y = self.origin[1] + self.max_range * math.sin(angle_rad)
            ray_line = LineString([self.origin, (end_x, end_y)])
            closest_dist = self.max_range
            hit_point = None
            intensity = 0.0
            for wall in walls:
                if ray_line.intersects(wall):
                    inter = ray_line.intersection(wall)
                    if isinstance(inter, Point):
                        p = inter.coords[0]
                        d = math.dist(self.origin, p)
                        if d < closest_dist:
                            closest_dist = d
                            hit_point = p                                            
                            coords = list(wall.coords)
                            dx, dy = coords[1][0]-coords[0][0], coords[1][1]-coords[0][1]
                            norm = np.array([-dy, dx])
                            norm /= (np.linalg.norm(norm) + 1e-6)
                            intensity = max(0, abs(np.dot(ray_vec, norm)))
            conf = (intensity**2) / (closest_dist + 0.1)
            if conf > self.best_conf:
                if hit_point:
                    if math.dist(self.origin, hit_point) >= 10:
                        self.best_conf = conf
                        self.best_hit_pos = hit_point
                    else:
                        self.is_locked = 2
        if self.best_conf >= self.rejection_threshold and self.is_locked != 2:
            self.is_locked = 1

    def draw(self, screen, offset):
        start_x = self.origin[0] + offset[0] - (self.arm_length * math.cos(self.heading_rad))
        start_y = self.origin[1] + offset[1] - (self.arm_length * math.sin(self.heading_rad))
        end_x, end_y = self.origin[0] + offset[0], self.origin[1] + offset[1]
        pygame.draw.line(screen, (130, 130, 150), (start_x, start_y), (end_x, end_y), 2)
        if self.is_locked == 1 and self.best_hit_pos:
            dist = math.dist(self.origin, self.best_hit_pos)
            proj_hit_x = self.origin[0] + dist * math.cos(self.heading_rad)
            proj_hit_y = self.origin[1] + dist * math.sin(self.heading_rad)
            hit_scr = (proj_hit_x + offset[0], proj_hit_y + offset[1])
            pygame.draw.line(screen, (0, 255, 150), (end_x, end_y), hit_scr, 1)
            pygame.draw.circle(screen, (0, 255, 0), (int(hit_scr[0]), int(hit_scr[1])), 4)

class Mouse(DynCircle):
    def __init__(self, space, position, radius, mass, color, maze_lines):
        super().__init__(mass, radius, 0.5, 0.1, space, color, position)
        self.maze_lines = maze_lines
        self.mount_angles = [0, 180, 270, 90]
        self.sensors = []
        for angle in self.mount_angles:
            new_sensor = SonarAgent(rel_angle_deg=angle, arm_length=0)
            self.sensors.append(new_sensor)
        self.telemetry = []
        self.map_points = [] 

    def update(self, screen, offset):
        super().update(screen, offset)
        heading_deg = round(math.degrees(self.body.angle) % 360, 2)
        row = {
            'position': (round(self.body.position.x, 2), round(self.body.position.y, 2)),
            'heading': round(heading_deg, 2),
            'readings': []
        }
        for s in self.sensors:
            s.update_sync(self.body.position, self.body.angle, self.radius, self.maze_lines)
            s.draw(screen, offset)
            if s.is_locked == 1 and s.best_hit_pos:
                dist = round(math.dist(s.origin, s.best_hit_pos), 2)
                conf = round(s.best_conf, 5)
                row['readings'].append((dist, conf))
                proj_x = s.origin[0] + dist * math.cos(s.heading_rad)
                proj_y = s.origin[1] + dist * math.sin(s.heading_rad)
                if len(self.telemetry) % 2 == 0:
                    self.map_points.append((proj_x, proj_y, conf))
            elif s.is_locked == 0:
                row['readings'].append((s.max_range, 0.0))
            elif s.is_locked == 2:
                row['readings'].append((0.0, 0.0))
        self.telemetry.append(row)
        for pt in self.map_points:
            px, py, raw_conf = pt
            norm_c = min(1.0, max(0.0, (raw_conf - 0.001) / 0.019))
            r, g, b = int(255 * (1 - norm_c)), int(255 * norm_c), 30 
            plot_x, plot_y = int(px + offset[0]), int(py + offset[1])
            pygame.draw.circle(screen, (r, g, b), (plot_x, plot_y), 3)

    def save_to_csv(self, filename="telemetry_log.csv"):
        if not self.telemetry:
            return
        flat_logs = []
        for entry in self.telemetry:
            flat_row = {'pos_x': entry['position'][0], 'pos_y': entry['position'][1], 'heading': entry['heading']}
            for i, (dist, conf) in enumerate(entry['readings']):
                flat_row[f's{i}_dist'] = dist
                flat_row[f's{i}_conf'] = conf
            flat_logs.append(flat_row)
        keys = flat_logs[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(flat_logs)
        print(f"Telemetry successfully saved to {filename}")

    def apply_friction(self):
        self.body.velocity *= 0.85
        self.body.angular_velocity *= 0.65

    def turn(self, direction):
        pwr = 50
        if direction == 'f': 
            self.fire_engine(180, self.radius, 90, -pwr)
            self.fire_engine(0, self.radius, 270, -pwr)
        elif direction == 'r': 
            self.fire_engine(180, self.radius, 90, pwr/2)
            self.fire_engine(0, self.radius, 270, -pwr/2)
        elif direction == 'l': 
            self.fire_engine(0, self.radius, 270, pwr/2)
            self.fire_engine(180, self.radius, 90, -pwr/2)

    def fire_engine(self, m_angle, m_dist, f_offset, pwr):
        theta_m = math.radians(m_angle - 90)
        offset = (m_dist * math.cos(theta_m), m_dist * math.sin(theta_m))
        force_angle = self.body.angle + theta_m + math.radians(f_offset)
        fx, fy = pwr * math.cos(force_angle), pwr * math.sin(force_angle)
        self.apply_force((fx, fy), offset=offset, duration=0.1)

def build_synced_maze(space_wrapper, connections, spacing):
    shapely_lines = []
    for start_node, end_node in connections:
        r1, c1 = start_node
        r2, c2 = end_node
        x1, y1 = c1 * spacing, r1 * spacing
        x2, y2 = c2 * spacing, r2 * spacing
        dx, dy = x2 - x1, y2 - y1
        length, angle = math.sqrt(dx**2 + dy**2), math.atan2(dy, dx)
        wall = StatSegment((x1, y1), length, angle, 2, space_wrapper, 0.8, 0.1, (0, 0, 0))
        space_wrapper.add_static(wall)
        shapely_lines.append(LineString([(x1, y1), (x2, y2)]))
    return shapely_lines

def main():
    sim = Space(width=1000, height=800, gravity=(0, 0), fps=12, caption="HLM Explorer")
    map_a = [      
             ((0,0), (0,4)), ((0,4), (0,6)), ((0,6), (6,6)), ((6,6), (6,2)), ((6,2), (6,0)), ((6,0), (0,0)),
             ((1,0), (1,2)), ((1,3), (1,5)), ((2,2), (2,3)), ((3,1), (3,4)), ((4,1), (4,2)), ((4,4), (4,5)), ((5,2), (5,4)),
             ((2,1), (3,1)), ((4,1), (6,1)), ((1,2), (2,2)), ((1,3), (2,3)), ((3,3), (5,3)), ((2,4), (3,4)), ((1,5), (5,5))
    ]
    map_b = [((0,0), (0,3)), ((0,3), (3,3)), ((3,3), (3,0)), ((3,0), (0,0)), 
             ((1,1), (1,2)), ((2,1), (2,2)), ((2,1), (3,1))]
    connections, sx, sy, spacing = map_a, 1, 6, 90
    
    maze_lines = build_synced_maze(sim, connections, spacing=spacing)
    grid = Grid(spacing, {'x':sx, 'y':sy})
    player = Mouse(sim, grid.start.pos, 25, 2.0, (50, 255, 150), maze_lines)
    
    sim.add_dynamic(player)
    sim.set_camera_target(player)
    tree = Tree(grid, player)

    def update_loop(space_ref):        
        player.apply_friction()
        if player.telemetry:
            move = tree.mind()
            if move:
                player.turn(move)        

    sim.run(update_handler=update_loop)

    csv_name = "mouse_telemetry.csv"
    player.save_to_csv(csv_name)
    
    generate_hilo_map(csv_file=csv_name, grid_res=50, wall_threshold=6, min_neighbors=2) #map_a
    generate_hilo_map(grid_res=50, wall_threshold=4, min_neighbors=1) # map_b

if __name__ == "__main__":
    main()
