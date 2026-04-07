#WORKS. MOUSE MERGED WITH PYMUNK. A BIT TOO STRIPPED DOWN
import pygame
import math
import pymunk
import numpy as np
import csv
from shapely.geometry import LineString, Point
from enum import Enum, auto

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
        self.walls      = {'n': None, 'e': None, 's': None, 'w': None}
        self.id         = Node._id_counter
        Node._id_counter += 1

class Grid:
    DIRS  = ('n', 'e', 's', 'w')
    DELTA = {'n': (0, 1), 'e': (1, 0), 's': (0, -1), 'w': (-1, 0)}

    def __init__(self, spacing: float, start_spec: dict):
        self.spacing = spacing
        self._nodes  = {}
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

class DFSExplorer:
    def __init__(self, start_node: Node):
        self.visited        = set()
        self._stack         = []
        self._came_from     = {start_node.id: None}
        self._current       = start_node

    def record_scan(self, node: Node, open_passages: set):
        node.passages = open_passages
        self.visited.add(node.id)
        self._current = node

    def decide_next(self, node: Node):
        exits = [node.grid_links[d] for d in node.passages if node.grid_links[d] and node.grid_links[d].id not in self.visited]
        if exits:
            if len(exits) > 1: self._stack.append((node, exits[1:]))
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
                if not fresh[1:]: self._stack.pop()
                target = fresh[0]
                self._came_from[target.id] = junction
                return ('backtrack', self._path_to(junction), target)
            self._stack.pop()
        return ('done', None)

    def _path_to(self, destination: Node):
        path, cursor, seen = [], self._current, set()
        while cursor and cursor is not destination and cursor.id not in seen:
            seen.add(cursor.id)
            parent = self._came_from.get(cursor.id)
            if parent: path.append(parent)
            cursor = parent
        return path

class Navigator:
    def __init__(self, arrival_rad=22.0, angle_tol=8.0):
        self.target, self.arrival_rad, self.angle_tol = None, arrival_rad, angle_tol

    def set_target(self, target): self.target = target
    def arrived(self, pos): return math.dist(self.target.pos, pos) < self.arrival_rad
    def steer(self, pos, heading):
        dx, dy = self.target.pos[0] - pos[0], self.target.pos[1] - pos[1]
        err = (math.degrees(math.atan2(dy, dx)) - heading + 180) % 360 - 180
        return 'r' if err > self.angle_tol else 'l' if err < -self.angle_tol else 'f'

class Tree:
    def __init__(self, grid, mouse):
        self.grid, self.mouse, self.dfs, self.nav = grid, mouse, DFSExplorer(grid.start), Navigator()
        self._at_node, self.FS, self._waypoints, self._post_bt_target = grid.start, FSM.scan, [], None

    def mind(self):
        if not self.mouse.telemetry: return None
        state = self.mouse.telemetry[-1]
        pos, heading = state['position'], state['heading']

        if self.FS == FSM.scan:
            self._do_scan(state)
        elif self.FS == FSM.drive:
            if self.nav.arrived(pos): 
                self._at_node, self.FS = self.nav.target, FSM.scan
            else: return self.nav.steer(pos, heading)
        elif self.FS == FSM.backtrack:
            if self.nav.arrived(pos):
                self._at_node = self.nav.target
                if self._waypoints: self.nav.set_target(self._waypoints.pop(0))
                else: 
                    self.nav.set_target(self._post_bt_target)
                    self.FS = FSM.drive
            else: return self.nav.steer(pos, heading)
        return None

    def _do_scan(self, state):
        mounts, readings = self.mouse.mount_angles, state['readings']
        wall_data = {}
        for (dist, conf), mount in zip(readings, mounts):
            card = {90:'n', 0:'e', 270:'s', 180:'w'}.get(round((mount + state['heading']) / 90) * 90 % 360)
            if card: wall_data[card] = dist <= 120.0
        
        for d, is_wall in wall_data.items(): self._at_node.walls[d] = is_wall
        action = self.dfs.record_scan(self._at_node, {d for d, w in wall_data.items() if not w}) or self.dfs.decide_next(self._at_node)
        
        if action[0] == 'drive':
            self.nav.set_target(action[1]); self.FS = FSM.drive
        elif action[0] == 'backtrack':
            self._waypoints, self._post_bt_target = action[1], action[2]
            if self._waypoints: self.nav.set_target(self._waypoints.pop(0)); self.FS = FSM.backtrack
            else: self.nav.set_target(self._post_bt_target); self.FS = FSM.drive

class Space:
    def __init__(self, width=1000, height=800, fps=15):
        pygame.init()
        self.screen, self.clock, self.fps, self.dt = pygame.display.set_mode((width, height)), pygame.time.Clock(), fps, 1/fps
        self.physics, self.dynamic_objects, self.static_objects = pymunk.Space(), [], []
        self.camera_offset, self.camera_target, self.running = [0, 0], None, True

    def step(self):
        for obj in self.dynamic_objects: obj._update_forces(self.dt)
        self.physics.step(self.dt)
        if self.camera_target:
            t_pos = self.camera_target.body.position
            self.camera_offset[0] += ((self.screen.get_width()/2 - t_pos.x) - self.camera_offset[0]) * 0.1
            self.camera_offset[1] += ((self.screen.get_height()/2 - t_pos.y) - self.camera_offset[1]) * 0.1

    def draw(self):
        self.screen.fill((10,10,10))
        for obj in self.static_objects + self.dynamic_objects: obj.update(self.screen, self.camera_offset)
        pygame.display.flip(); self.clock.tick(self.fps)

    def run(self, update_handler):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            update_handler(self); self.step(); self.draw()

class StatSegment:
    def __init__(self, pos, length, angle, space):
        self.shape = pymunk.Segment(space.physics.static_body, pos, (pos[0]+length*math.cos(angle), pos[1]+length*math.sin(angle)), 2)
        self.shape.friction, self.shape.elasticity = 0.8, 0.1
        space.physics.add(self.shape)

    def update(self, screen, offset):
        pa, pb = (self.shape.a[0]+offset[0], self.shape.a[1]+offset[1]), (self.shape.b[0]+offset[0], self.shape.b[1]+offset[1])
        pygame.draw.line(screen, (40,40,40), pa, pb, 4)

class DiscoveredMapRenderer:
    def __init__(self, grid): self.grid = grid
    def update(self, screen, offset):
        COLOR, THICK, HALF = (0, 255, 255), 3, self.grid.spacing / 2.0
        for node in self.grid._nodes.values():
            x, y = node.pos[0] + offset[0], node.pos[1] + offset[1]
            if node.walls['n'] is True: pygame.draw.line(screen, COLOR, (x-HALF, y+HALF), (x+HALF, y+HALF), THICK)
            if node.walls['e'] is True: pygame.draw.line(screen, COLOR, (x+HALF, y-HALF), (x+HALF, y+HALF), THICK)
            if node.walls['s'] is True: pygame.draw.line(screen, COLOR, (x-HALF, y-HALF), (x+HALF, y-HALF), THICK)
            if node.walls['w'] is True: pygame.draw.line(screen, COLOR, (x-HALF, y-HALF), (x-HALF, y+HALF), THICK)

class SonarAgent:
    def __init__(self, rel_angle_deg):
        self.rel_angle_deg, self.max_range, self.origin, self.heading_rad = rel_angle_deg, 500, (0,0), 0
        self.is_locked, self.best_hit_pos, self.best_conf = 0, None, 0.0

    def update_sync(self, pos, angle, radius, walls):
        self.heading_rad = angle + math.radians(self.rel_angle_deg)
        self.origin = (pos.x + radius * math.cos(self.heading_rad), pos.y + radius * math.sin(self.heading_rad))
        self.best_conf, self.best_hit_pos, self.is_locked = -1.0, None, 0
        
        for rel in range(-15, 16, 5):
            a_rad = self.heading_rad + math.radians(rel)
            ray = LineString([self.origin, (self.origin[0]+self.max_range*math.cos(a_rad), self.origin[1]+self.max_range*math.sin(a_rad))])
            closest, hit, intensity = self.max_range, None, 0.0
            
            for wall in walls:
                if ray.intersects(wall):
                    inter = ray.intersection(wall)
                    if isinstance(inter, Point):
                        d = math.dist(self.origin, inter.coords[0])
                        if d < closest:
                            closest, hit = d, inter.coords[0]
                            coords = list(wall.coords)
                            norm = np.array([-(coords[1][1]-coords[0][1]), coords[1][0]-coords[0][0]])
                            intensity = abs(np.dot(np.array([math.cos(a_rad), math.sin(a_rad)]), norm/(np.linalg.norm(norm)+1e-6)))
            
            conf = (intensity**2) / (closest + 1.0)
            if conf > self.best_conf and hit:
                self.best_conf, self.best_hit_pos, self.is_locked = conf, hit, 1

class Mouse:
    def __init__(self, space, pos, radius, maze_lines):
        moment = pymunk.moment_for_circle(2.0, 0, radius)
        self.body = pymunk.Body(2.0, moment)
        self.body.position, self.body.angle = pos, -math.pi/2
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction, self.shape.elasticity = 0.5, 0.1
        space.physics.add(self.body, self.shape)
        self.radius, self.maze_lines, self.mount_angles = radius, maze_lines, [0, 180, 270, 90]
        self.sensors = [SonarAgent(a) for a in self.mount_angles]
        self.telemetry, self.map_points, self.forces = [], [], []

    def update(self, screen, offset):
        pos, angle = self.body.position, self.body.angle
        row = {'position': (pos.x, pos.y), 'heading': math.degrees(angle)%360, 'readings': []}
        for s in self.sensors:
            s.update_sync(pos, angle, self.radius, self.maze_lines)
            if s.is_locked and s.best_hit_pos:
                dist = math.dist(s.origin, s.best_hit_pos)
                row['readings'].append((dist, s.best_conf))
                # Store persistent debug points
                if len(self.telemetry) % 2 == 0: self.map_points.append((s.best_hit_pos, s.best_conf))
                pygame.draw.line(screen, (0,255,100), (s.origin[0]+offset[0], s.origin[1]+offset[1]), (s.best_hit_pos[0]+offset[0], s.best_hit_pos[1]+offset[1]), 1)
            else: row['readings'].append((s.max_range, 0.0))
        self.telemetry.append(row)
        
        # Draw persistent mapped points
        for pt, conf in self.map_points:
            c = min(255, int(conf * 5000))
            pygame.draw.circle(screen, (255-c, c, 50), (int(pt[0]+offset[0]), int(pt[1]+offset[1])), 2)
            
        draw_p = (int(pos.x+offset[0]), int(pos.y+offset[1]))
        pygame.draw.circle(screen, (200,200,200), draw_p, self.radius)
        pygame.draw.line(screen, (0,0,0), draw_p, (draw_p[0]+self.radius*math.cos(angle), draw_p[1]+self.radius*math.sin(angle)), 2)

    def _update_forces(self, dt):
        for f in self.forces[:]:
            self.body.apply_force_at_world_point(f[0], self.body.local_to_world(f[1]))
            f[2] += dt
            if f[2] >= 0.1: self.forces.remove(f)
        self.body.velocity *= 0.85; self.body.angular_velocity *= 0.65

    def turn(self, d):
        p = 150
        if d == 'f': [self.forces.append([(p*math.cos(a), p*math.sin(a)), (self.radius*math.cos(o), self.radius*math.sin(o)), 0]) for a, o in [(self.body.angle, math.pi), (self.body.angle, 0)]]
        elif d == 'r': self.forces.append([(p*math.cos(self.body.angle+math.pi/2), p*math.sin(self.body.angle+math.pi/2)), (self.radius, 0), 0])
        elif d == 'l': self.forces.append([(p*math.cos(self.body.angle-math.pi/2), p*math.sin(self.body.angle-math.pi/2)), (-self.radius, 0), 0])

def main():
    sim = Space()
    map_data, spacing, sx, sy = [((0,0), (0,3)), ((0,3), (3,3)), ((3,3), (3,0)), ((3,0), (0,0)), ((1,1), (1,2)), ((2,1), (2,2)), ((2,1), (3,1))], 180, 1, 3
    maze_lines = [LineString([(s[1]*spacing, s[0]*spacing), (e[1]*spacing, e[0]*spacing)]) for s, e in map_data]
    for m in maze_lines: sim.static_objects.append(StatSegment(m.coords[0], m.length, math.atan2(m.coords[1][1]-m.coords[0][1], m.coords[1][0]-m.coords[0][0]), sim))
    
    grid = Grid(spacing, {'x':sx, 'y':sy})
    player = Mouse(sim, grid.start.pos, 45, maze_lines)
    sim.dynamic_objects.append(player); sim.camera_target = player
    sim.static_objects.append(DiscoveredMapRenderer(grid))
    tree = Tree(grid, player)

    sim.run(lambda s: player.turn(tree.mind()) if player.telemetry else None)

if __name__ == "__main__": main()
