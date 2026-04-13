import pygame
import math
import pymunk
from shapely.geometry import LineString, Point
from enum import Enum, auto

# --- LOGIC CONSTANTS ---
class BrainState(Enum):
    ACTIVE = auto()
    DONE   = auto()

class Node:
    _id_counter = 0
    def __init__(self, grid_pos, world_pos):
        self.grid_pos   = grid_pos
        self.pos        = world_pos
        self.grid_links = {}
        # Wall state: None (unknown), True (wall), False (passage)
        self.walls      = {'n': None, 'e': None, 's': None, 'w': None}
        self.value      = float('inf') 
        self.id         = Node._id_counter
        Node._id_counter += 1

class Grid:
    DIRS  = ('n', 'e', 's', 'w')
    DELTA = {'n': (0, 1), 'e': (1, 0), 's': (0, -1), 'w': (-1, 0)}

    def __init__(self, spacing: float, start_spec: dict):
        self.spacing = spacing
        self._nodes  = {}
        sx, sy = start_spec['x'], start_spec['y']
        
        for gx in range(sx - 5, sx + 10):
            for gy in range(sy - 5, sy + 10):
                world = (gx * spacing, gy * spacing)
                node  = Node((gx, gy), world)
                self._nodes[(gx, gy)] = node
        
        for (gx, gy), node in self._nodes.items():
            for d, (ddx, ddy) in self.DELTA.items():
                node.grid_links[d] = self._nodes.get((gx + ddx, gy + ddy))
        self.start = self._nodes.get((sx, sy))

    def node_at(self, grid_pos):
        return self._nodes.get(grid_pos)

    def reflood(self, goal_node):
        for n in self._nodes.values(): 
            n.value = float('inf')
        if not goal_node: return
        
        goal_node.value = 0
        queue = [goal_node]
        while queue:
            curr = queue.pop(0)
            for d, nbr in curr.grid_links.items():
                # Optimistic pathfinding: flow through False (path) or None (unknown)
                if nbr and curr.walls[d] is not True and nbr.value == float('inf'):
                    nbr.value = curr.value + 1
                    queue.append(nbr)

class Navigator:
    ARRIVAL_RADIUS = 25.0
    ANGLE_TOL      = 10.0
    def __init__(self):
        self.target = None

    def set_target(self, target: Node):
        self.target = target

    def arrived(self, state: dict) -> bool:
        if not self.target: return False
        return math.dist(self.target.pos, state['position']) < self.ARRIVAL_RADIUS

    def steer(self, state: dict) -> str:
        if not self.target: return None
        dx, dy = self.target.pos[0] - state['position'][0], self.target.pos[1] - state['position'][1]
        desired = math.degrees(math.atan2(dy, dx)) % 360.0
        err = (desired - state['heading'] + 180) % 360 - 180
        if   err >  self.ANGLE_TOL: return 'r'
        elif err < -self.ANGLE_TOL: return 'l'
        else:                       return 'f'

# --- SENSOR UTILS ---
_CARDINALS  = {90: 'n', 0: 'e', 270: 's', 180: 'w'}
def snap_to_cardinal(absolute_angle_deg: float):
    best, best_diff = None, 45.0
    for deg, label in _CARDINALS.items():
        diff = abs((absolute_angle_deg - deg + 180) % 360 - 180)
        if diff < best_diff:
            best_diff, best = diff, label
    return best

class Tree:
    def __init__(self, grid: Grid, mouse, goal_grid_pos=(5, 5)):
        self.grid = grid
        self.mouse = mouse
        self.nav = Navigator()
        self._at_node = grid.start
        self.goal_node = grid.node_at(goal_grid_pos)
        self.state = BrainState.ACTIVE
        self.grid.reflood(self.goal_node)

    def mind(self):
        if not self.mouse.telemetry or self.state == BrainState.DONE: return None
        state = self.mouse.telemetry[-1]

        # 1. Check Arrival
        if self.nav.target and self.nav.arrived(state):
            self._at_node = self.nav.target
            if self._at_node == self.goal_node:
                self.state = BrainState.DONE
                return None

        # 2. Update Map from Sensors
        map_changed = False
        for reading, mount in zip(state['readings'], self.mouse.mount_angles):
            dist, _ = reading
            abs_angle = (mount + state['heading']) % 360
            cardinal = snap_to_cardinal(abs_angle)
            if cardinal:
                is_wall = dist < 85.0 # Detection threshold
                if self._at_node.walls[cardinal] != is_wall:
                    self._at_node.walls[cardinal] = is_wall
                    nbr = self._at_node.grid_links.get(cardinal)
                    if nbr:
                        opp = {'n':'s', 's':'n', 'e':'w', 'w':'e'}[cardinal]
                        nbr.walls[opp] = is_wall
                    map_changed = True

        if map_changed: 
            self.grid.reflood(self.goal_node)

        # 3. Choose target (Gradient Descent)
        best_nbr, min_val = self._at_node, self._at_node.value
        for d, wall_active in self._at_node.walls.items():
            if wall_active is not True:
                nbr = self._at_node.grid_links.get(d)
                if nbr and nbr.value < min_val:
                    min_val, best_nbr = nbr.value, nbr

        self.nav.set_target(best_nbr)
        return self.nav.steer(state)

# --- PHYSICS & RENDERING ---
class Space:
    def __init__(self, width=1000, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.physics = pymunk.Space()
        self.dt = 1/60.0
        self.dynamic_objects = []
        self.static_objects = []
        self.camera_offset = [0, 0]
        self.running = True

    def run(self, loop_fn):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            
            loop_fn(self)
            
            for obj in self.dynamic_objects: obj._update_forces(self.dt)
            self.physics.step(self.dt)
            
            # Camera Follow
            if self.dynamic_objects:
                pos = self.dynamic_objects[0].body.position
                self.camera_offset[0] += ((self.screen.get_width()/2 - pos.x) - self.camera_offset[0]) * 0.1
                self.camera_offset[1] += ((self.screen.get_height()/2 - pos.y) - self.camera_offset[1]) * 0.1

            self.screen.fill((20, 20, 20))
            for obj in self.static_objects + self.dynamic_objects:
                obj.update(self.screen, self.camera_offset)
            
            pygame.display.flip()
            self.clock.tick(60)

class Mouse:
    def __init__(self, space, position, maze_lines):
        self.body = pymunk.Body(2.0, pymunk.moment_for_circle(2.0, 0, 18))
        self.body.position = position
        self.shape = pymunk.Circle(self.body, 18)
        self.shape.friction = 0.5
        space.physics.add(self.body, self.shape)
        
        self.maze_lines = maze_lines
        self.mount_angles = [0, 180, 270, 90] # E, W, S, N
        self.telemetry = []
        self.forces = []

    def _update_forces(self, dt):
        for f in self.forces[:]:
            self.body.apply_force_at_world_point(f["f"], self.body.local_to_world(f["o"]))
            f["e"] += dt
            if f["e"] >= 0.1: self.forces.remove(f)

    def turn(self, cmd):
        pwr = 200
        self.body.velocity *= 0.95
        self.body.angular_velocity *= 0.8
        if cmd == 'f': self._fire(180, 90, -pwr); self._fire(0, 270, -pwr)
        elif cmd == 'r': self._fire(180, 90, pwr/2); self._fire(0, 270, -pwr/2)
        elif cmd == 'l': self._fire(0, 270, pwr/2); self._fire(180, 90, -pwr/2)

    def _fire(self, m_angle, f_offset, pwr):
        theta = math.radians(m_angle - 90)
        f_angle = self.body.angle + theta + math.radians(f_offset)
        self.forces.append({"f": (pwr * math.cos(f_angle), pwr * math.sin(f_angle)), "o": (18 * math.cos(theta), 18 * math.sin(theta)), "e": 0})

    def update(self, screen, offset):
        p, a = self.body.position, self.body.angle
        draw_pos = (int(p.x + offset[0]), int(p.y + offset[1]))
        pygame.draw.circle(screen, (255, 255, 255), draw_pos, 18)
        pygame.draw.line(screen, (200, 0, 0), draw_pos, (draw_pos[0]+20*math.cos(a), draw_pos[1]+20*math.sin(a)), 3)
        
        readings = []
        for ang in self.mount_angles:
            h_rad = a + math.radians(ang)
            origin = (p.x + 18*math.cos(h_rad), p.y + 18*math.sin(h_rad))
            end = (origin[0] + 400*math.cos(h_rad), origin[1] + 400*math.sin(h_rad))
            ray = LineString([origin, end])
            hit_dist = 400
            for wall in self.maze_lines:
                if ray.intersects(wall):
                    inter = ray.intersection(wall)
                    if isinstance(inter, Point):
                        hit_dist = min(hit_dist, math.dist(origin, inter.coords[0]))
            readings.append((hit_dist, 1.0))
            if hit_dist < 400:
                h_pos = (origin[0] + hit_dist*math.cos(h_rad) + offset[0], origin[1] + hit_dist*math.sin(h_rad) + offset[1])
                pygame.draw.line(screen, (60, 60, 60), (origin[0]+offset[0], origin[1]+offset[1]), h_pos, 1)
        
        self.telemetry.append({'position': (p.x, p.y), 'heading': math.degrees(a)%360, 'readings': readings})

class MapView:
    def __init__(self, grid):
        self.grid = grid
        self.font = pygame.font.SysFont("Arial", 12)
    def update(self, screen, offset):
        for n in self.grid._nodes.values():
            x, y = n.pos[0] + offset[0], n.pos[1] + offset[1]
            h = self.grid.spacing / 2
            for d, color in [('n', (0,255,100)), ('e', (0,255,100)), ('s', (0,255,100)), ('w', (0,255,100))]:
                if n.walls[d] is True:
                    if d == 'n': pygame.draw.line(screen, color, (x-h, y+h), (x+h, y+h), 2)
                    if d == 'e': pygame.draw.line(screen, color, (x+h, y-h), (x+h, y+h), 2)
                    if d == 's': pygame.draw.line(screen, color, (x-h, y-h), (x+h, y-h), 2)
                    if d == 'w': pygame.draw.line(screen, color, (x-h, y-h), (x-h, y+h), 2)
            if n.value < 1000:
                txt = self.font.render(str(n.value), True, (80, 80, 80))
                screen.blit(txt, (x-5, y-5))

def main():
    sim = Space()
    # World units
    walls = [((0,0), (0,600)), ((0,600), (600,600)), ((600,600), (600,0)), ((600,0), (0,0)),
             ((100,0), (100,200)), ((200,200), (200,400)), ((400,100), (400,500)), ((100,400), (300,400))]
    
    maze_lines = []
    for p1, p2 in walls:
        seg = pymunk.Segment(sim.physics.static_body, p1, p2, 2)
        sim.physics.add(seg)
        maze_lines.append(LineString([p1, p2]))
        sim.static_objects.append(type('W', (), {'update': lambda s, sc, o, p1=p1, p2=p2: pygame.draw.line(sc, (50,50,50), (p1[0]+o[0], p1[1]+o[1]), (p2[0]+o[0], p2[1]+o[1]), 3)})())

    grid = Grid(100, {'x': 1, 'y': 1})
    player = Mouse(sim, grid.start.pos, maze_lines)
    tree = Tree(grid, player, goal_grid_pos=(5, 1))

    sim.dynamic_objects.append(player)
    sim.static_objects.append(MapView(grid))
    sim.run(lambda s: player.turn(tree.mind()))

if __name__ == "__main__":
    main()
