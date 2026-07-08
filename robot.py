import math
from typing import NamedTuple, Callable, List, Dict, Tuple, Union # Keep Union for type hints if needed

class RobotState:
    LOOK = "LOOK"
    MOVE = "MOVE"
    WAIT = "WAIT"
    TERMINATED = "TERMINATED"
    CRASH = "CRASH"

    @staticmethod
    def next_state(current_state: str) -> str: # Add type hints here too
        if current_state == RobotState.CRASH:
            return RobotState.CRASH
        elif current_state == RobotState.LOOK:
            return RobotState.MOVE
        elif current_state == RobotState.MOVE:
            return RobotState.WAIT
        elif current_state == RobotState.WAIT:
            return RobotState.LOOK
        return current_state # Default fallback

class SchedulerType:
    ASYNC = "Async"

class DistributionType:
    EXPONENTIAL = "Exponential"

class Algorithm:
    GATHERING = "Gathering"
    SEC = "SEC"
    TWOTASK = "TwoTask"
    GO_TO_CENTER = "GoToCenter"
    CIRCLE = "CircleFormation"
    SPREADING = "Spreading"
    PATTERN = "PatternFormation"

class FaultType:
    NONE = "none"
    CRASH = "crash"            # stops permanently
    BYZANTINE = "byzantine"    # moves arbitrarily/adversarially, never settles
    OMISSION = "omission"      # intermittently fails to move when activated
    DELAY = "delay"            # sluggish: moves at reduced speed
    ALL = ("crash", "byzantine", "omission", "delay")


Time = float
Id = int

class Coordinates(NamedTuple):
    x: float
    y: float
    def __str__(self):
        return f"({float(self.x):.4f}, {float(self.y):.4f})"

class Circle(NamedTuple):
    center: Coordinates
    radius: float
    def __str__(self):
        return f"Center: {self.center} ; radius: {float(self.radius):.4f}"

class SnapshotDetails(NamedTuple):
    pos: Coordinates
    state: str
    frozen: bool
    terminated: bool
    multiplicity: Union[int, None] # Use Union for type hint clarity
    light: Union[str, None] = None  # luminous-robot light colour (observable)

class Event(NamedTuple):
    time: Time # Use type alias
    id: Id    # Use type alias
    state: str

class Orientation(NamedTuple): # Keep definition even if unused for now
    translation: float
    rotation: float
    reflection: float

# --- Robot Class Definition (Mostly unchanged logic) ---

# Simple logging replacement
class SimpleLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    def warning(self, msg):
        print(f"WARN: {msg}")
    def error(self, msg):
        print(f"ERROR: {msg}")

class Robot:
    _logger = SimpleLogger()
    _generator = None # Will be set by scheduler

    def __init__(
        self,
        id: Id, # Use type alias
        coordinates: Coordinates,
        algorithm: str, # e.g., Algorithm.GATHERING
        speed: float = 1.0,
        color: Union[str, None] = None, # Use Union
        visibility_radius: Union[float, None] = None, # Use Union
        # orientation: Union[Orientation, None] = None, # Keep if needed later
        # obstructed_visibility: bool = False, # Keep if needed later
        multiplicity_detection: bool = False,
        rigid_movement: bool = False,
        threshold_precision: float = 5,
        width_bound: Union[float, None] = None,
        height_bound: Union[float, None] = None,
        fault_type: str = FaultType.NONE,
    ):
        self.speed = speed
        self.color = color
        # Ensure visibility_radius is float or inf
        self.visibility_radius = float(visibility_radius) if visibility_radius is not None else float('inf')
        # self.obstructed_visibility = obstructed_visibility
        self.multiplicity_detection = multiplicity_detection
        self.rigid_movement = rigid_movement
        self.width_bound = float(width_bound) if width_bound else None
        self.height_bound = float(height_bound) if height_bound else None
        # self.orientation = orientation
        self.start_time: Union[Time, None] = None # Use type alias and Union
        self.end_time: Union[Time, None] = None   # Use type alias and Union
        self.state: str = RobotState.WAIT # Use constant from this file
        self.start_position: Coordinates = coordinates
        self.calculated_position: Union[Coordinates, None] = None
        self.number_of_activations: int = 0
        self.travelled_distance: float = 0.0
        self.snapshot: Union[Dict[Id, SnapshotDetails], None] = None # Use Dict, Id
        self.coordinates: Coordinates = coordinates
        self.id: Id = id # Use type alias
        self.threshold_precision: float = threshold_precision
        self.frozen: bool = False
        self.terminated: bool = False
        self.sec: Union[Circle, None] = None # Stores the calculated SEC
        self.fault_type: str = fault_type
        self.current_light: Union[str, None] = None     # luminous-robot light
        self.last_light_event_time: float = -1.0

        # Assign algorithm type; validate eagerly against the selection table.
        self.algorithm_type = algorithm
        self._select_algorithm()  # raises ValueError for an unknown algorithm


    def set_fault(self, fault_type: str) -> None:
        self.fault_type = fault_type
        if fault_type == FaultType.CRASH:
            self.state = RobotState.CRASH
        elif fault_type == FaultType.DELAY:
            self.speed *= 0.4                          # sluggish actuation

    def set_faulty(self, faulty: bool) -> None:        # backwards-compatible alias
        if faulty:
            self.set_fault(FaultType.CRASH)

    def _byzantine_reach(self) -> float:
        # how far a Byzantine robot wanders: ~half the spread of what it sees
        ds = [math.dist(self.coordinates, v.pos)
              for k, v in (self.snapshot or {}).items() if k != self.id]
        return max(ds) * 0.5 if ds else 10.0

    def set_light(self, new_color: str, time: float) -> None:
        # Luminous-robot light: an externally visible colour other robots observe.
        # The 0.5 time-unit cooldown models a minimum interval between light events.
        if (new_color != self.current_light
                and time - self.last_light_event_time >= 0.5):
            self.current_light = new_color
            self.last_light_event_time = time
            Robot._logger.info(f"[{time:.2f}] {{R{self.id}}} LIGHT -> {new_color}")

    def look(
        self,
        snapshot: Dict[Id, SnapshotDetails], # Use Dict, Id
        time: Time,                         # Use Time
    ) -> None:
        if self.state == RobotState.CRASH: return

        self.state = RobotState.LOOK
        self.set_light("blue", time)        # LOOK

        self.snapshot = {}
        for key, value in snapshot.items():
            if self.visibility_radius == float('inf') or self._robot_is_visible(value.pos):
                transformed_pos = self._convert_coordinate(value.pos)
                # Make sure SnapshotDetails is used correctly
                self.snapshot[key] = SnapshotDetails(
                    transformed_pos,
                    value.state,
                    value.frozen,
                    value.terminated,
                    value.multiplicity,
                    value.light,
                )

        Robot._logger.info(
            f"[{time:.2f}] {{R{self.id}}} LOOK    -- Snapshot {self.prettify_snapshot(self.snapshot)}"
        )

        if self.fault_type == FaultType.BYZANTINE:
            # Adversarial: wander to an erratic nearby point and never settle, so the
            # correct robots (which observe its true, misleading position) are disrupted.
            reach = self._byzantine_reach()
            ang = float(Robot._generator.uniform(0, 2 * math.pi)) if Robot._generator else 0.0
            self.calculated_position = Coordinates(
                self.coordinates.x + reach * math.cos(ang),
                self.coordinates.y + reach * math.sin(ang))
            self.frozen = False
            self.terminated = False
            Robot._logger.info(f"[{time:.2f}] {{R{self.id}}} BYZANTINE -- erratic move")
            return

        active_visible_robots = [r for r_id, r in self.snapshot.items() if not r.terminated and r.state != RobotState.CRASH]

        if len(active_visible_robots) <= 1 and self.id in self.snapshot:
            self.frozen = True
            self.terminated = True
            Robot._logger.info(f"[{time:.2f}] {{R{self.id}}} TERMINATED (only self visible)")
            self.wait(time)
            return

        algo, algo_terminal = self._select_algorithm()
        # Type hint for algo: Callable[[], Tuple[Coordinates, List[any]]]
        # Type hint for algo_terminal: Callable[[Coordinates, List[any]], bool]
        self.calculated_position = self._compute(algo, algo_terminal, time)
        pos_str = (
            f"({self.calculated_position.x:.4f}, {self.calculated_position.y:.4f})" # Access .x, .y
            if self.calculated_position
            else "None"
        )
        Robot._logger.info(
            f"[{time:.2f}] {{R{self.id}}} COMPUTE -- Computed Pos: {pos_str}"
        )

        if self.terminated:
             Robot._logger.info(f"[{time:.2f}] {{R{self.id}}} TERMINATED (condition met in compute)")
             self.wait(time)
             return

        omit = (self.fault_type == FaultType.OMISSION and Robot._generator is not None
                and float(Robot._generator.random()) < 0.5)
        if omit or self.calculated_position is None or \
           math.dist(self.calculated_position, self.coordinates) < 10**-self.threshold_precision:
            self.frozen = True
            reason = "OMISSION (skipped move)" if omit else "FROZEN (target reached or no movement)"
            Robot._logger.info(f"[{time:.2f}] {{R{self.id}}} {reason}")
            self.wait(time)
        else:
            self.frozen = False

    def _compute(
        self,
        algo: Callable[[], Tuple[Coordinates, List[any]]],
        check_terminal: Callable[[Coordinates, List[any]], bool],
        time: Time
    ) -> Union[Coordinates, None]:
        try:
            coord, extra_args = algo()

            if check_terminal is None:
                Robot._logger.error("Algorithm termination function not passed in")
                return self.coordinates

            if check_terminal(coord, extra_args):
                Robot._logger.info(f"[{time:.2f}] {{R{self.id}}} Termination condition met during compute.")
                self.terminated = True
                return coord # Return calculated coord, but flag is set

            else:
                 return coord

        except Exception as e:
            Robot._logger.error(f"[{time:.2f}] {{R{self.id}}} Error during _compute: {e}")
            self.frozen = True
            return self.coordinates


    def move(self, start_time: Time) -> None:
        if self.state == RobotState.CRASH or self.terminated or self.frozen:
             Robot._logger.info(f"[{start_time:.2f}] {{R{self.id}}} Skipping MOVE (State: {self.state}, Term: {self.terminated}, Frozen: {self.frozen})")
             self.state = RobotState.WAIT # Ensure it goes back to WAIT if frozen/terminated/crashed tried to move
             return

        if self.calculated_position is None:
             Robot._logger.warning(f"[{start_time:.2f}] {{R{self.id}}} MOVE called with no calculated_position. Skipping move.")
             self.state = RobotState.WAIT
             return

        self.state = RobotState.MOVE
        self.set_light("red", start_time)   # MOVE
        Robot._logger.info(f"[{start_time:.2f}] {{R{self.id}}} MOVE -> {self.calculated_position}")
        self.start_time = start_time
        self.start_position = self.coordinates

    def wait(self, time: Time) -> None:
        # When finishing a rigid MOVE, snap straight to the computed target.
        # The WAIT event is scheduled at the (rigid) arrival time, so the robot is
        # meant to be there. Re-deriving position from elapsed time is fragile: a
        # tiny move can make the WAIT event share the MOVE's timestamp (float
        # underflow / rounding) -> elapsed_time == 0 strands the robot at its start,
        # which then re-LOOKs forever (the endless-simulation bug).
        if (self.state == RobotState.MOVE and self.start_time is not None
                and self.calculated_position is not None
                and (self.rigid_movement or time <= self.start_time + 1e-12)):
            final_pos = self.calculated_position
        else:
            final_pos = self.get_position(time)
        current_distance = 0.0
        if self.start_time is not None and self.state == RobotState.MOVE:
             # Make sure start_position is Coordinates type
             current_distance = math.dist(self.start_position, final_pos)
             self.travelled_distance += current_distance

        self.coordinates = final_pos

        self.start_time = None
        self.end_time = time
        self.state = RobotState.WAIT
        self.set_light("green", time)       # WAIT

        Robot._logger.info(
            f"[{time:.2f}] {{R{self.id}}} WAIT    -- Pos: {self.coordinates} Dist: {current_distance:.4f} Total: {self.travelled_distance:.4f} Frozen: {self.frozen} Term: {self.terminated}"
        )


    def get_position(self, time: Time) -> Coordinates:
        if self.state != RobotState.MOVE or self.start_time is None or self.calculated_position is None:
            return self.coordinates

        target_distance = math.dist(self.start_position, self.calculated_position)

        if target_distance < 1e-9:
            return self.calculated_position

        elapsed_time = time - self.start_time
        distance_covered = self.speed * elapsed_time

        if distance_covered >= target_distance - (10**-self.threshold_precision):
            return self.calculated_position
        else:
            factor = distance_covered / target_distance
            interpolated_coords = self._interpolate(
                self.start_position, self.calculated_position, factor
            )
            return interpolated_coords


    def _select_algorithm(self) -> Tuple[Callable, Callable]:
        table = {
            Algorithm.GATHERING:    (self._midpoint, self._midpoint_terminal),
            Algorithm.SEC:          (self._smallest_enclosing_circle, self._sec_terminal),
            Algorithm.GO_TO_CENTER: (self._go_to_center, self._gtc_terminal),
            Algorithm.CIRCLE:       (self._circle_formation, self._circle_terminal),
            Algorithm.SPREADING:    (self._spreading, self._spreading_terminal),
            Algorithm.PATTERN:      (self._pattern_formation, self._pattern_terminal),
        }
        if self.algorithm_type not in table:
            raise ValueError(f"Invalid algorithm type: {self.algorithm_type}")
        return table[self.algorithm_type]

    def _interpolate(
        self, start: Coordinates, end: Coordinates, t: float
    ) -> Coordinates:
        t = max(0.0, min(1.0, t))
        # Ensure Coordinates constructor is used
        return Coordinates(
            start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)
        )

    def _convert_coordinate(self, coord: Coordinates) -> Coordinates:
        return coord

    def _robot_is_visible(self, coord: Coordinates) -> bool:
        if self.visibility_radius == float('inf'):
             return True
        distance = math.dist(self.coordinates, coord)
        return distance <= self.visibility_radius


    def _midpoint(self) -> Tuple[Coordinates, List[any]]:
        if not self.snapshot:
             Robot._logger.warning(f"{{R{self.id}}} Midpoint calculation with empty snapshot. Staying put.")
             return (self.coordinates, [])

        x = y = 0
        num_visible = 0
        for _, value in self.snapshot.items():
            x += value.pos.x # Access pos.x
            y += value.pos.y # Access pos.y
            num_visible += 1

        if num_visible == 0:
             return (self.coordinates, [])

        x /= num_visible
        y /= num_visible

        return (Coordinates(x, y), []) # Construct Coordinates


    def _midpoint_terminal(self, coord: Coordinates, args: List[any] = None) -> bool:
        if not self.snapshot:
             return True

        for robot_id, details in self.snapshot.items():
            if details.state != RobotState.CRASH:
                 # Access details.pos
                if math.dist(details.pos, coord) > math.pow(10, -self.threshold_precision):
                    return False
        return True

    # --- SEC Algorithm Methods ---
    # Ensure Coordinates, Circle, SnapshotDetails, etc. are used correctly from this file's definitions
    # Make sure .x, .y are accessed correctly on Coordinates objects

    def _smallest_enclosing_circle(self) -> Tuple[Coordinates, List[Union[Circle, None]]]:
        visible_robots_details = {rid: r for rid, r in self.snapshot.items() if r.state != RobotState.CRASH}
        points_coords: List[Coordinates] = [r.pos for r in visible_robots_details.values()] # List of Coordinates

        num_robots = len(points_coords)
        destination: Union[Coordinates, None] = None
        calculated_sec: Union[Circle, None] = None

        try:
            if num_robots == 0:
                destination = self.coordinates
                calculated_sec = None
            elif num_robots == 1:
                destination = points_coords[0]
                calculated_sec = Circle(points_coords[0], 0) # Use Circle constructor
            elif num_robots == 2:
                a, b = points_coords[0], points_coords[1]
                calculated_sec = self._circle_from_two(a, b)
                destination = self._closest_point_on_circle(calculated_sec, self.coordinates)
            elif num_robots == 3:
                potential_sec = None
                for i in range(num_robots):
                    for j in range(i + 1, num_robots):
                        a, b = points_coords[i], points_coords[j]
                        sec_candidate = self._circle_from_two(a, b)
                        if self._valid_circle(sec_candidate, points_coords):
                            potential_sec = sec_candidate
                            break
                    if potential_sec: break

                if not potential_sec:
                     a, b, c = points_coords[0], points_coords[1], points_coords[2]
                     if abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) < 1e-9:
                         max_dist_sq = -1
                         p1, p2 = a, b
                         pairs = [(a, b), (a, c), (b, c)]
                         for p_i, p_j in pairs:
                             d_sq = (p_i.x - p_j.x)**2 + (p_i.y - p_j.y)**2
                             if d_sq > max_dist_sq:
                                 max_dist_sq = d_sq
                                 p1, p2 = p_i, p_j
                         potential_sec = self._circle_from_two(p1, p2)
                     else:
                         potential_sec = self._circle_from_three(a, b, c)

                calculated_sec = potential_sec
                if calculated_sec:
                     destination = self._closest_point_on_circle(calculated_sec, self.coordinates)
                else:
                     Robot._logger.warning(f"[{self.id}] Failed to calculate SEC for 3 points. Staying put.")
                     destination = self.coordinates
            else: # > 3 robots
                calculated_sec = self._sec_welzl_coords(points_coords)
                if calculated_sec:
                    destination = self._closest_point_on_circle(calculated_sec, self.coordinates)
                else:
                     Robot._logger.warning(f"[{self.id}] Failed to calculate SEC via Welzl. Staying put.")
                     destination = self.coordinates

            self.sec = calculated_sec
            # Ensure destination is always Coordinates
            final_destination = destination if destination is not None else self.coordinates
            return (final_destination, [self.sec])

        except Exception as e:
            Robot._logger.error(f"[{self.id}] Error in _smallest_enclosing_circle: {e}")
            return (self.coordinates, [None])


    def _sec_terminal(self, _, args: List[Union[Circle, None]]) -> bool:
        if not args or args[0] is None:
            return False

        circle: Circle = args[0] # Now explicitly Circle type

        visible_robots_details = {rid: r for rid, r in self.snapshot.items() if r.state != RobotState.CRASH}

        if not visible_robots_details:
             return True

        for robot_id, details in visible_robots_details.items():
            # Pass details.pos (Coordinates) to is_point_on_circle
            if not self._is_point_on_circle(details.pos, circle):
                 return False
        return True


    def _sec_welzl_coords(self, points: List[Coordinates]) -> Union[Circle, None]:
        if not points: return None
        points_copy = points.copy()
        if Robot._generator is None:
             Robot._logger.error("Robot._generator not set for Welzl shuffle!")
             import random
             random.shuffle(points_copy)
        else:
            Robot._generator.shuffle(points_copy)
        return self._sec_welzl_recur_coords(points_copy, [], len(points_copy))


    def _sec_welzl_recur_coords(self, P: List[Coordinates], R: List[Coordinates], n: int) -> Circle:
        if n == 0 or len(R) == 3:
            return self._min_circle(R)

        idx = Robot._generator.integers(0, n) if n > 0 else 0
        p = P[idx]
        P[idx], P[n - 1] = P[n - 1], P[idx]

        c = self._sec_welzl_recur_coords(P, R.copy(), n - 1)

        # Use math.dist with Coordinates objects
        if c is not None and c.radius >= 0 and \
           round(math.dist(c.center, p), self.threshold_precision) <= round(c.radius, self.threshold_precision):
             return c
        else:
            R.append(p)
            return self._sec_welzl_recur_coords(P, R.copy(), n - 1)


    def _min_circle(self, points: List[Coordinates]) -> Circle:
        if not points:
            return Circle(Coordinates(0, 0), 0) # Use constructors
        elif len(points) == 1:
            return Circle(points[0], 0)
        elif len(points) == 2:
            return self._circle_from_two(points[0], points[1])
        elif len(points) == 3:
             for i in range(3):
                 p1, p2 = points[i], points[(i + 1) % 3]
                 c = self._circle_from_two(p1, p2)
                 p3 = points[(i + 2) % 3]
                 # Use math.dist with Coordinates
                 if round(math.dist(c.center, p3), self.threshold_precision) <= round(c.radius, self.threshold_precision):
                      return c

             a, b, c_pts = points[0], points[1], points[2] # Rename c to avoid conflict
             # Access .x, .y for collinearity check
             if abs(a.x * (b.y - c_pts.y) + b.x * (c_pts.y - a.y) + c_pts.x * (a.y - b.y)) < 1e-9:
                 max_dist_sq = -1
                 p1_max, p2_max = a, b
                 pairs = [(a, b), (a, c_pts), (b, c_pts)]
                 for p_i, p_j in pairs:
                     # Access .x, .y for distance check
                     d_sq = (p_i.x - p_j.x)**2 + (p_i.y - p_j.y)**2
                     if d_sq > max_dist_sq:
                         max_dist_sq = d_sq
                         p1_max, p2_max = p_i, p_j
                 return self._circle_from_two(p1_max, p2_max)
             else:
                 return self._circle_from_three(a, b, c_pts)
        else:
             Robot._logger.error("Min_circle called with > 3 points")
             return Circle(Coordinates(0,0), -1)


    def _is_point_on_circle(self, p: Coordinates, c: Circle) -> bool:
        if c is None or c.radius < 0: return False
        # Use math.dist with Coordinates objects
        distance = math.dist(p, c.center)
        return abs(distance - c.radius) < math.pow(10, -self.threshold_precision)


    def _closest_point_on_circle(self, circle: Circle, point: Coordinates) -> Coordinates:
        if circle is None or circle.radius < 0: return point

        center: Coordinates = circle.center
        radius: float = circle.radius

        # Use math.dist with Coordinates
        if math.dist(center, point) < 1e-9:
             return Coordinates(center.x + radius, center.y) # Use constructor

        # Access .x, .y
        vx, vy = point.x - center.x, point.y - center.y
        d = math.sqrt(vx**2 + vy**2)
        scale = radius / d
        cx = center.x + vx * scale
        cy = center.y + vy * scale
        return Coordinates(cx, cy) # Use constructor


    def _valid_circle(self, circle: Circle, points: List[Coordinates]) -> bool:
        if circle is None or circle.radius < 0: return False
        for p in points:
            # Use math.dist with Coordinates
            if round(math.dist(circle.center, p), self.threshold_precision) > round(circle.radius, self.threshold_precision):
                return False
        return True


    def _circle_from_two(self, a: Coordinates, b: Coordinates) -> Circle:
        # Access .x, .y
        center_x = (a.x + b.x) / 2.0
        center_y = (a.y + b.y) / 2.0
        center = Coordinates(center_x, center_y) # Use constructor
        # Use math.dist with Coordinates
        radius = math.dist(a, b) / 2.0
        return Circle(center, radius) # Use constructor


    def _circle_from_three(self, a: Coordinates, b: Coordinates, c: Coordinates) -> Circle:
        # Access .x, .y
        A = b.x - a.x; B = b.y - a.y
        C = c.x - a.x; D = c.y - a.y
        E = A * (a.x + b.x) + B * (a.y + b.y)
        F = C * (a.x + c.x) + D * (a.y + c.y)
        G = 2 * (A * (c.y - b.y) - B * (c.x - b.x))

        if abs(G) < 1e-9:
             Robot._logger.warning(f"[{self.id}] _circle_from_three called with collinear points: {a}, {b}, {c}. Using diameter fallback.")
             max_dist_sq = -1
             p1_max, p2_max = a, b
             pairs = [(a, b), (a, c), (b, c)]
             for p_i, p_j in pairs:
                 d_sq = (p_i.x - p_j.x)**2 + (p_i.y - p_j.y)**2
                 if d_sq > max_dist_sq:
                     max_dist_sq = d_sq
                     p1_max, p2_max = p_i, p_j
             return self._circle_from_two(p1_max, p2_max)

        center_x = (D * E - B * F) / G
        center_y = (A * F - C * E) / G
        center = Coordinates(center_x, center_y) # Use constructor
        # Use math.dist with Coordinates
        radius = math.dist(center, a)
        return Circle(center, radius) # Use constructor

    # --- End SEC ---

    # --- Go-To-Center (Ando, Suzuki & Yamashita, IEEE T-RA 1999) ---
    # Limited-visibility point convergence: move toward the centre of the smallest
    # enclosing circle of the visible robots, but cap the step so every robot now
    # within visibility range V stays within V (connectivity is never broken). The
    # per-neighbour cap keeps the robot inside the disk of radius V/2 around the
    # midpoint of the pair, which guarantees the visibility edge survives. With
    # unlimited visibility this reduces to "go to the SEC centre".
    def _go_to_center(self) -> Tuple[Coordinates, List[Union[Circle, None]]]:
        pts = [r.pos for r in self.snapshot.values() if r.state != RobotState.CRASH]
        if len(pts) <= 1:
            return (self.coordinates, [None])

        sec = self._sec_welzl_coords(pts)
        self.sec = sec
        if sec is None or sec.radius < 0:
            return (self.coordinates, [None])

        me = self.coordinates
        gx, gy = sec.center.x - me.x, sec.center.y - me.y
        goal_dist = math.hypot(gx, gy)
        if goal_dist < math.pow(10, -self.threshold_precision):
            return (self.coordinates, [sec])            # already at the centre

        ux, uy = gx / goal_dist, gy / goal_dist         # unit vector toward centre
        step = goal_dist

        V = self.visibility_radius
        if V != float('inf'):
            R = V / 2.0
            for r in self.snapshot.values():
                if r.state == RobotState.CRASH:
                    continue
                jx, jy = r.pos.x - me.x, r.pos.y - me.y
                d = math.hypot(jx, jy)
                if d <= 1e-12 or d > V:                  # self, or not actually visible
                    continue
                # theta = angle between direction-to-centre and direction-to-j
                cos_t = max(-1.0, min(1.0, (ux * jx + uy * jy) / d))
                sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                r0 = d / 2.0
                # farthest point of the ray that stays inside B(midpoint_ij, V/2)
                limit_j = r0 * cos_t + math.sqrt(max(0.0, R * R - (r0 * sin_t) ** 2))
                step = min(step, limit_j)

        step = max(0.0, min(goal_dist, step))
        target = Coordinates(me.x + ux * step, me.y + uy * step)
        return (target, [sec])

    def _gtc_terminal(self, _, args: List[Union[Circle, None]]) -> bool:
        # Converged once the visible robots have collapsed to (essentially) a point.
        if not args or args[0] is None:
            return True
        sec: Circle = args[0]
        return sec.radius < math.pow(10, -self.threshold_precision)
    # --- End GTC ---

    # --- Uniform Circle Formation (Defago & Konagaya 2002; Flocchini et al.) ---
    # Robots arrange at equal angular spacing on a common circle = centroid +
    # mean-radius of the swarm. (A smooth, stable reference: using the live
    # smallest-enclosing circle makes the centre hop between its defining pairs
    # and the robots chase it forever.)
    #   Phase 1: a robot off the circle projects radially onto it.
    #   Phase 2: a robot on the circle slides tangentially to the angular bisector
    #            of its two neighbours (local gap-averaging) -> equal spacing.
    def _circle_formation(self) -> Tuple[Coordinates, List[Union[Circle, None]]]:
        pts = [r.pos for r in self.snapshot.values() if r.state != RobotState.CRASH]
        n = len(pts)
        if n <= 1:
            return (self.coordinates, [None])
        ox = sum(p.x for p in pts) / n
        oy = sum(p.y for p in pts) / n
        rad = sum(math.hypot(p.x - ox, p.y - oy) for p in pts) / n   # mean distance
        thr = math.pow(10, -self.threshold_precision)
        if rad <= thr:
            return (self.coordinates, [None])
        O = Coordinates(ox, oy)
        sec = Circle(O, rad)                             # agreed target circle (+ viz)
        self.sec = sec
        me = self.coordinates
        on_tol = rad * 1e-2                              # relative "on the circle" tol
        # Phase 1: snap onto the circle if not already on it.
        if abs(math.dist(me, O) - rad) > on_tol:
            return (self._closest_point_on_circle(sec, me), [sec])

        # Phase 2: tangential move to the bisector of the two angular neighbours.
        TWO_PI = 2.0 * math.pi
        my_ang = math.atan2(me.y - O.y, me.x - O.x)
        deltas = [(math.atan2(p.y - O.y, p.x - O.x) - my_ang) % TWO_PI
                  for p in pts if math.dist(p, me) > 1e-9]
        if not deltas:
            return (self.coordinates, [sec])
        ccw = min(deltas)                               # nearest neighbour CCW
        cw = TWO_PI - max(deltas)                       # nearest neighbour CW
        target_ang = my_ang + (ccw - cw) / 2.0
        target = Coordinates(O.x + rad * math.cos(target_ang),
                             O.y + rad * math.sin(target_ang))
        return (target, [sec])

    def _circle_terminal(self, _, args: List[Union[Circle, None]]) -> bool:
        sec = args[0] if args else None
        thr = math.pow(10, -self.threshold_precision)
        if sec is None or sec.radius <= thr:
            return True
        pts = [r.pos for r in self.snapshot.values() if r.state != RobotState.CRASH]
        n = len(pts)
        if n <= 1:
            return True
        O, rad = sec.center, sec.radius
        on_tol = rad * 2e-2
        TWO_PI = 2.0 * math.pi
        angs = []
        for p in pts:
            if abs(math.dist(p, O) - rad) > on_tol:
                return False                            # someone not on the circle
            angs.append(math.atan2(p.y - O.y, p.x - O.x) % TWO_PI)
        angs.sort()
        gaps = [(angs[(i + 1) % n] - angs[i]) % TWO_PI for i in range(n)]
        target_gap = TWO_PI / n
        return max(abs(g - target_gap) for g in gaps) < target_gap * 0.06
    # --- End Circle Formation ---

    # --- Spreading / uniform deployment (Lloyd's algorithm; Cortes et al. 2004) ---
    # Each robot moves to the centroid of its Voronoi cell, approximated by sampling
    # the region on a grid and assigning each sample to its nearest robot. Iterating
    # converges to a centroidal Voronoi tessellation -> uniform area coverage. The
    # region is the world box when known, else a padded bounding box of the swarm.
    _SPREAD_GRID = 32

    def _spread_region(self, positions: List[Coordinates]) -> Tuple[float, float, float, float]:
        if self.width_bound and self.height_bound:
            return (-self.width_bound / 2.0, self.width_bound / 2.0,
                    -self.height_bound / 2.0, self.height_bound / 2.0)
        xs = [p.x for p in positions]; ys = [p.y for p in positions]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pad = 0.15 * span
        return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def _spreading(self) -> Tuple[Coordinates, List[any]]:
        items = [(rid, r.pos) for rid, r in self.snapshot.items()
                 if r.state != RobotState.CRASH]
        if len(items) <= 1:
            return (self.coordinates, [])
        positions = [p for _, p in items]
        my_idx = next((k for k, (rid, _) in enumerate(items) if rid == self.id), None)
        if my_idx is None:
            return (self.coordinates, [])
        me = positions[my_idx]

        xmin, xmax, ymin, ymax = self._spread_region(positions)
        G = Robot._SPREAD_GRID
        sx = (xmax - xmin) / G; sy = (ymax - ymin) / G
        tol = 0.004 * math.hypot(xmax - xmin, ymax - ymin)   # region-relative "settled"
        ax = ay = cnt = 0
        for i in range(G):
            px = xmin + (i + 0.5) * sx
            for j in range(G):
                py = ymin + (j + 0.5) * sy
                best = 0; bestd = float('inf')
                for k, p in enumerate(positions):       # nearest robot to this sample
                    dd = (p.x - px) ** 2 + (p.y - py) ** 2
                    if dd < bestd:
                        bestd = dd; best = k
                if best == my_idx:                      # sample belongs to my cell
                    ax += px; ay += py; cnt += 1

        if cnt > 0:
            return (Coordinates(ax / cnt, ay / cnt), [tol])

        # Starved cell (coincident with / dominated by another robot): step away
        # from the nearest neighbour so the pair separates and gets distinct cells.
        nearest = min((p for k, p in enumerate(positions) if k != my_idx),
                      key=lambda p: (p.x - me.x) ** 2 + (p.y - me.y) ** 2, default=None)
        if nearest is None:
            return (self.coordinates, [])
        dx, dy = me.x - nearest.x, me.y - nearest.y
        d = math.hypot(dx, dy)
        if d < 1e-9:                                    # exactly coincident: id-based nudge
            a = (self.id % 12) * (math.pi / 6.0)
            dx, dy, d = math.cos(a), math.sin(a), 1.0
        step = max(sx, sy)
        return (Coordinates(me.x + dx / d * step, me.y + dy / d * step), [tol])

    def _spreading_terminal(self, coord: Coordinates, args: List[any]) -> bool:
        # Settled once the robot already sits at its Voronoi-cell centroid.
        if not args:
            return False
        return math.dist(coord, self.coordinates) < args[0]
    # --- End Spreading ---

    # --- Pattern Formation (Suzuki & Yamashita, SIAM J. Comput. 1999) ---
    # Robots form a target geometric pattern. Each robot deterministically derives a
    # common reference frame from the snapshot -- origin = centroid, scale = distance
    # of the farthest robot, orientation = world axes (this sim shares a global frame,
    # which sidesteps the orientation-symmetry obstruction) -- embeds the (normalised)
    # target pattern in it, matches robots to target points by a consistent angular
    # ordering, and moves to its matched point. The frame is centroid-based so it does
    # not collapse the way an SEC-based frame can.
    def _pattern_unit_points(self, n: int) -> List[List[float]]:
        # n points spread evenly along a 5-pointed star outline, centred at the
        # centroid and scaled so the farthest point sits at distance 1.
        spikes, inner = 5, 0.382
        verts = []
        for i in range(2 * spikes):
            ang = math.pi / 2 + i * math.pi / spikes
            rad = 1.0 if i % 2 == 0 else inner
            verts.append((rad * math.cos(ang), rad * math.sin(ang)))
        m = len(verts)
        segs = []; total = 0.0
        for i in range(m):
            a, b = verts[i], verts[(i + 1) % m]
            L = math.hypot(b[0] - a[0], b[1] - a[1]); segs.append(L); total += L
        pts = []
        for k in range(n):
            d = (k / n) * total; acc = 0.0
            for i, L in enumerate(segs):
                if acc + L >= d or i == m - 1:
                    t = (d - acc) / L if L > 1e-12 else 0.0
                    a, b = verts[i], verts[(i + 1) % m]
                    pts.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
                    break
                acc += L
        cx = sum(p[0] for p in pts) / n; cy = sum(p[1] for p in pts) / n
        pts = [[p[0] - cx, p[1] - cy] for p in pts]
        rms = math.sqrt(sum(p[0] ** 2 + p[1] ** 2 for p in pts) / n) or 1.0
        return [[p[0] / rms, p[1] / rms] for p in pts]

    def _pattern_embed(self) -> Union[Tuple, None]:
        items = [(rid, r.pos) for rid, r in self.snapshot.items()
                 if r.state != RobotState.CRASH]
        n = len(items)
        if n <= 1:
            return None
        positions = [p for _, p in items]
        my_idx = next((k for k, (rid, _) in enumerate(items) if rid == self.id), None)
        if my_idx is None:
            return None
        ox = sum(p.x for p in positions) / n
        oy = sum(p.y for p in positions) / n
        R = math.sqrt(sum((p.x - ox) ** 2 + (p.y - oy) ** 2 for p in positions) / n)
        if R < math.pow(10, -self.threshold_precision):
            return None
        unit = self._pattern_unit_points(n)
        targets = [Coordinates(ox + R * u[0], oy + R * u[1]) for u in unit]

        def ang_rad(p):
            return (math.atan2(p.y - oy, p.x - ox), math.hypot(p.x - ox, p.y - oy))
        robot_order = sorted(range(n), key=lambda k: ang_rad(positions[k]) + (items[k][0],))
        target_order = sorted(range(n), key=lambda k: ang_rad(targets[k]))
        rp = [positions[k] for k in robot_order]
        tp = [targets[k] for k in target_order]
        # Pick the cyclic rotation that best aligns robots to targets. This removes
        # the rotational ambiguity of two angle-sorted rings that would otherwise let
        # the matching flip between rounds and oscillate forever. Tie-break to the
        # smallest offset for determinism.
        best_s, best_cost = 0, None
        for s in range(n):
            cost = sum((rp[i].x - tp[(i + s) % n].x) ** 2 +
                       (rp[i].y - tp[(i + s) % n].y) ** 2 for i in range(n))
            if best_cost is None or cost < best_cost - 1e-9:
                best_cost, best_s = cost, s
        my_rank = robot_order.index(my_idx)
        my_target = tp[(my_rank + best_s) % n]
        err = max(math.dist(rp[i], tp[(i + best_s) % n]) for i in range(n))
        return my_target, R, err

    def _pattern_formation(self) -> Tuple[Coordinates, List[any]]:
        emb = self._pattern_embed()
        if emb is None:
            return (self.coordinates, [None])
        my_target, R, err = emb
        return (my_target, [R, err])

    def _pattern_terminal(self, _, args: List[any]) -> bool:
        if not args or args[0] is None:
            return True
        R, err = args[0], args[1]
        return err < max(R * 0.03, math.pow(10, -self.threshold_precision))
    # --- End Pattern Formation ---

    def prettify_snapshot(self, snapshot: Dict[Id, SnapshotDetails]) -> str:
        if not snapshot: return " <empty>"
        result = ""
        sorted_ids = sorted(snapshot.keys())
        for key in sorted_ids:
            value = snapshot[key] # value is SnapshotDetails
            frozen = "*" if value.frozen else ""
            terminated = "#" if value.terminated else ""
            crashed = "!" if value.state == RobotState.CRASH else ""
            multi = f"({value.multiplicity})" if self.multiplicity_detection and value.multiplicity and value.multiplicity > 1 else ""
            state_str = value.state
            # value.pos is Coordinates
            result += f"\n\t{key}{frozen}{terminated}{crashed}{multi}: {state_str} @ {value.pos}"
        return result


    def __str__(self):
         state_str = self.state
         term_str = "#" if self.terminated else ""
         frozen_str = "*" if self.frozen else ""
         crash_str = "!" if state_str == RobotState.CRASH else ""
         # self.coordinates is Coordinates
         return f"R{self.id}{term_str}{frozen_str}{crash_str} @ {self.coordinates}, St: {state_str}, Spd: {self.speed:.2f}, VRad: {self.visibility_radius}"

print("robot.py including types/enums loaded.")