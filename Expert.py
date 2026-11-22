import math

import pygame

from Car import Car
from City import City


def throttle_for(target_speed: float, speed: float) -> float:
    return max(-1.0, min(1.0, (target_speed - speed) / 15))


class Expert:
    def __init__(self, car: Car, city: City, route: list[pygame.Vector2]):
        self.car = car
        self.city = city
        self.route = route

        self.cruise_speed = 80
        self.approach_speed = 35
        self.braking_distance = 75
        self.max_brake = 180
        self.stop_buffer = car.length / 2 + 4
        self.reach_radius = 40

        self.done = False
        self.target_speed = 0.0

        self._yellow_for = None
        self._yellow_go = False
        self._wait_time = 0.0
        self._cleared = None

        self.target_index = 0
        self.lookahead_distance = 65

    def get_target(self) -> pygame.Vector2:
        return self.route[self.target_index]

    def get_target_speed(self, dt: float = 0.0) -> float:
        result = self.city.get_next_intersection(self.car.position, self.car.heading)

        if self._cleared is not None and (
            result is None or result[0] is not self._cleared
        ):
            self._cleared = None

        if result is None:
            return self.cruise_speed

        intersection, approach, distance = result
        gap = distance - self.stop_buffer
        instruction = intersection.get_instruction(approach)

        if instruction != "caution":
            self._yellow_for = None

        # stop sign: full stop, wait a second, then go
        if intersection.control == "stop":
            if intersection is self._cleared or gap < -8:
                return self.cruise_speed
            if gap < 3 and self.car.speed < 2:
                self._wait_time += dt
                if self._wait_time > 1.0:
                    self._cleared = intersection
                    self._wait_time = 0.0
                    return self.cruise_speed
            else:
                self._wait_time = 0.0
            return self._stop_profile(gap)

        if instruction == "go":
            return self.cruise_speed

        if instruction == "yield":
            return self.approach_speed

        # already over the stop line: clear the intersection
        if gap < -8:
            return self.cruise_speed

        if instruction == "caution":
            # decide once per yellow: can we stop in time?
            if self._yellow_for is not intersection:
                self._yellow_for = intersection
                v = self.car.speed
                self._yellow_go = gap < v * v / (2 * self.max_brake) + 10
            if self._yellow_go:
                return self.cruise_speed

        return self._stop_profile(gap)

    def update_target(self):
        if self.car.position.distance_to(self.route[-1]) < self.reach_radius:
            self.done = True

    def get_heading_error(self):
        target = self.get_lookahead_target()
        difference = target - self.car.position

        target_angle = math.atan2(difference.y, difference.x)

        error = target_angle - self.car.heading

        return (error + math.pi) % (2 * math.pi) - math.pi

    def get_steering(self):
        error = self.get_heading_error()

        return max(-1, min(1, error / 0.6))

    def update(self, dt: float):
        next_intersection = self.city.get_next_intersection(
            self.car.position, self.car.heading
        )
        if next_intersection:
            intersection, approach, distance = next_intersection
            instruction = intersection.get_instruction(approach)

            print(instruction, distance)
        self.update_target()
        steer = self.get_steering()

        return steer

    def _point_at(self, s: float) -> pygame.Vector2:
        cum = 0.0
        for a, b in zip(self.route, self.route[1:]):
            ab = b - a
            length = ab.length()
            if length == 0:
                continue
            if s <= cum + length:
                return a + ab * ((s - cum) / length)
            cum += length
        return self.route[-1]

    def point_ahead(self, distance: float) -> pygame.Vector2:
        s, _ = self._progress()
        return self._point_at(s + distance)

    def get_lookahead_target(self) -> pygame.Vector2:
        return self.point_ahead(20 + 0.35 * self.car.speed)

    def draw(self, screen):
        pygame.draw.lines(screen, (0, 255, 255), False, self.route, 2)

        target = self.get_lookahead_target()

        pygame.draw.circle(screen, (255, 0, 255), target, 6)

    def reset(self):
        self.target_index = 0
        self.done = False
        self.target_speed = 0.0
        self._yellow_for = None
        self._yellow_go = False
        self._wait_time = 0.0
        self._cleared = None

    def _stop_profile(self, gap: float) -> float:
        return max(
            0.0, min(self.cruise_speed, gap / self.braking_distance * self.cruise_speed)
        )

    def _progress(self) -> tuple[float, float]:
        """(arc length of the car's nearest point on the route, distance to route)"""
        p = self.car.position
        best, s_at, cum = float("inf"), 0.0, 0.0

        for a, b in zip(self.route, self.route[1:]):
            ab = b - a
            length = ab.length()
            if length == 0:
                continue
            t = max(0.0, min(1.0, (p - a).dot(ab) / (length * length)))
            d = p.distance_to(a + ab * t)
            if d < best:
                best, s_at = d, cum + t * length
            cum += length

        return s_at, best

    def distance_to_route(self) -> float:
        return self._progress()[1]

    def _direction_at(self, s: float) -> pygame.Vector2:
        cum = 0.0
        direction = pygame.Vector2(1, 0)

        for a, b in zip(self.route, self.route[1:]):
            ab = b - a
            length = ab.length()
            if length == 0:
                continue
            direction = ab / length
            if s <= cum + length:
                break
            cum += length

        return direction

    def get_command(self) -> float:
        """Nav hint for the CNN: -1 left turn, 0 straight, +1 right turn.
        Based on progress along the route, on from 150px before a turn until just after it.
        """
        s, _ = self._progress()
        before = self._direction_at(s - 25)
        ahead = self._direction_at(s + 150)

        turn = math.degrees(
            math.atan2(before.cross(ahead), before.dot(ahead))
        )  # y-down: + is clockwise

        if turn > 30:
            return 1.0
        if turn < -30:
            return -1.0
        return 0.0

    def act(self, dt: float) -> tuple[float, float]:
        """Returns (steer, throttle). Sets self.target_speed."""
        self.update_target()

        steer = self.get_steering()

        target = self.get_target_speed(dt)
        target = min(target, self.curve_speed_limit())
        target = min(target, self.cruise_speed * (1 - 0.5 * abs(steer)))
        if self.done:
            target = 0.0

        self.target_speed = target
        return steer, throttle_for(target, self.car.speed)

    def curve_speed_limit(self) -> float:
        """Slow to 40 when the route bends more than ~25 degrees in the next 70px."""
        s, _ = self._progress()
        a, b = self._direction_at(s), self._direction_at(s + 70)
        bend = abs(math.degrees(math.atan2(a.cross(b), a.dot(b))))
        return 40.0 if bend > 25 else self.cruise_speed
