import math

import pygame
from Car import Car
from City import City


class Expert:
    def __init__(self, car: Car, city: City, route: list[pygame.Vector2]):
        self.car = car
        self.city = city
        self.route = route

        self.cruise_speed = 80
        self.approach_speed = 35
        self.braking_distance = 120
        self.stop_buffer = 5

        self.target_index = 0
        self.lookahead_distance = 65

    def get_target(self) -> pygame.Vector2:
        return self.route[self.target_index]

    def get_target_speed(self) -> float:
        result = self.city.get_next_intersection(self.car.position, self.car.heading)

        if result is None:
            return self.cruise_speed

        intersection, approach, distance = result
        instruction = intersection.get_instruction(approach)

        if instruction == "go":
            return self.cruise_speed

        if instruction == "caution":
            return self.approach_speed

        if distance <= self.stop_buffer:
            return 0

        return min(
            self.cruise_speed,
            max(0, distance / self.braking_distance * self.cruise_speed),
        )

    def update_target(self):
        target = self.get_target()

        if self.car.position.distance_to(target) < 40:
            self.target_index = min(self.target_index + 1, len(self.route) - 1)

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

    def get_lookahead_target(self):
        remaining = self.lookahead_distance
        start = self.car.position

        for i in range(self.target_index, len(self.route)):
            end = self.route[i]
            segment = end - start
            length = segment.length()

            if length >= remaining and length > 0:
                return start + segment.normalize() * remaining

            remaining -= length
            start = end

        return self.route[-1]

    def draw(self, screen):
        pygame.draw.lines(screen, (0, 255, 255), False, self.route, 2)

        target = self.get_lookahead_target()

        pygame.draw.circle(screen, (255, 0, 255), target, 6)
