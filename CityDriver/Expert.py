import math

import pygame
from Car import Car


class Expert:
    def __init__(self, car: Car, route: list[pygame.Vector2]):
        self.car = car
        self.route = route
        self.target_index = 0
        self.lookahead_distance = 65

    def get_target(self) -> pygame.Vector2:
        return self.route[self.target_index]

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
        pygame.draw.lines(
            screen,
            (0, 255, 255),
            False,
            self.route,
            2
        )

        target = self.get_lookahead_target()

        pygame.draw.circle(
            screen, (255, 0, 255),
            target, 6
        )