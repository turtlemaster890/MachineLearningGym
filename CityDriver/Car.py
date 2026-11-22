import math

import pygame


class Car:
    def __init__(self, x: float, y: float) -> None:
        self.position = pygame.Vector2(x, y)
        self.heading = 0.0
        self.speed = 0.0

        self.length = 28
        self.width = 16

    def update(self, steer: float, throttle: float, dt: float) -> None:
        self.heading += steer * 2.5 * dt

        if throttle > 0:
            self.speed += throttle * 100 * dt
        else:
            self.speed -= 180 * dt
        self.speed += throttle * 100 * dt
        
        self.speed = max(0, min(self.speed, 120))

        direction = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))

        self.position += direction * self.speed * dt

    def draw(self, screen: pygame.Surface) -> None:
        forward = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        right = pygame.Vector2(-forward.y, forward.x)

        corners = [
            self.position + forward * self.length / 2 + right * self.width / 2,
            self.position + forward * self.length / 2 - right * self.width / 2,
            self.position - forward * self.length / 2 - right * self.width / 2,
            self.position - forward * self.length / 2 + right * self.width / 2,
        ]

        pygame.draw.polygon(screen, (40, 130, 255), corners)
