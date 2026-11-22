import math

import pygame

from Agent import ConvolutionalAgent
from Brain import RELU, ConvolutionalBrain

SCREEN = (1000, 700)
GRASS = (40, 120, 40)

OBS_SIZE = 16  # CNN input is OBS_SIZE x OBS_SIZE
OBS_CHANNELS = 4  # R, G, B + nav command plane
VIEW = 240  # world pixels covered by the view
AHEAD = 80  # view centre is this far in front of the car
CRUISE = 80


class WorldRenderer:
    """City only (no car / debug overlays): what a camera would see."""

    def __init__(self, city):
        self.city = city
        self.static = pygame.Surface(SCREEN)
        self.static.fill(GRASS)
        city.draw(self.static)
        city.draw_markings(self.static)
        city.draw_stop_lines(self.static)
        self.world = pygame.Surface(SCREEN)

    def render(self) -> pygame.Surface:
        self.world.blit(self.static, (0, 0))
        self.city.draw_intersections(self.world)
        return self.world


class Observer:
    """Egocentric top-down view: car faces up, downscaled to OBS_SIZE^2."""

    def __init__(self):
        self.big = math.ceil(VIEW * math.sqrt(2)) + 2
        self.margin = self.big // 2 + AHEAD + 10
        self.padded = pygame.Surface(
            (SCREEN[0] + 2 * self.margin, SCREEN[1] + 2 * self.margin)
        )

    def observe(
        self, world: pygame.Surface, car, command: float
    ) -> list[list[list[float]]]:
        self.padded.fill(GRASS)
        self.padded.blit(world, (self.margin, self.margin))

        forward = pygame.Vector2(math.cos(car.heading), math.sin(car.heading))
        centre = car.position + forward * AHEAD

        rect = pygame.Rect(0, 0, self.big, self.big)
        rect.center = (round(centre.x) + self.margin, round(centre.y) + self.margin)
        rect.clamp_ip(self.padded.get_rect())

        rotated = pygame.transform.rotate(
            self.padded.subsurface(rect), 90 + math.degrees(car.heading)
        )
        view_rect = pygame.Rect(0, 0, VIEW, VIEW)
        view_rect.center = rotated.get_rect().center
        small = pygame.transform.smoothscale(
            rotated.subsurface(view_rect), (OBS_SIZE, OBS_SIZE)
        )

        n = OBS_SIZE
        volume = [[[0.0] * n for _ in range(n)] for _ in range(3)]
        for y in range(n):
            for x in range(n):
                c = small.get_at((x, y))
                volume[0][y][x] = c.r / 255
                volume[1][y][x] = c.g / 255
                volume[2][y][x] = c.b / 255
        volume.append([[command] * n for _ in range(n)])
        return volume


def encode(steer: float, target_speed: float) -> list[float]:
    return [0.5 + 0.45 * steer, 0.05 + 0.9 * min(target_speed, CRUISE) / CRUISE]


def decode(output: list[float]) -> tuple[float, float]:
    steer = max(-1.0, min(1.0, (output[0] - 0.5) / 0.45))
    speed = max(0.0, min(CRUISE, (output[1] - 0.05) / 0.9 * CRUISE))
    return steer, speed

def load_agent(path: str) -> ConvolutionalAgent:
    return ConvolutionalAgent(ConvolutionalBrain.load(path, RELU))  # type: ignore