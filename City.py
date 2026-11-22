import math
import random
from typing import Literal

import pygame

type IntersectionControl = Literal["stop", "traffic_light", "yield", "none"]

type Approach = Literal["north", "south", "east", "west"]
type Instruction = Literal["stop", "go", "yield", "caution"]

class Intersection:
    def __init__(self, position: pygame.Vector2, control: IntersectionControl = "none"):
        self.position = position
        self.control: IntersectionControl = control

    def update(self, dt: float):
        pass

    def draw(self, screen: pygame.Surface):
        pass

    def get_instruction(self, approach: Approach) -> Instruction:
        if self.control == "stop":
            return "stop"
        if self.control == "yield":
            return "yield"

        return "go"


class TrafficLightIntersection(Intersection):
    def __init__(self, position: pygame.Vector2):
        super().__init__(position, control="traffic_light")
        self.phase = 0
        self.phase_timer = 0

        self.phases = [
            ("ns_green", 8),
            ("ns_yellow", 2),
            ("all_red", 1),
            ("ew_green", 8),
            ("ew_yellow", 2),
            ("all_red", 1),
        ]

    def update(self, dt: float):
        self.phase_timer += dt

        duration = self.phases[self.phase][1]

        if self.phase_timer >= duration:
            self.phase_timer -= duration
            self.phase = (self.phase + 1) % len(self.phases)

    def draw(self, screen: pygame.Surface):
        red = (220, 30, 30)
        yellow = (255, 210, 0)
        green = (30, 220, 60)

        phase_name = self.phases[self.phase][0]

        ns_colour = red
        ew_colour = red

        if phase_name == "ns_green":
            ns_colour = green
        elif phase_name == "ns_yellow":
            ns_colour = yellow
        elif phase_name == "ew_green":
            ew_colour = green
        elif phase_name == "ew_yellow":
            ew_colour = yellow

        x, y = self.position.x, self.position.y

        pygame.draw.circle(screen, ns_colour, (x + 32, y - 48), 7)
        pygame.draw.circle(screen, ns_colour, (x - 32, y + 48), 7)

        pygame.draw.circle(screen, ew_colour, (x - 48, y - 32), 7)
        pygame.draw.circle(screen, ew_colour, (x + 48, y + 32), 7)

    def get_instruction(self, approach: Approach) -> Instruction:
        phase_name = self.phases[self.phase][0]

        north_south = approach in ("north", "south")

        if phase_name == "all_red":
            return "stop"

        if north_south:
            if phase_name == "ns_green":
                return "go"
            elif phase_name == "ns_yellow":
                return "caution"

        else:
            if phase_name == "ew_green":
                return "go"
            elif phase_name == "ew_yellow":
                return "caution"

        return "stop"


class City:
    def __init__(self):
        self.road_width = 80

        self.vertical_roads = [200, 500, 800]
        self.horizontal_roads = [150, 350, 550]

        self.intersections: list[Intersection] = [
            TrafficLightIntersection(pygame.Vector2(x, y))
            for x in self.vertical_roads
            for y in self.horizontal_roads
        ]
        for intersection in self.intersections:
            if isinstance(intersection, TrafficLightIntersection):
                intersection.phase_timer = random.random() * 5

    def update(self, dt: float):
        for intersection in self.intersections:
            intersection.update(dt)

    def draw(self, screen: pygame.Surface):
        for x in self.vertical_roads:
            pygame.draw.rect(
                screen,
                (60, 60, 60),
                (x - self.road_width // 2, 0, self.road_width, screen.get_height()),
            )

        for y in self.horizontal_roads:
            pygame.draw.rect(
                screen,
                (60, 60, 60),
                (0, y - self.road_width // 2, screen.get_width(), self.road_width),
            )

    def draw_markings(self, screen: pygame.Surface):
        yellow = (230, 190, 40)
        white = (220, 220, 220)

        for x in self.vertical_roads:
            for y in range(screen.get_height()):
                if any(abs(y - road) < 45 for road in self.horizontal_roads):
                    continue
                pygame.draw.line(screen, yellow, (x, y), (x, y), 3)

                pygame.draw.line(
                    screen,
                    white,
                    (x - self.road_width // 4, y),
                    (x - self.road_width // 4, y),
                    2,
                )

                pygame.draw.line(
                    screen,
                    white,
                    (x + self.road_width // 4, y),
                    (x + self.road_width // 4, y),
                    2,
                )

        for y in self.horizontal_roads:
            for x in range(screen.get_width()):
                if any(abs(x - road) < 45 for road in self.vertical_roads):
                    continue
                pygame.draw.line(screen, yellow, (x, y), (x, y), 3)

                pygame.draw.line(
                    screen,
                    white,
                    (x, y - self.road_width // 4),
                    (x, y - self.road_width // 4),
                    2,
                )

                pygame.draw.line(
                    screen,
                    white,
                    (x, y + self.road_width // 4),
                    (x, y + self.road_width // 4),
                    2,
                )

    def draw_stop_lines(self, screen: pygame.Surface):
        white = (240, 240, 240)
        half = self.road_width // 2

        for intersection in self.intersections:
            x, y = intersection.position.x, intersection.position.y

            pygame.draw.line(
                screen, white, (x - half, y - half - 5), (x, y - half - 5), 4
            )

            pygame.draw.line(
                screen, white, (x, y + half + 5), (x + half, y + half + 5), 4
            )

            pygame.draw.line(
                screen, white, (x - half - 5, y), (x - half - 5, y + half), 4
            )

            pygame.draw.line(
                screen, white, (x + half + 5, y - half), (x + half + 5, y), 4
            )

    def draw_intersections(self, screen: pygame.Surface):
        for intersection in self.intersections:
            intersection.draw(screen)

    def get_next_intersection(self, position: pygame.Vector2, heading: float) -> tuple[Intersection, Approach, float] | None:
        closest = None
        closest_distance = float("inf")

        forward = pygame.Vector2(math.cos(heading), math.sin(heading))

        for intersection in self.intersections:
            difference = intersection.position - position

            forward_distance = difference.dot(forward)
            side_distance = abs(difference.cross(forward))

            if forward_distance < self.road_width / 2:
                continue

            if side_distance > self.road_width / 2:
                continue

            if forward_distance < closest_distance:
                closest = intersection
                closest_distance = forward_distance

        if closest is None:
            return None

        if abs(forward.x) > abs(forward.y):
            approach = "west" if forward.x > 0 else "east"
        else:
            approach = "north" if forward.y > 0 else "south"

        distance_to_stop_line = closest_distance - (self.road_width / 2 + 5)

        return closest, approach, distance_to_stop_line