import math
import random
import sys

import pygame

from Car import Car
from City import City
from Expert import Expert, throttle_for
from Policy import SCREEN, Observer, WorldRenderer, decode, load_agent

DT = 1 / 30


LANE = 30
GRID_X = [200, 500, 800]
GRID_Y = [150, 350, 550]

LEFT_RADIUS = 60
RIGHT_RADIUS = 18

LEFT_TURN_LANE = 12
LANE_CHANGE_START = 150

START_DISTANCE = 50
END_DISTANCE = 100

GRID_SIZE = 3
MIN_ROUTE_NODES = 7
MAX_ROUTE_NODES = 11


def _right(d: pygame.Vector2) -> pygame.Vector2:
    return pygame.Vector2(-d.y, d.x)


def fillet(corner, u_in, u_out, radius, steps=12):
    """Arc of the given radius tangent to the incoming and outgoing lane lines."""
    start = corner - u_in * radius
    centre = start + u_out * radius
    a1 = math.atan2(-u_out.y, -u_out.x)
    a2 = math.atan2(u_in.y, u_in.x)
    delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi

    return [
        centre
        + pygame.Vector2(
            math.cos(a1 + delta * i / steps), math.sin(a1 + delta * i / steps)
        )
        * radius
        for i in range(steps + 1)
    ]


def make_random_nodes() -> list[tuple[int, int]]:
    """Create a connected random route, starting southbound."""
    nodes = [(0, 0), (0, 1)]
    target_length = random.randint(MIN_ROUTE_NODES, MAX_ROUTE_NODES)

    while len(nodes) < target_length:
        column, row = nodes[-1]
        previous = nodes[-2]

        candidates = [
            (column + dc, row + dr)
            for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0))
            if 0 <= column + dc < GRID_SIZE
            and 0 <= row + dr < GRID_SIZE
            and (column + dc, row + dr) != previous
        ]

        nodes.append(random.choice(candidates))

    return nodes


def reset_car(car: Car):
    car.position = pygame.Vector2(170, 100)
    car.heading = math.pi / 2
    car.speed = 0.0


def make_route():
    node_indices = make_random_nodes()
    nodes = [pygame.Vector2(GRID_X[c], GRID_Y[r]) for c, r in node_indices]

    dirs = [(b - a).normalize() for a, b in zip(nodes, nodes[1:])]
    dirs.append(dirs[-1])

    route = [nodes[0] + _right(dirs[0]) * LANE - dirs[0] * START_DISTANCE]

    for i, node in enumerate(nodes):
        d_in = dirs[i - 1] if i > 0 else dirs[0]
        d_out = dirs[i]

        if d_in.dot(d_out) > 0.99:
            continue

        left = d_in.cross(d_out) < 0

        if left:
            # Move into the left-turn lane before the intersection.
            route.append(node - d_in * LANE_CHANGE_START + _right(d_in) * LANE)
            route.append(node - d_in * LEFT_RADIUS + _right(d_in) * LEFT_TURN_LANE)

            corner = node + _right(d_in) * LEFT_TURN_LANE + _right(d_out) * LANE

            route.extend(fillet(corner, d_in, d_out, LEFT_RADIUS))

        else:
            corner = node + (_right(d_in) + _right(d_out)) * LANE
            route.extend(fillet(corner, d_in, d_out, RIGHT_RADIUS))

    route.append(nodes[-1] + _right(dirs[-1]) * LANE + dirs[-1] * END_DISTANCE)

    return route


def run(mode: str = "expert", brain_path: str = "city_driver.brain"):
    pygame.init()

    screen = pygame.display.set_mode(SCREEN)
    clock = pygame.time.Clock()

    city = City()
    renderer = WorldRenderer(city)
    car = Car(170, 100)
    reset_car(car)
    expert = Expert(car, city, make_route())

    observer = agent = None
    if mode == "cnn":
        observer = Observer()
        agent = load_agent(brain_path)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        city.update(DT)
        world = renderer.render()

        if agent is None:
            steer, throttle = expert.act(DT)
        else:
            assert observer is not None
            expert.update_target()  # route progress -> nav command
            volume = observer.observe(world, car, expert.get_command())
            steer, target = decode(agent.brain.predict_image(volume))
            throttle = throttle_for(target, car.speed)

        car.update(steer, throttle, DT)

        if expert.done:
            reset_car(car)
            expert.route = make_route()
            expert.reset()

        screen.blit(world, (0, 0))
        expert.draw(screen)
        car.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "expert")
