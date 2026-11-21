import math
import random

import pygame

from Agent import EvolutionAgent
from Brain import Brain

WIDTH, HEIGHT = 1000, 700
POPULATION_SIZE = 100
MAX_GEN_SECONDS = 18

RAY_ANGLES: list[float] = [-75, -45, -15, 0, 15, 45, 75]
MAX_RAY_LENGTH: float = 220
ROAD_COLOUR = (45, 45, 45)

STARTING_POS = (500, 565)
STARTING_HEADING = 0


def cast_rays(
    track: pygame.Surface, x: float, y: float, heading_rad: float
) -> list[float]:
    width, height = track.get_size()
    readings: list[float] = []

    for deg_offset in RAY_ANGLES:
        angle = heading_rad + math.radians(deg_offset)
        dx = math.cos(angle)
        dy = math.sin(angle)

        dist: float = 0
        while dist < MAX_RAY_LENGTH:
            dist += 4
            rx = int(x + dx * dist)
            ry = int(y + dy * dist)

            if rx < 0 or ry < 0 or rx >= width or ry >= height:
                break
            if track.get_at((rx, ry))[:3] != ROAD_COLOUR:
                break

        readings.append(min(dist / MAX_RAY_LENGTH, 1))

    return readings


def mut_agent(
    agent: EvolutionAgent, mutation_rate: float = 0.25, mutation_scale: float = 0.15
) -> EvolutionAgent:
    child = agent.copy()

    for layer in child.brain.hiddenLayers:
        for neuron in layer.neurons:
            neuron.weights = [
                (
                    w + random.gauss(0, mutation_scale)
                    if random.random() < mutation_rate
                    else w
                )
                for w in neuron.weights
            ]
            if random.random() < mutation_rate:
                neuron.bias += random.gauss(0, mutation_scale)

    for neuron in child.brain.outputLayer.neurons:
        neuron.weights = [
            (
                w + random.gauss(0, mutation_scale)
                if random.random() < mutation_rate
                else w
            )
            for w in neuron.weights
        ]
        if random.random() < mutation_rate:
            neuron.bias += random.gauss(0, mutation_scale)

    return child


def create_next_generation(
    scored_agents: list[tuple[EvolutionAgent, float]],
    population_size: int = 30,
    elite_count: int = 3,
    top_pool_size: int = 8,
) -> list[EvolutionAgent]:
    scored_agents.sort(key=lambda x: x[1], reverse=True)
    ranked = [agent for agent, _ in scored_agents]

    next_gen: list[EvolutionAgent] = []

    for i in range(elite_count):
        next_gen.append(ranked[i].copy())

    parent_pool = ranked[: max(1, min(top_pool_size, len(ranked)))]
    while len(next_gen) < population_size:
        parent = random.choice(parent_pool)
        scale = 0.12 if len(next_gen) < population_size * 0.75 else 0.25
        next_gen.append(mut_agent(parent, mutation_rate=0.3, mutation_scale=scale))

    return next_gen


def make_oval_track(width: int, height: int):
    surf = pygame.Surface((width, height))
    surf.fill((34, 139, 34))

    pygame.draw.ellipse(surf, ROAD_COLOUR, (90, 90, width - 180, height - 180))
    pygame.draw.ellipse(surf, (34, 139, 34), (240, 190, width - 480, height - 380))
    return surf


def rotate_vector(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class Car:
    def __init__(self, agent: EvolutionAgent | None = None):
        self.agent = (
            agent
            if agent is not None
            else EvolutionAgent.new(len(RAY_ANGLES), [8, 6], 2)
        )
        self.x = STARTING_POS[0]
        self.y = STARTING_POS[1]
        self.heading = STARTING_HEADING
        self.speed: float = 120
        self.distance: float = 0
        self.alive = True
        self.frames_alive = 0
        self.idle_frames = 0

    @property
    def fitness(self) -> float:
        avg_speed = self.distance / max(self.frames_alive, 1)
        return self.distance + avg_speed * 250

    def update(self, track: pygame.Surface, dt: float):
        if not self.alive:
            return

        self.frames_alive += 1
        sensors = cast_rays(track, self.x, self.y, self.heading)
        steer_raw, accel_raw = self.agent.brain.predict(sensors)

        steer = (steer_raw - 0.5) * 2
        throttle: float = lerp(0.35, 1, accel_raw)

        target_speed: float = 105 + throttle * 150
        self.speed += (target_speed - self.speed) * min(1, dt * 4)
        self.heading += steer * 3.2 * dt

        step_dist = self.speed * dt
        self.x += math.cos(self.heading) * step_dist
        self.y += math.sin(self.heading) * step_dist
        self.distance += step_dist

        if self.speed < 105:
            self.idle_frames += 1
        else:
            self.idle_frames = 0

        ix, iy = int(self.x), int(self.y)
        off_track = (
            ix < 0
            or iy < 0
            or ix >= track.get_width()
            or iy >= track.get_height()
            or track.get_at((ix, iy))[:3] != ROAD_COLOUR
        )
        if off_track or self.idle_frames > 90:
            self.alive = False

    def draw(self, screen: pygame.Surface, is_leader: bool):
        if not self.alive and not is_leader:
            return
        colour = (255, 215, 0) if is_leader else (80, 180, 255)
        iX, iY = int(self.x), int(self.y)
        v1 = rotate_vector(-5, -5, self.heading)
        v2 = rotate_vector(10, 0, self.heading)
        v3 = rotate_vector(-5, 5, self.heading)
        pygame.draw.polygon(
            screen,
            colour,
            [
                (iX, iY),
                (iX + v1[0], iY + v1[1]),
                (iX + v2[0], iY + v2[1]),
                (iX + v3[0], iY + v3[1]),
            ],
            6,
        )


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Racing Neural Networks")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    track = make_oval_track(WIDTH, HEIGHT)
    generation = 1
    gen_elapsed: float = 0
    best_all_time: float = 0
    leader: Car | None = None

    cars = [
        Car(EvolutionAgent.new(len(RAY_ANGLES), hidden_layers=[8, 6], output_size=2))
        for _ in range(POPULATION_SIZE)
    ]

    running = True
    while running:
        dt = min(clock.tick(60) / 1000, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.type == pygame.K_s:
                    if leader:
                        leader.agent.brain.save("racing.brain")
                    else:
                        screen.blit(
                            font.render("No leader to save.", True, (255, 0, 0)),
                            (16, 40),
                        )
                if event.type == pygame.K_l:
                    try:
                        brain = Brain.load("racing.brain")
                        cars[0].agent = EvolutionAgent(brain)
                        screen.blit(
                            font.render(
                                "Brain loaded successfully.", True, (0, 255, 0)
                            ),
                            (16, 60),
                        )
                    except Exception as e:
                        screen.blit(
                            font.render(
                                f"Failed to load brain: {e}", True, (255, 0, 0)
                            ),
                            (16, 60),
                        )

        gen_elapsed += dt
        for car in cars:
            car.update(track, dt)

        leader = max(cars, key=lambda c: c.fitness)
        best_all_time = max(best_all_time, leader.fitness)

        alive_count = sum(1 for c in cars if c.alive)
        if alive_count == 0 or gen_elapsed >= MAX_GEN_SECONDS:
            agents = [(car.agent, car.fitness) for car in cars]
            nextAgents = create_next_generation(
                agents, population_size=POPULATION_SIZE, elite_count=3, top_pool_size=8
            )
            cars = [Car(agent) for agent in nextAgents]
            generation += 1
            gen_elapsed = 0

        screen.blit(track, (0, 0))
        for car in cars:
            car.draw(screen, is_leader=(car is leader))

        hud = font.render(
            f"Generation: {generation} | Alive: {alive_count}/{POPULATION_SIZE} | Best Fitness: {best_all_time:.1f} | Leader Speed: {leader.speed:.1f}\nUse 'S' to save leader brain, 'L' to load brain.",
            True,
            (255, 255, 255),
        )
        screen.blit(hud, (16, 14))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
