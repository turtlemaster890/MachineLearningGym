import os
import pickle
import random
import sys

from CityDriver import DT, make_route, reset_car

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from Car import Car
from City import City, TrafficLightIntersection
from Expert import Expert, throttle_for
from Policy import Observer, WorldRenderer, decode, encode, load_agent

SAMPLE_EVERY = 5  # frames between samples (6 Hz at DT=1/30)


def collect(
    laps: int = 20, path: str = "dataset.pkl", agent=None, max_time: float = 200.0
):
    """Expert labels every sample.
    agent=None : expert drives, with random steering kicks so we also see recoveries.
    agent=CNN  : CNN drives, expert only labels (DAgger)."""
    pygame.init()
    city = City()
    renderer = WorldRenderer(city)
    observer = Observer()

    data = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)

    for lap in range(laps):
        for inter in city.intersections:  # vary light timing between laps
            if isinstance(inter, TrafficLightIntersection):
                inter.phase = random.randrange(len(inter.phases))
                inter.phase_timer = random.uniform(0, 3)

        car = Car(170, 100)
        reset_car(car)
        car.position.x += random.uniform(-10, 10)
        car.heading += random.uniform(-0.25, 0.25)
        car.speed = random.uniform(0, 60)
        expert = Expert(car, city, make_route())

        t, frame = 0.0, 0
        noise, noise_left = 0.0, 0.0
        held = (0.0, 0.0)
        start = len(data)

        while not expert.done and t < max_time:
            city.update(DT)
            steer, throttle = expert.act(DT)  # expert label for the current state

            if frame % SAMPLE_EVERY == 0:
                keep = not (
                    car.speed < 1 and random.random() < 0.8
                )  # skip most idle frames
                if keep or agent is not None:
                    volume = observer.observe(
                        renderer.render(), car, expert.get_command()
                    )
                    if keep:
                        data.append((volume, encode(steer, expert.target_speed)))
                    if agent is not None:
                        s, v = decode(agent.brain.predict_image(volume))
                        held = (s, throttle_for(v, car.speed))

            if agent is not None:
                a_steer, a_throttle = held
                if expert.distance_to_route() > 150:
                    break  # CNN got lost
            else:
                if noise_left > 0:
                    noise_left -= DT
                else:
                    noise = 0.0
                    if random.random() < DT / 2.5:
                        noise = random.uniform(-0.6, 0.6)
                        noise_left = random.uniform(0.2, 0.5)
                a_steer = max(-1.0, min(1.0, steer + noise))
                a_throttle = throttle

            car.update(a_steer, a_throttle, DT)
            t += DT
            frame += 1

        print(
            f"lap {lap + 1}/{laps}: {len(data) - start} samples, finished={expert.done}"
        )

    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("total samples:", len(data))


if __name__ == "__main__":
    laps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    brain = sys.argv[2] if len(sys.argv) > 2 else None  # pass a brain path for DAgger
    collect(laps, agent=load_agent(brain) if brain else None)
