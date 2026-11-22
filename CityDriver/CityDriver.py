import math

import pygame
from Car import Car
from City import City
from Expert import Expert


def quarter_turn(center, radius, start_angle, end_angle, steps=10):
    points = []

    for i in range(steps + 1):
        t = i / steps
        angle = start_angle + (end_angle - start_angle) * t

        point = pygame.Vector2(math.cos(angle), math.sin(angle)) * radius

        points.append(center + point)

    return points


def run():
    pygame.init()

    screen = pygame.display.set_mode((1000, 700))
    clock = pygame.time.Clock()

    city = City()
    car = Car(170, 100)

    route = [
        pygame.Vector2(170, 100),
        pygame.Vector2(170, 550),
        *quarter_turn(pygame.Vector2(200, 550), 30, math.pi, math.pi / 2),
        pygame.Vector2(800, 580),
        *quarter_turn(pygame.Vector2(800, 550), 30, math.pi / 2, 0),
        pygame.Vector2(830, 100),
    ]

    driver = Expert(car, city, route)
    # intersection = city.intersections[0]

    # print(intersection.get_instruction("north"))
    # print(intersection.get_instruction("east"))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.tick(60) / 1000
        city.update(dt)
        steer = driver.update(dt)
        target_speed = driver.get_target_speed()
        throttle = 0.5 if car.speed < target_speed else 0
        car.update(steer, throttle, dt)

        # keys = pygame.key.get_pressed()
        # steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        # throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]

        # car.update(steer, throttle, dt)

        screen.fill((40, 120, 40))
        city.draw(screen)
        city.draw_markings(screen)
        city.draw_stop_lines(screen)
        city.draw_intersections(screen)

        driver.draw(screen)

        car.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
