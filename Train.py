import pickle
import random
import sys
import time

from Agent import ConvolutionalAgent
from Brain import RELU
from Policy import OBS_CHANNELS, OBS_SIZE, load_agent


def build_agent() -> ConvolutionalAgent:
    return ConvolutionalAgent.new(
        (OBS_SIZE, OBS_SIZE),
        [(3, 3)],
        [24],
        2,
        conv_activation=RELU,
        input_channels=OBS_CHANNELS,
        filters_per_layer=4,
    )


def mse(agent, samples):
    total = 0.0
    for volume, target in samples:
        out = agent.brain.predict_image(volume)
        total += sum((t - o) ** 2 for t, o in zip(target, out)) / len(target)
    return total / max(1, len(samples))


def train(
    path: str = "dataset.pkl",
    out: str = "city_driver.brain",
    epochs: int = 10,
    lr: float = 0.1,
    resume: str | None = None,
):
    with open(path, "rb") as f:
        data = pickle.load(f)
    random.shuffle(data)

    split = max(1, int(len(data) * 0.1))
    val, train_set = data[:split], data[split:]
    print(f"{len(train_set)} train / {len(val)} val")

    agent = load_agent(resume) if resume else build_agent()
    best = float("inf")

    for epoch in range(epochs):
        random.shuffle(train_set)
        start = time.time()
        running = 0.0

        for i, (volume, target) in enumerate(train_set):
            agent.backpropagate(volume, target, lr)
            # outputs still hold the pre-update prediction
            running += (
                sum((t - o) ** 2 for t, o in zip(target, agent.brain.get_output())) / 2
            )
            if (i + 1) % 200 == 0:
                print(
                    f"  {i + 1}/{len(train_set)} loss {running / (i + 1):.4f}",
                    flush=True,
                )

        v = mse(agent, val)
        print(
            f"epoch {epoch + 1}: train {running / len(train_set):.4f} val {v:.4f} ({time.time() - start:.0f}s)"
        )

        if v < best:
            best = v
            agent.brain.save(out)
            print("  saved", out)

        lr *= 0.9


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    resume = sys.argv[2] if len(sys.argv) > 2 else None
    train(epochs=epochs, resume=resume)
