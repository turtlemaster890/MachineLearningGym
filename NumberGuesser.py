import tkinter

from Agent import Agent
from Brain import Brain

demo_training_data: dict[int, list[list[list[float]]]] = {
    0: [
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
    ],
    1: [
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1],
        ],
    ],
    2: [
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
    ],
    3: [
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
    ],
    4: [
        [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
    ],
    5: [
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
        ],
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
        ],
    ],
    6: [
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
        ],
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
    ],
    7: [
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        [
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ],
    ],
    8: [
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
    ],
    9: [
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
    ],
}


def test():
    agent = Agent.new(5 * 5, hidden_layers=[24], output_size=10)

    for _ in range(5000):
        for number, data in demo_training_data.items():
            for d in data:
                input_data = [pixel for row in d for pixel in row]
                target_output: list[float] = [0] * 10
                target_output[number] = 1
                agent.backpropagate(input_data, target_output, learning_rate=0.1)

    for number in range(10):
        for d in demo_training_data[number]:
            input_data = [pixel for row in d for pixel in row]
            pred = agent.brain.predict(input_data)
            print(
                f"Number: {number}, Predicted: {pred}, Closest: {pred.index(max(pred))}, Confindence: {max(pred)}"
            )
    # test_data = demo_training_data[5][0]
    # input_data = [pixel for row in test_data for pixel in row]
    # pred = agent.brain.predict(input_data)
    # print("Predicted:", agent.brain.predict(input_data))
    # print("Closest:", pred.index(max(pred)))

    test_data: list[list[float]] = [
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0],
    ]

    input_data = [pixel for row in test_data for pixel in row]
    pred = agent.brain.predict(input_data)
    print(
        f"Number: 3, Predicted: {pred}, Closest: {pred.index(max(pred))}, Confidence: {max(pred)}"
    )


W = 10
H = 10
CELL_SIZE = 40


class DrawingPanel(tkinter.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.grid_data: list[list[float]] = [[0 for _ in range(W)] for _ in range(H)]

        self.bind("<Button-1>", self.draw)
        self.bind("<B1-Motion>", self.draw)

        for x in range(W + 1):
            self.create_line(
                x * CELL_SIZE, 0, x * CELL_SIZE, H * CELL_SIZE, fill="#484848"
            )

        for y in range(H + 1):
            self.create_line(
                0, y * CELL_SIZE, W * CELL_SIZE, y * CELL_SIZE, fill="#484848"
            )

    def draw(self, event: tkinter.Event):
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE

        if 0 <= x < W and 0 <= y < H:
            if self.grid_data[y][x] == 0:
                self.grid_data[y][x] = 1
                self.create_rectangle(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    (x + 1) * CELL_SIZE,
                    (y + 1) * CELL_SIZE,
                    fill="black",
                    outline="black",
                )

    def get_flattened_data(self) -> list[float]:
        return [pixel for row in self.grid_data for pixel in row]

    def clear(self):
        self.grid_data = [[0 for _ in range(W)] for _ in range(H)]
        self.delete("all")
        for x in range(W + 1):
            self.create_line(
                x * CELL_SIZE, 0, x * CELL_SIZE, H * CELL_SIZE, fill="#484848"
            )

        for y in range(H + 1):
            self.create_line(
                0, y * CELL_SIZE, W * CELL_SIZE, y * CELL_SIZE, fill="#484848"
            )


type Data = list[list[float]]


def gui():
    agent = Agent.new(W * H, hidden_layers=[5, 5], output_size=10)

    training_data: dict[int, list[Data]] = {}

    root = tkinter.Tk()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    drawing = DrawingPanel(root, width=W * CELL_SIZE, height=H * CELL_SIZE)
    drawing.grid(column=0, row=1, sticky="sew")

    state: int | None = 0

    prompt = tkinter.Label(root, text="Draw a 0!")
    prompt.grid(column=0, row=0, sticky="new")

    def sub_drawing():
        if state is None:
            o = agent.brain.predict(drawing.get_flattened_data())
            prompt.config(
                text=f"PREDICTION: {o.index(max(o))} with confidence {max(o)}"
            )
            drawing.clear()
        else:
            training_data.setdefault(state, []).append(drawing.grid_data)
            # for _ in range(5):
            # agent.backpropagate(
            #     drawing.get_flattened_data(),
            #     [1 if i == state else 0 for i in range(10)],
            #     learning_rate=0.1,
            # )
            prompt.config(
                text=f"Great! Draw another {state}, or press Next to move on."
            )
            drawing.clear()

    def next_state():
        nonlocal state
        if state is None:
            prompt.config(text="Draw a number and press Submit!")
        else:
            if training_data.get(state) is None:
                prompt.config(
                    text=f"You haven't drawn any {state} yet! Draw one and press Submit!"
                )
                return
            if state < 9:
                state += 1
                prompt.config(text=f"Great! now draw a {state}!")
            else:
                state = None
                prompt.config(text="Starting training...")
                prompt.update_idletasks()
                for _ in range(5000):
                    for number, data_list in training_data.items():
                        for data in data_list:
                            agent.backpropagate(
                                [pixel for row in data for pixel in row],
                                [1 if i == number else 0 for i in range(10)],
                                learning_rate=0.1,
                            )
                prompt.config(text="All done! Now draw a number and press Submit!")

    buttonFrame = tkinter.Frame(root)
    buttonFrame.grid(row=2, column=0, sticky="w")

    submitButton = tkinter.Button(buttonFrame, text="Submit", command=sub_drawing)
    submitButton.pack(side="left")

    nextButton = tkinter.Button(buttonFrame, text="Next", command=next_state)
    nextButton.pack(side="left")

    saveButton = tkinter.Button(
        buttonFrame,
        text="Save",
        command=lambda: (
            agent.brain.save("numbers.brain"),
            prompt.config(
                text="Brain saved successfully! Draw a number and press Submit!"
            ),
        ),
    )
    saveButton.pack(side="left")

    def load_brain():
        nonlocal agent, state
        try:
            agent = Agent(Brain.load("numbers.brain"))
            prompt.config(
                text="Brain loaded successfully! Draw a number and press Submit!"
            )
            state = None
        except Exception as e:
            prompt.config(text=f"Failed to load brain: {e}")

    loadButton = tkinter.Button(buttonFrame, text="Load", command=lambda: load_brain())
    loadButton.pack(side="left")

    def train(training_data: Data, label: int, num_epochs=5000, learning_rate=0.1):
        for _ in range(num_epochs):
            agent.backpropagate(
                [pixel for row in training_data for pixel in row],
                [1 if i == label else 0 for i in range(10)],
                learning_rate=learning_rate,
            )
        prompt.config(text=f"Trained label {label}.")
        drawing.clear()

    labelEntry = tkinter.Entry(buttonFrame)

    trainButton = tkinter.Button(
        buttonFrame,
        text="Train",
        command=lambda: train(drawing.grid_data, int(labelEntry.get()), num_epochs=500),
    )
    trainButton.pack(side="left")

    labelEntry.pack(side="left")

    root.mainloop()


if __name__ == "__main__":
    # main()
    gui()
