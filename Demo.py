from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
os.chdir(APP_DIR)


class Demo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Neural Network Demos")
        self.root.geometry("440x300")
        self.root.resizable(False, False)
        self.root.configure(bg="#111827")

        tk.Label(
            root,
            text="NEURAL NETWORK DEMOS",
            bg="#111827",
            fg="white",
            font=("Segoe UI Semibold", 17),
        ).pack(pady=(24, 18))

        buttons = tk.Frame(root, bg="#111827")
        buttons.pack(padx=28, fill="x")

        self.buttons = []
        entries = [
            ("Expert Driver", lambda: self.launch("CityDriver", "run", "expert")),
            ("CNN Driver", lambda: self.launch("CityDriver", "run", "cnn", "cnn_brain.json")),
            ("Racing", lambda: self.launch("Racing", "main")),
            ("Number Guesser", lambda: self.launch("NumberGuesser", "gui")),
            ("Train City Driver", self.train_city_driver),
        ]

        for i, (label, command) in enumerate(entries):
            button = tk.Button(
                buttons,
                text=label,
                command=command,
                bg="#26364d",
                fg="white",
                activebackground="#38577d",
                activeforeground="white",
                relief="flat",
                borderwidth=0,
                font=("Segoe UI Semibold", 11),
                cursor="hand2",
                height=2,
            )
            button.grid(
                row=i // 2,
                column=i % 2,
                sticky="ew",
                padx=5,
                pady=5,
            )
            self.buttons.append(button)

        buttons.columnconfigure((0, 1), weight=1)

        self.status = tk.StringVar(value="Ready")
        tk.Label(
            root,
            textvariable=self.status,
            bg="#111827",
            fg="#9caec4",
            font=("Segoe UI", 9),
        ).pack(pady=(13, 0))

    def set_busy(self, busy: bool, text: str):
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
        self.status.set(text)

    def launch(self, module_name: str, function_name: str, *args):
        self.root.withdraw()
        try:
            import importlib

            module = importlib.import_module(module_name)
            getattr(module, function_name)(*args)
        except Exception as exc:
            messagebox.showerror(
                "Demo failed",
                f"{type(exc).__name__}: {exc}",
                parent=self.root,
            )
        finally:
            self.root.deiconify()
            self.root.lift()
            self.status.set("Ready")

    def train_city_driver(self):
        if not (APP_DIR / "dataset.pkl").exists():
            # The first collection stage creates the dataset.
            pass

        self.set_busy(True, "Training City Driver...")
        threading.Thread(target=self._training_worker, daemon=True).start()

    def _training_worker(self):
        try:
            import Collect
            import Train
            from Policy import load_agent

            # 1. Expert demonstrations -> dataset.pkl
            self._set_status("1/4  Collecting expert data...")
            Collect.collect(20, path="dataset.pkl")

            # 2. Initial CNN training -> cnn_brain.json
            self._set_status("2/4  Training initial CNN...")
            Train.train(path="dataset.pkl", out="cnn_brain.json", epochs=10)

            # 3. DAgger: CNN drives, expert labels -> append to dataset.pkl
            self._set_status("3/4  Collecting DAgger data...")
            cnn_agent = load_agent("cnn_brain.json")
            Collect.collect(10, path="dataset.pkl", agent=cnn_agent)

            # 4. Fine-tune the same CNN on the expanded dataset
            self._set_status("4/4  Fine-tuning CNN...")
            Train.train(
                path="dataset.pkl",
                out="cnn_brain.json",
                epochs=6,
                resume="cnn_brain.json",
            )

        except Exception as exc:
            self.root.after(0, self._training_failed, exc)
        else:
            self.root.after(0, self._training_finished)

    def _set_status(self, text: str):
        self.root.after(0, self.status.set, text)

    def _training_finished(self):
        self.set_busy(False, "Training complete")
        messagebox.showinfo(
            "Training complete",
            "City Driver has been trained.\n\nModel saved as cnn_brain.json.",
            parent=self.root,
        )

    def _training_failed(self, exc: Exception):
        self.set_busy(False, "Training failed")
        messagebox.showerror(
            "Training failed",
            f"{type(exc).__name__}: {exc}",
            parent=self.root,
        )


def main():
    root = tk.Tk()
    Demo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
