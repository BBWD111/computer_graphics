import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror
import numpy as np
import time


class PixelPath(tk.Tk):
    time = 0

    def __init__(self):
        super().__init__()
        self.title("PixelPath")
        self.geometry("800x600")
        self.style = ttk.Style(self)
        self.style.configure("TButton", font=("Arial", 12), padding=(0, 0))
        self.style.configure("TFrame", background="lightblue")

        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # дискретные уровни зума: сколько пикселей на 1 единицу координат
        self.zoom_levels = [5, 10, 20, 40, 80]
        self.zoom_index = 2          # стартуем с 20 px/ед
        self.scale = self.zoom_levels[self.zoom_index]

        self.start_x = 400   # экранные координаты (0,0)
        self.start_y = 300
        self.origin_initialized = False

        self.x0_entry = None
        self.x1_entry = None
        self.y0_entry = None
        self.y1_entry = None
        self.o1_entry = None
        self.o2_entry = None
        self.radius_entry = None

        self.lines = []      # [x0, y0, x1, y1, 's'/'d'/'b'/'c'] или 'circle'
        self.circles = []    # [cx, cy, r]

        self.hover_button = None
        self.param_frame = None

        self.pan_start_x = 0
        self.pan_start_y = 0

        # бинды
        self.canvas.bind("<B1-Motion>", self.pan_canvas)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Configure>", lambda event: self.redraw())

        # зум с клавиатуры
        self.bind("+", lambda e: self.zoom_with_scroll(e, "in"))
        self.bind("-", lambda e: self.zoom_with_scroll(e, "out"))
        self.bind("<KP_Add>", lambda e: self.zoom_with_scroll(e, "in"))
        self.bind("<KP_Subtract>", lambda e: self.zoom_with_scroll(e, "out"))

        self.create_main_interface()

    # --------- UI ---------

    def create_main_interface(self):
        main_frame = tk.Frame(self, width=50, height=20, padx=0, pady=0)
        main_frame.place(x=0, y=0)

        self.hover_button = ttk.Button(main_frame, text="Menu",
                                       command=self.open_params_window)
        self.hover_button.pack(pady=0)

    def open_params_window(self):
        self.param_frame = tk.Frame(self, width=390, height=120, padx=10, pady=10)
        self.param_frame.place(x=0, y=0)

        part_window = self.param_frame

        a_label = tk.Label(part_window, text="A(x, y): ")
        a_label.grid(row=0, column=0, padx=(5, 2), pady=0, sticky="w")

        self.x0_entry = ttk.Entry(part_window, width=5)
        self.x0_entry.grid(row=0, column=0, padx=(0, 0), pady=0)

        self.y0_entry = ttk.Entry(part_window, width=5)
        self.y0_entry.grid(row=0, column=0, padx=(120, 0), pady=0)

        b_label = tk.Label(part_window, text="B(x, y): ")
        b_label.grid(row=1, column=0, padx=(5, 2), pady=0, sticky="w")

        self.x1_entry = ttk.Entry(part_window, width=5)
        self.x1_entry.grid(row=1, column=0, padx=(2, 2), pady=0)

        self.y1_entry = ttk.Entry(part_window, width=5)
        self.y1_entry.grid(row=1, column=0, padx=(120, 0), pady=0)

        o_label = tk.Label(part_window, text="O(x, y): ")
        o_label.grid(row=2, column=0, padx=(5, 2), pady=0, sticky="w")

        self.o1_entry = ttk.Entry(part_window, width=5)
        self.o1_entry.grid(row=2, column=0, padx=(2, 2), pady=0)

        self.o2_entry = ttk.Entry(part_window, width=5)
        self.o2_entry.grid(row=2, column=0, padx=(120, 0), pady=0)

        radius_label = tk.Label(part_window, text="R:")
        radius_label.grid(row=3, column=0, padx=(37, 2), pady=0, sticky="w")

        self.radius_entry = ttk.Entry(part_window, width=5)
        self.radius_entry.grid(row=3, column=0, padx=(6, 5), pady=0)

        step_button = ttk.Button(part_window, text="Step-By-Step",
                                 command=self.create_line, width=15)
        step_button.grid(row=4, column=0, padx=0, pady=0)

        dda_button = ttk.Button(part_window, text="DDA",
                                command=self.dda, width=15)
        dda_button.grid(row=5, column=0, padx=0, pady=0)

        bresenham_button = ttk.Button(part_window, text="Bresenham's line",
                                      command=self.bresenham_line, width=15)
        bresenham_button.grid(row=6, column=0, padx=0, pady=0)

        castle_button = ttk.Button(part_window, text="Castle-Pitway Line",
                                   command=self.castle_pitway, width=15)
        castle_button.grid(row=7, column=0, padx=0, pady=0)

        circle_button = ttk.Button(part_window, text="Bresenham's circle",
                                   command=self.bresenham_circle, width=15)
        circle_button.grid(row=8, column=0, padx=0, pady=0)

        undo_button = ttk.Button(part_window, text="Undo", command=self.undo)
        undo_button.grid(row=9, column=0, padx=0, pady=0, sticky="w")

        hide_button = ttk.Button(part_window, text="Hide", command=self.hide_settings)
        hide_button.grid(row=9, column=0, padx=0, pady=0, sticky="e")

        # кнопки зума
        zoom_in_button = ttk.Button(part_window, text="Zoom +",
                                    command=lambda: self.zoom_with_scroll(None, "in"),
                                    width=15)
        zoom_in_button.grid(row=10, column=0, padx=0, pady=0)

        zoom_out_button = ttk.Button(part_window, text="Zoom -",
                                     command=lambda: self.zoom_with_scroll(None, "out"),
                                     width=15)
        zoom_out_button.grid(row=11, column=0, padx=0, pady=0)

    def hide_settings(self):
        if self.param_frame:
            self.param_frame.place_forget()

    # --------- координаты / сетка / зум ---------

    def draw_grid(self):
        """Сетка с учётом масштаба и диапазона [-100, 100]."""
        self.canvas.delete("grid")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if not self.origin_initialized:
            self.start_x = width // 2
            self.start_y = height // 2
            self.origin_initialized = True

        cx, cy = self.start_x, self.start_y

        # оси
        self.canvas.create_line(cx, 0, cx, height, fill="black", tags="grid", width=2)
        self.canvas.create_line(0, cy, width, cy, fill="black", tags="grid", width=2)

        # стрелки + подписи X/Y
        arrow_size = 6
        # ось Y (верх)
        self.canvas.create_line(cx, 0, cx - arrow_size, arrow_size, fill="black", tags="grid")
        self.canvas.create_line(cx, 0, cx + arrow_size, arrow_size, fill="black", tags="grid")
        self.canvas.create_text(cx - 15, 0, text="Y", anchor="nw",
                                font=("Arial", 12), fill="black", tags="grid")
        # ось X (право)
        self.canvas.create_line(width, cy, width - arrow_size, cy - arrow_size,
                                fill="black", tags="grid")
        self.canvas.create_line(width, cy, width - arrow_size, cy + arrow_size,
                                fill="black", tags="grid")
        self.canvas.create_text(width - 15, cy - 18, text="X", anchor="nw",
                                font=("Arial", 12), fill="black", tags="grid")

        # шаг сетки по единицам в зависимости от масштаба
        if self.scale >= 40:
            grid_step = 1
        elif self.scale >= 20:
            grid_step = 2
        elif self.scale >= 10:
            grid_step = 5
        else:
            grid_step = 10

        max_units = 100

        # вертикальные линии и подписи по X
        for ux in range(-max_units, max_units + 1):
            if ux % grid_step != 0:
                continue
            x = cx + ux * self.scale
            self.canvas.create_line(x, 0, x, height, fill="gray", tags="grid")
            if ux != 0:
                # подпись раз в grid_step единиц
                self.canvas.create_text(x, cy + 10, text=str(ux),
                                        fill="black", tags="grid")

        # горизонтальные линии и подписи по Y
        for uy in range(-max_units, max_units + 1):
            if uy % grid_step != 0:
                continue
            y = cy + uy * self.scale
            self.canvas.create_line(0, y, width, y, fill="gray", tags="grid")
            if uy != 0:
                # по математике вверх — плюс, поэтому знак инвертируем
                self.canvas.create_text(cx + 15, y, text=str(-uy),
                                        fill="black", tags="grid")

        # центр
        self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                fill="black", tags="grid")
        self.canvas.create_text(cx + 11, cy + 10, text="0,0",
                                fill="black", tags="grid")

    def redraw(self):
        """Полная перерисовка сцены."""
        self.canvas.delete("all")
        self.draw_grid()

        for item in self.lines:
            if item == 'circle':
                continue
            x0, y0, x1, y1, t = item
            func = self.get_reprint_func(t)
            if func:
                func(x0, y0, x1, y1, new=False)

        for cx, cy, r in self.circles:
            self.bresenham_circle(cx, cy, r, new=False)

    def zoom_with_scroll(self, event, direction=None):
        """Дискретный зум: direction='in'/'out'."""
        if direction == "in":
            if self.zoom_index < len(self.zoom_levels) - 1:
                self.zoom_index += 1
        elif direction == "out":
            if self.zoom_index > 0:
                self.zoom_index -= 1
        else:
            return

        self.scale = self.zoom_levels[self.zoom_index]
        self.redraw()

    # --------- панорамирование ---------

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_canvas(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.start_x += dx
        self.start_y += dy
        self.canvas.move("all", dx, dy)
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    # --------- служебные методы ---------

    def get_reprint_func(self, type_line):
        if type_line == 's':
            return self.create_line
        if type_line == 'd':
            return self.dda
        if type_line == 'b':
            return self.bresenham_line
        if type_line == 'c':
            return self.castle_pitway
        return None

    def undo(self):
        if not self.lines and not self.circles:
            return

        if self.lines and self.lines[-1] == 'circle':
            self.lines.pop()
            if self.circles:
                self.circles.pop()
        elif self.lines:
            self.lines.pop()

        self.redraw()

    def get_points(self, alg="simple"):
        """Чтение A,B в логических координатах."""
        try:
            x0 = int(self.x0_entry.get())
            y0 = int(self.y0_entry.get())
            x1 = int(self.x1_entry.get())
            y1 = int(self.y1_entry.get())
            if x0 == x1 and y0 == y1:
                showerror(title="Error", message="Input two different points")
                return
        except ValueError:
            showerror(title="Error",
                      message="Incorrect input: input must be an integer")
            return

        # внутренний Y направлен вниз -> инвертируем
        y0i, y1i = -y0, -y1

        rec = [x0, y0i, x1, y1i, None]
        if alg == "simple":
            rec[-1] = 's'
        elif alg == "dda":
            rec[-1] = 'd'
        elif alg == "bres":
            rec[-1] = 'b'
        elif alg == "castle":
            rec[-1] = 'c'
        self.lines.append(rec)

        return x0, y0i, x1, y1i

    def get_circle_coordinates(self):
        try:
            o1, o2 = int(self.o1_entry.get()), int(self.o2_entry.get())
            r = int(self.radius_entry.get())
            if r < 0:
                showerror(title="Error", message="Radius must be positive")
                return
        except ValueError:
            showerror(title="Error",
                      message="Incorrect input: input must be an integer")
            return
        self.lines.append('circle')
        self.circles.append([o1, o2, r])
        return o1, o2, r

    def draw_polyline(self, points, tag):
        """Рисует одну сплошную линию по списку логических точек."""
        if len(points) < 2:
            return
        coords = []
        for x, y in points:
            px = self.start_x + x * self.scale
            py = self.start_y + y * self.scale
            coords.extend([px, py])
        self.canvas.create_line(*coords, fill="black",
                                width=1, tags=["grid", tag])

    # --------- алгоритмы линий ---------

    def create_line(self, x1=0, y1=0, x2=0, y2=0, new=True):
        """Step-by-step, но рисуем сплошную линию."""
        try:
            if new:
                res = self.get_points(alg='simple')
                if not res:
                    return
                x1, y1, x2, y2 = res

            start_time = time.time()
            d_x = x2 - x1
            d_y = y2 - y1
            k = d_y / d_x if d_x != 0 else 0
            sign_d_x = int(np.sign(d_x))
            sign_d_y = int(np.sign(d_y))

            pts = [(x1, y1)]
            if d_x != 0:
                for i in range(1, abs(d_x) + 1):
                    x = x1 + sign_d_x * i
                    y = y1 + sign_d_x * int(k * i)
                    pts.append((x, y))
            else:
                for i in range(1, abs(d_y) + 1):
                    y = y1 + sign_d_y * i
                    pts.append((x1, y))

            self.draw_polyline(pts, "s")
            PixelPath.time = time.time() - start_time
        except TypeError:
            return

    def dda(self, x0=0, y0=0, x1=0, y1=0, new=True):
        """DDA – тоже сплошная линия."""
        try:
            if new:
                res = self.get_points(alg='dda')
                if not res:
                    return
                x0, y0, x1, y1 = res

            start_time = time.time()
            d_x, d_y = x1 - x0, y1 - y0
            k = d_y / d_x if d_x != 0 else 0
            sign_d_x, sign_d_y = int(np.sign(d_x)), int(np.sign(d_y))

            pts = []
            if abs(d_x) > abs(d_y):
                for i in range(abs(d_x) + 1):
                    x = x0 + i * sign_d_x
                    y = y0 + int(k * i * sign_d_x)
                    pts.append((x, y))
            else:
                for i in range(abs(d_y) + 1):
                    x = x0 + (int(i / k) * sign_d_y if k != 0 else 0)
                    y = y0 + i * sign_d_y
                    pts.append((x, y))

            self.draw_polyline(pts, "d")
            PixelPath.time = time.time() - start_time
        except TypeError:
            return

    def bresenham_line(self, x0=0, y0=0, x1=0, y1=0, new=True):
        """Bresenham – сплошная линия."""
        try:
            if new:
                res = self.get_points(alg="bres")
                if not res:
                    return
                x0, y0, x1, y1 = res

            start_time = time.time()

            d_x = abs(x1 - x0)
            d_y = abs(y1 - y0)
            sign_d_x = 1 if x1 > x0 else -1
            sign_d_y = 1 if y1 > y0 else -1
            error = d_x - d_y
            x, y = x0, y0

            pts = []
            while True:
                pts.append((x, y))
                if x == x1 and y == y1:
                    break
                error2 = 2 * error
                if error2 > -d_y:
                    error -= d_y
                    x += sign_d_x
                if error2 < d_x:
                    error += d_x
                    y += sign_d_y

            self.draw_polyline(pts, "b")
            PixelPath.time = time.time() - start_time
        except TypeError:
            return

    def castle_word(a, b):
        x, y = a - b, b
        m1 = "s"
        m2 = "d"
        while x != y:
            if x > y:
                x = x - y
                m2 = m1 + m2
            else:
                y = y - x
                m1 = m2 + m1
        return (m2 + m1) * x

    def castle_pitway(self, x0=0, y0=0, x1=0, y1=0, new=True):
        """Castle-Pitway, рисуем сплошную линию."""
        try:
            if new:
                res = self.get_points(alg='castle')
                if not res:
                    return
                x0, y0, x1, y1 = res

            start_time = time.time()

            dx = x1 - x0
            dy = y1 - y0

            if dx == 0 and dy == 0:
                return

            step_x = 1 if dx >= 0 else -1
            step_y = 1 if dy >= 0 else -1
            adx = abs(dx)
            ady = abs(dy)

            pts = []

            if ady == 0:
                x, y = x0, y0
                for _ in range(adx + 1):
                    pts.append((x, y))
                    x += step_x
            elif adx == 0:
                x, y = x0, y0
                for _ in range(ady + 1):
                    pts.append((x, y))
                    y += step_y
            else:
                def build_castle_word(a, b):
                    a = int(a)
                    b = int(b)
                    if b == 0:
                        return "s" * a
                    if a == b:
                        return "d" * a
                    x = a - b
                    y = b
                    m1 = "s"
                    m2 = "d"
                    while x != y:
                        if x > y:
                            x -= y
                            m2 = m1 + m2[::-1]
                        else:
                            y -= x
                            m1 = m2 + m1[::-1]
                    m = m2 + m1[::-1]
                    return m

                if adx >= ady:
                    a, b = adx, ady
                    swap_axes = False
                else:
                    a, b = ady, adx
                    swap_axes = True

                word = build_castle_word(a, b)

                x, y = x0, y0
                pts.append((x, y))
                for ch in word:
                    if not swap_axes:
                        if ch == "s":
                            x += step_x
                        else:
                            x += step_x
                            y += step_y
                    else:
                        if ch == "s":
                            y += step_y
                        else:
                            y += step_y
                            x += step_x
                    pts.append((x, y))

            self.draw_polyline(pts, "c")
            PixelPath.time = time.time() - start_time

        except TypeError:
            return

    # --------- окружность ---------

    def bresenham_circle(self, xc=0, yc=0, radius=0, new=True):
        try:
            if new:
                res = self.get_circle_coordinates()
                if not res:
                    return
                xc, yc, radius = res

            start_time = time.time()
            if radius == 0:
                px = self.start_x + xc * self.scale
                py = self.start_y - yc * self.scale
                self.canvas.create_oval(px, py, px + 1, py + 1,
                                        fill="black", width=1,
                                        tags=["grid", "cir"])
                return

            x = 0
            y = radius
            d = 3 - 2 * radius
            self.plot_circle_points(xc, yc, x, y)
            while y >= x:
                x += 1
                if d > 0:
                    y -= 1
                    d = d + 4 * (x - y) + 10
                else:
                    d = d + 4 * x + 6
                self.plot_circle_points(xc, yc, x, y)

            PixelPath.time = time.time() - start_time
        except TypeError:
            return

    def plot_circle_points(self, xc, yc, x, y):
        cx, cy = self.start_x, self.start_y
        s = self.scale

        def dot(X, Y):
            self.canvas.create_rectangle(X, Y, X + 1, Y + 1,
                                         outline="black", fill="black",
                                         tags=["grid", "cir"])

        dot(cx + (xc + x) * s, cy - (yc + y) * s)
        dot(cx + (xc - x) * s, cy - (yc + y) * s)
        dot(cx + (xc + x) * s, cy - (yc - y) * s)
        dot(cx + (xc - x) * s, cy - (yc - y) * s)
        dot(cx + (xc + y) * s, cy - (yc + x) * s)
        dot(cx + (xc - y) * s, cy - (yc + x) * s)
        dot(cx + (xc + y) * s, cy - (yc - x) * s)
        dot(cx + (xc - y) * s, cy - (yc - x) * s)


if __name__ == "__main__":
    app = PixelPath()
    app.mainloop()
