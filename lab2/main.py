import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import gc



def get_histogram_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return np.zeros((3, 256), dtype=np.int64)

    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "Ожидается цветное изображение (BGR)"

    hist = np.zeros((3, 256), dtype=np.int64)
    for c in range(3):
        hist[c] = np.bincount(img_bgr[:, :, c].ravel(), minlength=256)

    return hist


def equalize_histogram_rgb(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.copy()
    h, w, ch = img.shape
    total_pixels = h * w

    hist = get_histogram_rgb(img)          # (3, 256)
    cdf = hist.cumsum(axis=1)              # (3, 256)

    # минимальное ненулевое значение CDF по каждому каналу
    cdf_min = np.zeros(3, dtype=np.int64)
    for c in range(3):
        nz = hist[c] > 0
        if np.any(nz):
            cdf_min[c] = cdf[c][nz].min()
        else:
            cdf_min[c] = 0

    lut = np.zeros((3, 256), dtype=np.uint8)
    for c in range(3):
        denom = total_pixels - cdf_min[c]
        if denom <= 0:
            continue
        diff = cdf[c] - cdf_min[c]
        diff = np.clip(diff, 0, None)
        equalized = diff.astype(np.float32) / float(denom) * 255.0
        lut[c] = equalized.astype(np.uint8)

    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = lut[c][img[:, :, c]]

    return result


def equalize_histogram_brightness(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # яркость (учёт порядка BGR)
    brightness = 0.114 * B + 0.587 * G + 0.299 * R
    bright_int = np.clip(brightness, 0, 255).astype(np.int32)

    h, w = bright_int.shape
    total_pixels = h * w

    hist = np.bincount(bright_int.ravel(), minlength=256)
    cdf = hist.cumsum()

    nz = hist > 0
    if np.any(nz):
        cdf_min = cdf[nz].min()
    else:
        cdf_min = 0

    denom = total_pixels - cdf_min
    if denom <= 0:
        return img_bgr.copy()

    brightness_map = np.zeros(256, dtype=np.float32)
    diff = cdf - cdf_min
    diff = np.clip(diff, 0, None)
    equalized = diff.astype(np.float32) / float(denom) * 255.0
    brightness_map[:] = equalized

    new_brightness = brightness_map[bright_int]

    scale = np.ones_like(brightness, dtype=np.float32)
    mask_pos = brightness > 0
    scale[mask_pos] = new_brightness[mask_pos] / brightness[mask_pos]

    R_new = np.clip(R * scale, 0, 255)
    G_new = np.clip(G * scale, 0, 255)
    B_new = np.clip(B * scale, 0, 255)

    mask_zero = brightness == 0
    R_new[mask_zero] = new_brightness[mask_zero]
    G_new[mask_zero] = new_brightness[mask_zero]
    B_new[mask_zero] = new_brightness[mask_zero]

    out = np.stack([B_new, G_new, R_new], axis=-1).astype(np.uint8)
    return out


def linear_contrast(img_bgr: np.ndarray, min_out: int = 0, max_out: int = 255) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    brightness = 0.114 * B + 0.587 * G + 0.299 * R
    bright_int = np.clip(brightness, 0, 255).astype(np.int32)

    min_brightness = int(bright_int.min())
    max_brightness = int(bright_int.max())

    if max_brightness == min_brightness:
        return img_bgr.copy()

    normalized = (bright_int - min_brightness).astype(np.float32) / float(
        max_brightness - min_brightness
    )

    new_brightness = min_out + normalized * (max_out - min_out)
    new_brightness = np.clip(new_brightness, 0, 255)

    scale = np.ones_like(brightness, dtype=np.float32)
    mask_pos = bright_int > 0
    scale[mask_pos] = new_brightness[mask_pos] / bright_int[mask_pos].astype(np.float32)

    R_new = np.clip(R * scale, 0, 255)
    G_new = np.clip(G * scale, 0, 255)
    B_new = np.clip(B * scale, 0, 255)

    out = np.stack([B_new, G_new, R_new], axis=-1).astype(np.uint8)
    return out


# ====== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ======

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processing Application")
        self.master.configure(background="#f0f0f0")

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # Левая панель с кнопками
        self.control_frame = Frame(master, background="#f0f0f0")
        self.control_frame.pack(side=LEFT, padx=5, pady=5)

        button_width = 32
        font_size = 9
        wrap = 220

        # === Кнопки базовой обработки ===
        self.load_button = Button(
            self.control_frame,
            text="Загрузить изображение",
            command=self.load_image,
            bg="#4CAF50",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.load_button.grid(row=0, column=0, padx=3, pady=3)

        self.contrast_button = Button(
            self.control_frame,
            text="Линейное контрастирование",
            command=self.linear_contrast_cmd,
            bg="#2196F3",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.contrast_button.grid(row=1, column=0, padx=3, pady=3)

        # Построение гистограммы (по яркости)
        self.hist_button = Button(
            self.control_frame,
            text="Показать гистограммы (R, G, B, яркость)",
            command=self.show_histograms_rgb_brightness,
            bg="#2196F3",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.hist_button.grid(row=2, column=0, padx=3, pady=3)


        # Эквализация гистограммы – по каналам BGR
        self.eq_rgb_button = Button(
            self.control_frame,
            text="Выравнивание гистограммы RGB (по каналам)",
            command=self.equalize_rgb_cmd,
            bg="#2196F3",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.eq_rgb_button.grid(row=3, column=0, padx=3, pady=3)

        # Эквализация гистограммы – по яркости
        self.eq_brightness_button = Button(
            self.control_frame,
            text="Выравнивание гистограммы по яркости",
            command=self.equalize_brightness_cmd,
            bg="#2196F3",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.eq_brightness_button.grid(row=4, column=0, padx=3, pady=3)

        # Простая и адаптивная пороговая обработка

        # Поля для поэлементных операций и JPEG-качества
        self.pixel_value_label_add_sub = Label(
            self.control_frame,
            text="Введите значение (сложение/вычитание):",
            bg="#f0f0f0",
            font=("Arial", font_size),
        )
        self.pixel_value_label_add_sub.grid(row=7, column=0, padx=3, pady=3)

        self.pixel_value_entry_add_sub = Entry(
            self.control_frame, font=("Arial", font_size), width=7
        )
        self.pixel_value_entry_add_sub.grid(row=8, column=0, padx=3, pady=3)

        self.pixel_value_label_mul_div = Label(
            self.control_frame,
            text="Введите значение (умножение/деление):",
            bg="#f0f0f0",
            font=("Arial", font_size),
        )
        self.pixel_value_label_mul_div.grid(row=9, column=0, padx=3, pady=3)

        self.pixel_value_entry_mul_div = Entry(
            self.control_frame, font=("Arial", font_size), width=7
        )
        self.pixel_value_entry_mul_div.grid(row=10, column=0, padx=3, pady=3)

        self.jpeg_quality_label = Label(
            self.control_frame,
            text="Качество JPEG (0-200):",
            bg="#f0f0f0",
            font=("Arial", font_size),
        )
        self.jpeg_quality_label.grid(row=11, column=0, padx=3, pady=3)

        self.jpeg_quality_entry = Entry(
            self.control_frame, font=("Arial", font_size), width=7
        )
        self.jpeg_quality_entry.grid(row=12, column=0, padx=3, pady=3)

        # Поэлементные операции
        self.add_button = Button(
            self.control_frame,
            text="Поэлементное сложение",
            command=self.elementwise_add,
            bg="#FFC107",
            fg="black",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.add_button.grid(row=13, column=0, padx=3, pady=3)

        self.subtract_button = Button(
            self.control_frame,
            text="Поэлементное вычитание",
            command=self.elementwise_subtract,
            bg="#FFC107",
            fg="black",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.subtract_button.grid(row=14, column=0, padx=3, pady=3)

        self.multiply_button = Button(
            self.control_frame,
            text="Поэлементное умножение",
            command=self.elementwise_multiply,
            bg="#FFC107",
            fg="black",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.multiply_button.grid(row=15, column=0, padx=3, pady=3)

        self.divide_button = Button(
            self.control_frame,
            text="Поэлементное деление",
            command=self.elementwise_divide,
            bg="#FFC107",
            fg="black",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.divide_button.grid(row=16, column=0, padx=3, pady=3)

        # Сжатие и сохранение
        self.jpeg_button = Button(
            self.control_frame,
            text="Сжать в JPEG",
            command=self.jpeg_compression,
            bg="#FF5722",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.jpeg_button.grid(row=17, column=0, padx=3, pady=3)

        self.save_uncompressed_button = Button(
            self.control_frame,
            text="Сохранить изображение",
            command=self.save_uncompressed_image,
            bg="#FF5722",
            fg="white",
            font=("Arial", font_size),
            width=button_width,
            wraplength=wrap,
            justify=CENTER,
        )
        self.save_uncompressed_button.grid(row=18, column=0, padx=3, pady=3)

        # Канвас для отображения изображения/гистограммы
        self.canvas = Canvas(
            master, width=600, height=600, bg="#ffffff", highlightbackground="#ffffff"
        )
        self.canvas.pack(side=RIGHT, padx=10, pady=10)

    # ===== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =====

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
                return
            self.processed_image = self.original_image.copy()
            self.show_image(self.original_image)

    def show_image(self, img):
        # поддержка как цветных, так и одноканальных изображений
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        # подгоняем под размер Canvas (макс 600x600 при сохранении пропорций)
        max_size = 600
        scale = min(max_size / h, max_size / w, 1.0)
        if scale != 1.0:
            img_rgb = cv2.resize(
                img_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(300, 300, image=tk_img)
        self.canvas.img = tk_img

    def _get_work_image(self):
        if self.processed_image is not None:
            return self.processed_image
        return self.original_image

    # ===== Методы: гистограмма и контраст =====

    def show_histograms_rgb_brightness(self):
        """
        Рисует 4 гистограммы:
        - B (синяя)
        - G (зелёная)
        - R (красная)
        - яркость (чёрная)
        и выводит их одной картинкой в основном окне.
        """
        img = self._get_work_image()
        if img is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        if img.ndim == 2:
            gray = img
            b = g = r = gray
        else:
            b, g, r = cv2.split(img)
            gray = np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)

        hist_h = 120
        hist_w = 256

        def make_hist_panel(channel, color_bgr, label_text):
            # channel — одноканальное изображение (uint8)
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
            panel = np.full((hist_h, hist_w, 3), 255, dtype=np.uint8)

            if hist.max() > 0:
                hist_norm = hist / hist.max() * (hist_h - 25)
            else:
                hist_norm = hist

            for x, value in enumerate(hist_norm):
                h_val = int(value)
                cv2.line(
                    panel,
                    (x, hist_h - 1),
                    (x, hist_h - 1 - h_val),
                    color_bgr,
                    1,
                )

            cv2.putText(
                panel,
                label_text,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

            return panel

        panel_b = make_hist_panel(b, (255, 0, 0), "B")
        panel_g = make_hist_panel(g, (0, 255, 0), "G")
        panel_r = make_hist_panel(r, (0, 0, 255), "R")
        panel_y = make_hist_panel(gray, (0, 0, 0), "Bright")

        # Склеиваем 4 панели вертикально
        combined = np.vstack([panel_b, panel_g, panel_r, panel_y])

        self.show_image(combined)


    def equalize_rgb_cmd(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return
        self.processed_image = equalize_histogram_rgb(self.original_image)
        self.show_image(self.processed_image)
        gc.collect()

    def equalize_brightness_cmd(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return
        self.processed_image = equalize_histogram_brightness(self.original_image)
        self.show_image(self.processed_image)
        gc.collect()

    def linear_contrast_cmd(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return
        # Можно сделать ввод min/max из полей, но в задании обычно 0–255
        self.processed_image = linear_contrast(self.original_image, 0, 255)
        self.show_image(self.processed_image)
        gc.collect()
        
    # ===== Чтение значений из полей =====

    def get_pixel_value_add_sub(self):
        try:
            return int(self.pixel_value_entry_add_sub.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное целое число")
            return None

    def get_pixel_value_mul_div(self):
        try:
            return int(self.pixel_value_entry_mul_div.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное целое число")
            return None

    def get_jpeg_quality(self):
        try:
            quality = int(self.jpeg_quality_entry.get())
            if 0 <= quality <= 200:
                return quality
            else:
                messagebox.showerror(
                    "Ошибка", "Качество должно быть в диапазоне от 0 до 200."
                )
                return None
        except ValueError:
            messagebox.showerror(
                "Ошибка", "Введите корректное целое число для качества."
            )
            return None

    # ===== Поэлементные операции =====

    def elementwise_add(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        value = self.get_pixel_value_add_sub()
        if value is None:
            return

        added_image = cv2.add(
            self.original_image,
            np.full(self.original_image.shape, value, dtype=np.uint8),
        )
        self.processed_image = added_image
        self.show_image(added_image)
        gc.collect()

    def elementwise_subtract(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        value = self.get_pixel_value_add_sub()
        if value is None:
            return

        subtracted_image = cv2.subtract(
            self.original_image,
            np.full(self.original_image.shape, value, dtype=np.uint8),
        )
        self.processed_image = subtracted_image
        self.show_image(subtracted_image)
        gc.collect()

    def elementwise_multiply(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        value = self.get_pixel_value_mul_div()
        if value is None:
            return

        multiplied_image = cv2.multiply(
            self.original_image,
            np.full(self.original_image.shape, value, dtype=np.uint8),
        )
        self.processed_image = multiplied_image
        self.show_image(multiplied_image)
        gc.collect()

    def elementwise_divide(self):
        if self.original_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение")
            return

        value = self.get_pixel_value_mul_div()
        if value is None:
            return

        if value == 0:
            messagebox.showerror("Ошибка", "Деление на ноль не допускается")
            return

        divided_image = cv2.divide(
            self.original_image,
            np.full(self.original_image.shape, value, dtype=np.uint8),
        )
        self.processed_image = divided_image
        self.show_image(divided_image)
        gc.collect()

    # ===== Сжатие и сохранение =====

    def jpeg_compression(self):
        if self.processed_image is None:
            messagebox.showerror(
                "Ошибка", "Сначала выполните обработку изображения"
            )
            return

        quality = self.get_jpeg_quality()
        if quality is None:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg;*.jpeg"),
                ("Все файлы", "*.*"),
            ],
        )
        if save_path:
            cv2.imwrite(
                save_path,
                self.processed_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            messagebox.showinfo(
                "Сжатие JPEG", "Изображение сжато и сохранено успешно."
            )
        gc.collect()

    def save_uncompressed_image(self):
        if self.processed_image is None:
            messagebox.showerror(
                "Ошибка", "Сначала выполните обработку изображения."
            )
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("Все файлы", "*.*"),
            ],
        )
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            messagebox.showinfo("Сохранение", "Изображение сохранено успешно.")
        gc.collect()


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
