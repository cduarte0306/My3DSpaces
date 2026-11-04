"""A small Tk palette window that sits beside the 3D viewer.

Open3D's legacy viewer cannot draw widgets, so the palette is its own window.
It never touches the mesh: it reports what the user chose through callbacks, and
`set_color` / `set_status` let the viewer's keyboard shortcuts update it back.
Drive it from the viewer's event loop with `update()`.
"""
import colorsys
import tkinter as tk
from tkinter import colorchooser


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(min(max(c, 0), 1) * 255)) for c in rgb)


def _from_hex(text):
    text = text.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ValueError(text)
    return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4))


class PaletteWindow:
    def __init__(self, palette, on_color, on_undo, on_reset, on_save, columns=4):
        self.on_color = on_color
        self._busy = False                      # set while we update our own widgets
        self._from_slider = False               # a slider drag is the source: don't write sliders back
        self.root = tk.Tk()
        self.root.title("Paint palette")
        self.root.resizable(False, False)
        pad = {"padx": 8, "pady": 4}

        tk.Label(self.root, text="Presets", anchor="w").grid(row=0, column=0, columnspan=columns, sticky="we", **pad)
        for i, (name, rgb) in enumerate(palette):
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            tk.Button(self.root, text=f"{i + 1}\n{name}", width=9, height=3, bg=_hex(rgb),
                      fg="black" if lum > 0.55 else "white", activebackground=_hex(rgb),
                      command=lambda n=name, c=rgb: self.on_color(c, n)
                      ).grid(row=1 + i // columns, column=i % columns, padx=3, pady=3)
        r = 1 + (len(palette) + columns - 1) // columns

        self.preview = tk.Label(self.root, text="", height=3, relief="sunken")
        self.preview.grid(row=r, column=0, columnspan=columns, sticky="we", **pad)

        self.sliders = {}
        for k, (label, hi) in enumerate([("Hue", 360), ("Saturation", 100), ("Brightness", 100)]):
            tk.Label(self.root, text=label, anchor="w").grid(row=r + 1 + k, column=0, sticky="w", padx=8)
            s = tk.Scale(self.root, from_=0, to=hi, orient="horizontal", length=260, showvalue=True)
            s.grid(row=r + 1 + k, column=1, columnspan=columns - 1, sticky="we", padx=8)
            # Not Scale(command=...): that also fires when we set() the slider from code, and
            # Tk delivers it later, after any "busy" flag is cleared. Bind real user input only.
            # after_idle so the value is read after Tk's own class binding has moved the slider.
            for ev in ("<B1-Motion>", "<ButtonRelease-1>", "<KeyRelease>"):
                s.bind(ev, lambda _e: self.root.after_idle(self._from_sliders))
            self.sliders[label] = s
        r += 4

        tk.Label(self.root, text="Hex", anchor="w").grid(row=r, column=0, sticky="w", padx=8)
        self.hex_var = tk.StringVar()
        entry = tk.Entry(self.root, textvariable=self.hex_var, width=10)
        entry.grid(row=r, column=1, sticky="w", padx=8)
        entry.bind("<Return>", lambda _e: self._from_hex())
        tk.Button(self.root, text="More colours…", command=self.choose).grid(row=r, column=2, columnspan=columns - 2, sticky="we", padx=8)
        r += 1

        bar = tk.Frame(self.root)
        bar.grid(row=r, column=0, columnspan=columns, sticky="we", **pad)
        for text, cmd in (("Undo (U)", on_undo), ("Reset (R)", on_reset), ("Save (V)", on_save)):
            tk.Button(bar, text=text, command=cmd).pack(side="left", expand=True, fill="x", padx=2)
        r += 1

        self.status = tk.Label(self.root, text="Ctrl+click a surface in the 3D window to paint it.",
                               anchor="w", justify="left", wraplength=340, fg="#444")
        self.status.grid(row=r, column=0, columnspan=columns, sticky="we", **pad)
        self.alive = True
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.geometry("+40+40")            # top-left, out of the way of the 3D window

    # -- called by the viewer ---------------------------------------------------------
    def set_color(self, rgb, name="custom"):
        """Show `rgb` as the active colour without firing the callback."""
        self._busy = True
        if not self._from_slider:               # rewriting them mid-drag loses the hue of greys
            h, s, v = colorsys.rgb_to_hsv(*rgb)
            self.sliders["Hue"].set(round(h * 360))
            self.sliders["Saturation"].set(round(s * 100))
            self.sliders["Brightness"].set(round(v * 100))
        self.hex_var.set(_hex(rgb))
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        self.preview.config(bg=_hex(rgb), fg="black" if lum > 0.55 else "white", text=f"{name}   {_hex(rgb)}")
        self._busy = False

    def set_status(self, text):
        self.status.config(text=text)

    def update(self):
        """Pump Tk events; returns False once the window has been closed."""
        if not self.alive:
            return False
        try:
            self.root.update()
        except tk.TclError:
            self.alive = False
        return self.alive

    def close(self):
        if self.alive:
            self.alive = False
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    # -- user input inside the window -------------------------------------------------
    def _from_sliders(self):
        if self._busy:
            return
        h, s, v = (self.sliders[k].get() for k in ("Hue", "Saturation", "Brightness"))
        self._from_slider = True
        try:
            self.on_color(colorsys.hsv_to_rgb(h / 360, s / 100, v / 100), "custom")
        finally:
            self._from_slider = False

    def _from_hex(self):
        try:
            rgb = _from_hex(self.hex_var.get())
        except ValueError:
            self.set_status(f"'{self.hex_var.get()}' is not a hex colour (try #c0a080).")
            return
        self.on_color(rgb, "custom")

    def choose(self):
        picked, _ = colorchooser.askcolor(color=self.hex_var.get() or "#ffffff", title="Paint colour", parent=self.root)
        if picked:
            self.on_color(tuple(c / 255 for c in picked), "custom")
