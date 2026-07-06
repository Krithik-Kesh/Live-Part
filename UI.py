"""
UI.py  Plant 3D Live Previewer

Open a Plant 3D part script and see the geometry it builds, live. Edit the
script in any editor and hit save -> the preview updates automatically. Or tweak
a parameter on the right and press Enter to see it move.

Run:  python UI.py
Needs: matplotlib (numpy comes with it). No Plant 3D, no AutoCAD required.
"""

import os
import traceback
import tkinter as tk
from tkinter import filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import runner
import render

POLL_MS = 400                       # how often check the file for changes

#  State 
state = {
    "path": None,                   # current script path
    "mtime": None,                  # last seen modification time
    "specs": [],                    # parameter specs from the script
    "entries": {},                  # name -> tk.Entry widget
    "xlim": None, "ylim": None, "zlim": None,   # remembered zoom box
}

#  Window 
window = tk.Tk()
window.title("Plant 3D Live Previewer")
window.geometry("1180x720")
window.configure(bg="#f0f0f0")

#  Toolbar 
toolbar = tk.Frame(window, bg="#222831", pady=8, padx=12)
toolbar.pack(fill="x")

tk.Label(toolbar, text="Plant 3D Live Previewer", bg="#222831", fg="white",
         font=("Segoe UI", 12, "bold")).pack(side="left")

status_var = tk.StringVar(value="Open a part script to begin.")
tk.Label(toolbar, textvariable=status_var, bg="#222831", fg="#bfe6ea",
         font=("Segoe UI", 9)).pack(side="right")

#  Body: viewport (left) + parameter panel (right)
body = tk.Frame(window, bg="#f0f0f0")
body.pack(fill="both", expand=True)

panel = tk.Frame(body, bg="#eaeaea", width=240)
panel.pack(side="right", fill="y")
panel.pack_propagate(False)

tk.Label(panel, text="Parameters", bg="#eaeaea", font=("Segoe UI", 10, "bold")
         ).pack(anchor="w", padx=12, pady=(12, 2))
tk.Label(panel, text="Type a value, press Enter.", bg="#eaeaea", fg="#666",
         font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(0, 8))
param_holder = tk.Frame(panel, bg="#eaeaea")
param_holder.pack(fill="x")

# 3D viewport 
fig = Figure(figsize=(8, 6), dpi=96)
ax = fig.add_subplot(111, projection="3d")
ax.text(0, 0, 0, "Open a script\nto see your part here",
        ha="center", va="center", fontsize=11, color="#aaaaaa")
ax.set_axis_off()
fig.patch.set_facecolor("#f0f0f0")

canvas = FigureCanvasTkAgg(fig, master=body)
canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=8, pady=8)
canvas.draw()


#  Scroll-wheel zoom 
def on_scroll(event):
    """Mouse wheel over the viewport zooms in/out around the current view centre."""
    if event.inaxes != ax:
        return
    factor = 0.88 if event.button == "up" else 1.0 / 0.88
    new_lims = []
    for get_lim in (ax.get_xlim3d, ax.get_ylim3d, ax.get_zlim3d):
        lo, hi = get_lim()
        centre = (lo + hi) / 2
        half = (hi - lo) / 2 * factor
        new_lims.append((centre - half, centre + half))
    ax.set_xlim3d(*new_lims[0])
    ax.set_ylim3d(*new_lims[1])
    ax.set_zlim3d(*new_lims[2])
    state["xlim"], state["ylim"], state["zlim"] = new_lims   # remember for next reload
    canvas.draw_idle()


canvas.mpl_connect("scroll_event", on_scroll)


#  Rendering 
def overrides_from_entries():
    """Read the parameter entry boxes into a dict, skipping blanks/bad input."""
    out = {}
    for name, entry in state["entries"].items():
        text = entry.get().strip()
        if not text:
            continue
        try:
            out[name] = float(text)
        except ValueError:
            pass
    return out


def render_now():
    """Run the current script and redraw. Errors show in the status bar."""
    path = state["path"]
    if not path:
        return
    # keep the current camera angle across redraws
    elev, azim = ax.elev, ax.azim
    try:
        scene, specs, used = runner.run_script(path, overrides_from_entries())
    except Exception as exc:                                  # noqa: BLE001
        traceback.print_exc()
        status_var.set("ERROR: " + str(exc).splitlines()[-1][:80])
        return

    # rebuild the parameter panel if the set of parameters changed
    if [s["name"] for s in specs] != [s["name"] for s in state["specs"]]:
        build_param_panel(specs, used)

    ax.set_axis_on()
    render.draw(ax, scene)
    ax.view_init(elev=elev, azim=azim)
    # Reapply a remembered zoom (set via scroll wheel) so editing a parameter
    # or auto-reloading on save doesn't snap back to full zoom-out.
    if state["xlim"] is not None:
        ax.set_xlim3d(*state["xlim"])
        ax.set_ylim3d(*state["ylim"])
        ax.set_zlim3d(*state["zlim"])
    canvas.draw()
    status_var.set("OK  -  {0}  -  {1} solids".format(
        os.path.basename(path), len(scene.render_solids())))


def recenter():
    """Drop the remembered zoom and let render.draw()'s autoscale fit everything again."""
    state["xlim"] = state["ylim"] = state["zlim"] = None
    render_now()


def build_param_panel(specs, values):
    """Create one labelled entry per numeric parameter."""
    for child in param_holder.winfo_children():
        child.destroy()
    state["entries"] = {}
    state["specs"] = specs
    for spec in specs:
        if not spec["numeric"]:
            continue
        row = tk.Frame(param_holder, bg="#eaeaea")
        row.pack(fill="x", padx=12, pady=3)
        label = spec["name"]
        tip = spec["tooltip"]
        tk.Label(row, text=label, width=5, anchor="w", bg="#eaeaea",
                 font=("Consolas", 10, "bold")).pack(side="left")
        entry = tk.Entry(row, width=9, font=("Consolas", 10))
        entry.insert(0, str(values.get(spec["name"], spec["default"])))
        entry.pack(side="left")
        entry.bind("<Return>", lambda e: render_now())
        entry.bind("<FocusOut>", lambda e: render_now())
        state["entries"][spec["name"]] = entry
        if tip:
            tk.Label(param_holder, text=tip, bg="#eaeaea", fg="#888",
                     font=("Segoe UI", 7), anchor="w", wraplength=210,
                     justify="left").pack(fill="x", padx=12, pady=(0, 2))


# File handling + live reload 
def open_script():
    path = filedialog.askopenfilename(
        title="Open Plant 3D part script",
        filetypes=[("Python part scripts", "*.py"), ("All files", "*.*")])
    if not path:
        return
    state["path"] = path
    state["mtime"] = None           # force first render
    state["specs"] = []             # force panel rebuild
    state["xlim"] = state["ylim"] = state["zlim"] = None   # fresh script -> fresh autoscale
    poll_file()


def poll_file():
    """Re-render whenever the script file's modification time changes."""
    path = state["path"]
    if path and os.path.exists(path):
        mtime = os.path.getmtime(path)
        if mtime != state["mtime"]:
            state["mtime"] = mtime
            render_now()
    window.after(POLL_MS, poll_file)


open_button = tk.Button(toolbar, text="Open Script", command=open_script,
                        bg="#00adb5", fg="white", font=("Segoe UI", 9),
                        relief="flat", padx=12, pady=4)
open_button.pack(side="left", padx=(20, 0))

recenter_button = tk.Button(toolbar, text="Recenter", command=recenter,
                            bg="#393e46", fg="white", font=("Segoe UI", 9),
                            relief="flat", padx=12, pady=4)
recenter_button.pack(side="left", padx=(8, 0))

# Start 
window.after(POLL_MS, poll_file)
window.mainloop()
