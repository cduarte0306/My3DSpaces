import argparse
import json, zipfile, liblzfse, numpy as np
import sys, time
from argparse import ArgumentParser
import cv2, os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Running this file by path puts roommodel/lidar on sys.path, not the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from roommodel.lidar.camera_nav import turn
from roommodel.lidar.palette_ui import PaletteWindow
from roommodel.lidar.paint import Painter, PALETTE, pixel_ray, adjust_color, to_hex
from roommodel.lidar.segmentation import segment_planes, color_by_label, save_segmentation, summarize

FLIP = np.diag([1, -1, -1, 1])               # OpenGL camera -> OpenCV camera
OUTPUT_PATH = "data/output/"
USE_NEW_VIEWER = False                       # set by --viewer new
SEGMENT = False                              # set by --segment

def preprocess(pcd, voxel):
    p = pcd.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return p, f

def register(new, model, voxel=0.05, min_overlap=0.30, max_tilt_deg=10.0):
    """Align `new` onto `model`. Returns (connected, T, fitness, rmse, tilt_deg)."""
    reg = o3d.pipelines.registration
    n, nf = preprocess(new, voxel)
    m, mf = preprocess(model, voxel)
    dist = voxel * 1.5
    # 1. global: no starting guess needed (the two scans share no frame)
    coarse = reg.registration_ransac_based_on_feature_matching(
        n, m, nf, mf, True, dist,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         reg.CorrespondenceCheckerBasedOnDistance(dist)],
        reg.RANSACConvergenceCriteria(1_000_000, 0.999))
    # 2. local: ICP polishes it
    fine = reg.registration_icp(n, m, voxel * 0.8, coarse.transformation,
                                reg.TransformationEstimationPointToPlane())
    T = fine.transformation                       # new -> model frame
    tilt = np.degrees(np.arccos(np.clip(T[1, 1], -1, 1)))  # how far y (up) tipped
    ok = fine.fitness >= min_overlap and tilt <= max_tilt_deg
    return ok, T, fine.fitness, fine.inlier_rmse, tilt

def show_mesh(mesh, step=0.15, alt=None, painter=None, out_prefix=""):
    """View a mesh in the legacy (OpenGL) viewer with WASD walking.

    `alt` is an optional second mesh (the segmented copy); press T to toggle.
    `painter` enables click-to-paint (Ctrl+click a surface; 1-8 pick the colour,
    [ ] hue, - = brightness, ; ' saturation, K colour dialog, U undo, R reset, V save). Legacy viewer only.

    The new viewer (--viewer new) needs Vulkan; on this WSL box that is Lavapipe,
    a CPU renderer, so it is painfully slow and is opt-in only.
    """
    if USE_NEW_VIEWER:
        try:
            o3d.visualization.draw([mesh] if alt is None else [{"name": "scan", "geometry": mesh},
                                                               {"name": "segments", "geometry": alt, "is_visible": False}],
                                   show_skybox=False)
            return
        except RuntimeError as e:
            print(f"New viewer unavailable ({e}); using the legacy viewer.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    title = "Room  |  W/S/A/D walk  Q/E up-down  Arrows turn  X/C speed  T scan/segments"
    if painter is not None:
        title += "  |  Ctrl+click paint  1-8 colour  U undo  R reset  V save  [ ] hue  - = bright  ; ' sat  K picker"
    if not vis.create_window(title):
        raise RuntimeError("Could not open a viewer window (no display / OpenGL context).")
    vis.add_geometry(mesh)
    vc = vis.get_view_control()
    vc.set_constant_z_near(0.01)                 # let the camera get right up to surfaces
    speed = [step]
    turn_deg = [3.0]                             # degrees per arrow-key press

    def move(fwd, right, up):
        def cb(v):
            vc.camera_local_translate(fwd * speed[0], right * speed[0], up * speed[0])
            return False
        return cb

    def look(yaw, pitch):
        def cb(v):
            cam = vc.convert_to_pinhole_camera_parameters()
            cam.extrinsic = turn(cam.extrinsic, yaw * turn_deg[0], pitch * turn_deg[0])
            vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            return False
        return cb

    def scale(f):
        def cb(v):
            speed[0] *= f
            return False
        return cb

    for key, args in {"W": (1, 0, 0), "S": (-1, 0, 0), "D": (0, 1, 0),
                      "A": (0, -1, 0), "Q": (0, 0, 1), "E": (0, 0, -1)}.items():
        vis.register_key_callback(ord(key), move(*args))
    showing = [mesh]
    # GLFW key codes: right 262, left 263, down 264, up 265. Turn on the spot, not orbit.
    for code, args in {262: (1, 0), 263: (-1, 0), 265: (0, 1), 264: (0, -1)}.items():
        vis.register_key_callback(code, look(*args))
    if alt is not None:

        def toggle(v):
            nxt = alt if showing[0] is mesh else mesh
            v.clear_geometries()
            v.add_geometry(nxt, reset_bounding_box=False)   # keep the camera where it is
            showing[0] = nxt
            return False
        vis.register_key_callback(ord("T"), toggle)
    vis.register_key_callback(ord("X"), scale(2.0))
    vis.register_key_callback(ord("C"), scale(0.5))

    if painter is not None:
        color = [np.array(PALETTE[0][1])]           # the active colour (RGB 0..1)
        label = ["white"]
        last = [None]                               # plane painted most recently; adjustments follow it
        mouse = {"pos": (0.0, 0.0), "down": None, "mods": 0}

        # The palette window (Tk). Optional: if it cannot open, the keyboard controls still work.
        ui = [None]
        try:
            ui[0] = PaletteWindow(PALETTE,
                                  on_color=lambda rgb, name: set_color(rgb, name),
                                  on_undo=lambda: undo(vis), on_reset=lambda: reset(vis),
                                  on_save=lambda: save(vis))
            ui[0].set_color(color[0], label[0])
            ui[0].update()
        except Exception as e:
            print(f"Palette window unavailable ({e}); using keyboard controls only.")

        def say(text):
            print(text)
            if ui[0] is not None:
                ui[0].set_status(text)

        def set_color(rgb, name):
            color[0], label[0] = np.clip(np.asarray(rgb, dtype=float), 0, 1), name
            if name != "custom":                    # a preset applies to the NEXT click, not the last wall
                last[0] = None
            if last[0] is not None:                 # live-update the surface you just painted
                painter.recolor(last[0], color[0])
                refresh(vis)
            if ui[0] is not None:
                ui[0].set_color(color[0], name)
            print(f"Colour: {name} {to_hex(color[0])}")

        def select(i):
            def cb(v):
                set_color(PALETTE[i][1], PALETTE[i][0])
                return False
            return cb
        for i in range(len(PALETTE)):
            vis.register_key_callback(ord(str(i + 1)), select(i))
        print(f"Colour: {label[0]}  (keys 1-{len(PALETTE)} presets; [ ] hue, - = brightness, ; ' saturation, K dialog)")

        def tweak(dh=0.0, ds=0.0, dv=0.0):
            def cb(v):
                set_color(adjust_color(color[0], dh, ds, dv), "custom")
                return False
            return cb
        for key, args in {"[": (-0.03, 0, 0), "]": (0.03, 0, 0), "-": (0, 0, -0.05),
                          "=": (0, 0, 0.05), ";": (0, -0.05, 0), "'": (0, 0.05, 0)}.items():
            vis.register_key_callback(ord(key), tweak(*args))

        def dialog(v):
            if ui[0] is not None:
                ui[0].choose()
                return False
            try:
                import tkinter
                from tkinter import colorchooser
                root = tkinter.Tk()
                root.withdraw()
                picked, _ = colorchooser.askcolor(color=to_hex(color[0]), title="Paint colour")
                root.destroy()
            except Exception as e:                  # no tkinter / no display for the dialog
                print(f"Colour dialog unavailable ({e}); use the keys instead.")
                return False
            if picked:
                set_color(np.array(picked) / 255.0, "custom")
            return False
        vis.register_key_callback(ord("K"), dialog)

        def refresh(v):
            if showing[0] is not mesh:              # painting acts on the scan view
                v.clear_geometries()
                v.add_geometry(mesh, reset_bounding_box=False)
                showing[0] = mesh
            v.update_geometry(mesh)
            v.update_renderer()

        def on_move(v, x, y):
            mouse["pos"] = (x, y)
            return False

        def on_button(v, button, action, mods):
            if button != 0:
                return False
            if action == 1:                         # press
                mouse["down"], mouse["mods"] = mouse["pos"], mods
                return False
            down, mouse["down"] = mouse["down"], None
            moved = down is None or np.hypot(mouse["pos"][0] - down[0], mouse["pos"][1] - down[1]) > 4
            if moved or not (mouse["mods"] & 2):    # Ctrl+click only; drags are camera moves
                return False
            cam = vc.convert_to_pinhole_camera_parameters()
            origin, direction = pixel_ray(cam, *mouse["pos"])
            pid = painter.pick(origin, direction)
            if pid is None:
                say("No recognised surface there (furniture or clutter is not painted).")
                return False
            painter.apply(pid, color[0])
            last[0] = pid
            say(f"Painted {painter.planes[pid].kind} #{pid} {label[0]} {to_hex(color[0])}")
            refresh(v)
            return False

        def undo(v):
            pid = painter.undo()
            last[0] = None
            say("Nothing to undo" if pid is None else f"Undid plane #{pid}")
            refresh(v)
            return False

        def reset(v):
            painter.reset()
            last[0] = None
            say("Reset all paint")
            refresh(v)
            return False

        def save(v):
            painter.save(out_prefix)
            say(f"Saved {out_prefix}room_painted.ply and paint.json")
            return False

        vis.register_mouse_move_callback(on_move)
        vis.register_mouse_button_callback(on_button)
        vis.register_key_callback(ord("U"), undo)
        vis.register_key_callback(ord("R"), reset)
        vis.register_key_callback(ord("V"), save)

        if ui[0] is not None:
            # vis.run() would block the palette window, so pump both ourselves
            while vis.poll_events():
                vis.update_renderer()
                if ui[0] is not None and not ui[0].update():
                    ui[0] = None                    # palette closed; keep going without it
                time.sleep(0.01)
            if ui[0] is not None:
                ui[0].close()
            vis.destroy_window()
            return
    vis.run()
    vis.destroy_window()


def present(mesh):
    """Optionally segment the mesh, save the results, then show it."""
    alt = painter = None
    if SEGMENT:
        planes, labels = segment_planes(mesh)
        print(summarize(planes, labels))
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        save_segmentation(OUTPUT_PATH, planes, labels)
        alt = color_by_label(mesh, planes, labels)
        painter = Painter(mesh, planes, labels)
        print("Painting: Ctrl+click a surface to recolour it (see the window title for keys).")
    show_mesh(mesh, alt=alt, painter=painter, out_prefix=OUTPUT_PATH)


def coarse_cloud(load, n_frames, s, dev, voxel=0.03, step=8):
    """Fuse a scan into a throwaway coarse grid and return its points.

    Registration does not need 5 mm detail, so this is far cheaper than the real
    fusion. Frames keep their own poses, so the cloud is in the scan's OWN frame.
    `load` is the per-frame loader from loadFromR3D.
    """
    F32 = o3d.core.Dtype.Float32
    tmp = o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight"), (F32, F32), ((1), (1)),
        voxel_size=voxel, block_resolution=16, block_count=50000, device=dev)
    for i in range(0, n_frames, step):
        d, _, K, c2w = load(i, s)
        D   = o3d.t.geometry.Image(o3d.core.Tensor(d, device=dev))
        Kt  = o3d.core.Tensor(K, o3d.core.Dtype.Float64)
        ext = o3d.core.Tensor(np.linalg.inv(c2w), o3d.core.Dtype.Float64)
        coords = tmp.compute_unique_block_coordinates(D, Kt, ext, 1.0, 4.0)
        tmp.integrate(coords, D, Kt, ext, 1.0, 4.0)
    return tmp.extract_point_cloud().to_legacy()

def loadFromR3D(file_path, vbg=None):
    z = zipfile.ZipFile(file_path)
    meta = json.loads(z.read("metadata").decode("utf-8"))
    w, h = meta["w"], meta["h"]
    dw, dh = meta["dw"], meta["dh"]
    s = dh / h  # scale factor for intrinsic coefficients

    def load(i, s):
        depth = np.frombuffer(liblzfse.decompress(z.read(f"rgbd/{i}.depth")),
                            np.float32).reshape(dh, dw).copy()
        conf = np.frombuffer(liblzfse.decompress(z.read(f"rgbd/{i}.conf")),
                            np.uint8).reshape(dh, dw)
        rgb = cv2.imdecode(np.frombuffer(z.read(f"rgbd/{i}.jpg"), np.uint8),
                        cv2.IMREAD_COLOR)
        # shrink RGB to the depth grid so pixels line up 1:1
        rgb = cv2.cvtColor(cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)
        depth[conf < 1] = 0                      # zero = "no measurement"
        fx, fy, cx, cy = np.array(meta["perFrameIntrinsicCoeffs"][i]) * s
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        p = meta["poses"][i]                     # qx qy qz qw tx ty tz
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_quat(p[:4]).as_matrix()
        c2w[:3, 3] = p[4:]
        return depth, rgb, K, c2w @ FLIP

    # Probe for GPU and use it if available, otherwise CPU
    dev = None
    if o3d.core.cuda.is_available():
        print("Using GPU")
        dev = o3d.core.Device("CUDA:0")
    else:
        print("Using CPU")
        dev = o3d.core.Device("CPU:0")

    # Open 3D to draw frame
    F32 = o3d.core.Dtype.Float32
    T = None  # new-scan -> model transform; stays None for the first scan (it defines the world frame)
    if vbg is None:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            ("tsdf", "weight", "color"), (F32, F32, F32), ((1), (1), (3)),
            voxel_size=0.006, block_resolution=20, block_count=120000,
            device=dev)
    else:
        vbg = vbg.to(dev)  # to() returns a new grid; it does not move in place
        # --- NEW: align this scan to the existing model before integrating ---
        new_cloud   = coarse_cloud(load, len(meta["poses"]), s, dev)  # new scan in its OWN frame
        model_cloud = o3d.geometry.PointCloud(vbg.extract_triangle_mesh().to_legacy().vertices)
        ok, T, fit, rmse, tilt = register(new_cloud, model_cloud)
        print(f"overlap fitness={fit:.2f} rmse={rmse*100:.1f}cm tilt={tilt:.1f}deg")
        if not ok:
            print("Scan does not connect to the model; not merging.")
            return              

    for i in range(0, len(meta["poses"]), 8):    # every 8th frame: 60 fps is redundant
        d, rgb, K, c2w = load(i, s)
        if T is not None:
            c2w = T @ c2w                        # move the new scan into the model's frame
        D   = o3d.t.geometry.Image(o3d.core.Tensor(d, device=dev))
        C   = o3d.t.geometry.Image(o3d.core.Tensor(rgb.astype(np.float32) / 255, device=dev))
        Kt  = o3d.core.Tensor(K, o3d.core.Dtype.Float64)
        ext = o3d.core.Tensor(np.linalg.inv(c2w), o3d.core.Dtype.Float64)  # world->cam
        coords = vbg.compute_unique_block_coordinates(D, Kt, ext, 1.0, 4.0)
        vbg.integrate(coords, D, C, Kt, ext, 1.0, 4.0)   # depth_scale=1 (already metres), trunc 4 m

    mesh = vbg.extract_triangle_mesh().to_legacy()
    if os.path.exists(OUTPUT_PATH) == False:
        os.makedirs(OUTPUT_PATH)

    # Save ply of the mesh and npz of the voxel grid
    o3d.io.write_triangle_mesh(OUTPUT_PATH + "room.ply", mesh)
    vbg.save(OUTPUT_PATH + "room.npz")
    present(mesh)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file",   type=str, required=True,       help="Path to .r3d or .ply file")
    parser.add_argument("--append", type=str, default="",          help="Path to a saved .npz voxel grid to grow instead of creating a new one")
    parser.add_argument("--path",   type=str, default=OUTPUT_PATH, help="Path to save the output files")
    parser.add_argument("--viewer", choices=["legacy", "new"], default="legacy", help="legacy = OpenGL with WASD walking (fast); new = Vulkan viewer with Fly mode (CPU-rendered on this WSL setup)")
    parser.add_argument("--segment", action="store_true", default=False, help="Find floor / ceiling / walls and colour them; press T in the viewer to toggle against the scan. Saves labels.npy + planes.json")
    args = parser.parse_args()
    SEGMENT = args.segment
    USE_NEW_VIEWER = args.viewer == "new"

    OUTPUT_PATH = args.path
    voxel_grid_path = args.append
    file_path = args.file
    vbg = None
    if voxel_grid_path:
        vbg = o3d.t.geometry.VoxelBlockGrid.load(voxel_grid_path)  # Load existing voxel grid and use it to grow the new one

    if "r3d" in file_path:
        loadFromR3D(file_path, vbg)
    elif "ply" in file_path:
        mesh = o3d.io.read_triangle_mesh(file_path)
        present(mesh)