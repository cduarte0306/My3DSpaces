import argparse
import json, zipfile, liblzfse, numpy as np
from argparse import ArgumentParser
import cv2, os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

FLIP = np.diag([1, -1, -1, 1])               # OpenGL camera -> OpenCV camera
OUTPUT_PATH = "data/output/"
USE_NEW_VIEWER = False                       # set by --viewer new

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

def show_mesh(mesh, step=0.15):
    """View a mesh in the legacy (OpenGL) viewer with WASD walking.

    The new viewer (--viewer new) needs Vulkan; on this WSL box that is Lavapipe,
    a CPU renderer, so it is painfully slow and is opt-in only.
    """
    if USE_NEW_VIEWER:
        try:
            o3d.visualization.draw([mesh], show_skybox=False)
            return
        except RuntimeError as e:
            print(f"New viewer unavailable ({e}); using the legacy viewer.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window("Room  |  W/S forward-back  A/D left-right  Q/E up-down  X/C faster-slower"):
        raise RuntimeError("Could not open a viewer window (no display / OpenGL context).")
    vis.add_geometry(mesh)
    vc = vis.get_view_control()
    vc.set_constant_z_near(0.01)                 # let the camera get right up to surfaces
    speed = [step]

    def move(fwd, right, up):
        def cb(v):
            vc.camera_local_translate(fwd * speed[0], right * speed[0], up * speed[0])
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
    vis.register_key_callback(ord("X"), scale(2.0))
    vis.register_key_callback(ord("C"), scale(0.5))
    vis.run()
    vis.destroy_window()


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
            voxel_size=0.01, block_resolution=16, block_count=100000,
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
    show_mesh(mesh)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file",   type=str, required=True,       help="Path to .r3d or .ply file")
    parser.add_argument("--append", type=str, default="",          help="Path to a saved .npz voxel grid to grow instead of creating a new one")
    parser.add_argument("--path",   type=str, default=OUTPUT_PATH, help="Path to save the output files")
    parser.add_argument("--viewer", choices=["legacy", "new"], default="legacy", help="legacy = OpenGL with WASD walking (fast); new = Vulkan viewer with Fly mode (CPU-rendered on this WSL setup)")
    args = parser.parse_args()
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
        show_mesh(mesh)