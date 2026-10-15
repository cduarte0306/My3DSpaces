# My3DSpaces

Turn a capture of a room into a 3D model you can design against.

There are two independent routes. Pick one and follow it end to end — neither
section depends on the other.

| | **Route A — LiDAR** | **Route B — Monocular video** |
|---|---|---|
| Phone | iPhone **Pro** (12 Pro or newer) | **Any** iPhone |
| Input | 3D Scanner App "All Data" export | A plain `.mov` from the Camera app |
| Camera poses | ARKit, recorded by the app | Solved by COLMAP |
| Geometry | The app's LiDAR mesh | Depth Anything V2 + TSDF fusion |
| Real-world scale | ARKit poses (already metric) | The floor + an assumed camera height |
| Quality | Clean and reliable | Rough — see [What to expect](#what-to-expect) |

Route A is the reference. Route B is the one that has to work on an ordinary
phone, and it is measured against Route A.

---

## Setup (both routes)

Prototyping runs in WSL2 (Ubuntu) on Windows. Install the NVIDIA driver on
Windows only — do **not** install a Linux driver inside WSL. Keep the project on
the Linux filesystem (`~/`), not `/mnt/c/...`, or IO will crawl.

```bash
python3 -m venv .venv && source .venv/bin/activate

# PyTorch first, matching your CUDA build (it brings its own CUDA runtime)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
git submodule update --init --recursive
```

Download the Depth Anything V2 **Small** checkpoint to exactly this path (note
the singular `checkpoint/`):

```
data/output/checkpoint/depth_anything_v2_vits.pth
```

Confirm the GPU is visible:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---
---

# Route A — LiDAR (iPhone Pro)

## A1. Capture the room

Every Pro iPhone since the 12 Pro has LiDAR, so a 14 Pro works. Non-Pro models
do not — use [Route B](#route-b--monocular-video-any-iphone).

1. Install **3D Scanner App** (Laan Labs) from the App Store. It is free.
2. Open it and choose a **LiDAR** capture mode. The sample data here was made
   with app version 2.5.4; menu labels move between releases, so look for the
   mode that builds a **mesh** rather than a bare point cloud.
3. Scan the room:
   - **Walk, don't pivot.** Turning on the spot gives no parallax.
   - Move slowly, keep consecutive views overlapping.
   - Hold the phone at normal standing height, ~1.5 m.
   - Turn the lights on — dim rooms give motion-blurred frames.
   - 30–60 seconds is plenty for one room.
4. Tap **Process** and wait for the mesh to build.

## A2. Export — the step that matters

Tap **Share / Export** and choose **All Data**.

This is the one place people go wrong. The geometry formats (OBJ, USDZ, PLY,
GLB…) contain only the mesh and **throw away the per-frame camera poses** this
project runs on.

AirDrop or iCloud the zip to your PC, unzip it, and copy the folder into
`data/lidar-data/` inside WSL. You should have:

```
data/lidar-data/<scan>/
├── frame_00000.jpg      # RGB, native landscape orientation
├── frame_00000.json     # that frame's ARKit pose + intrinsics
├── ...
├── export.obj           # LiDAR mesh, geometry only
├── textured_output.obj  # same mesh + texture atlas (.mtl, .jpg)
└── info.json            # metadata, incl. the user-drawn room box
```

If the `frame_*.json` files are there, you exported correctly.

## A3. Check the export loads

```bash
python3 -c "
from roommodel.lidar.scanner_app import ScannerAppScan
s = ScannerAppScan('data/lidar-data/<scan>')
w, h, d = s.room_dimensions()
print(f'{len(s)} frames, room {w:.2f} x {d:.2f} m, ceiling {h:.2f} m')
"
```

Expected on the sample scan:

```
144 frames, room 4.19 x 3.42 m, ceiling 2.53 m
```

Sanity-check those dimensions against the real room. They are your ground truth
for everything downstream.

## A4. Reconstruct a single frame

The smallest end-to-end step, and the best debugging tool: it predicts depth for
one frame, scales it against the LiDAR mesh, back-projects it into the room, and
draws it with the camera frustum and room box.

```bash
python3 scripts/reconstruct_single_frame.py data/lidar-data/<scan>
```

Useful variants:

```bash
# a different frame, with the room mesh overlaid
python3 scripts/reconstruct_single_frame.py data/lidar-data/<scan> --frame 60 --show-mesh

# the LiDAR mesh itself as a reference view (what we're aiming at)
python3 scripts/reconstruct_single_frame.py data/lidar-data/<scan> --source lidar

# save instead of opening a window
python3 scripts/reconstruct_single_frame.py data/lidar-data/<scan> \
    --save-ply out/frame0.ply --no-show
```

| Flag | Purpose |
|---|---|
| `--frame N` | Which frame to reconstruct |
| `--source mono\|lidar` | Predicted depth (default), or the LiDAR mesh |
| `--show-mesh` | Overlay the room mesh |
| `--mesh-color gray\|frames\|texture` | How to colour that overlay |
| `--stride N` | Keep every Nth pixel (default 3) |
| `--save-ply PATH`, `--no-show` | Write output / skip the window |

It prints the error against LiDAR — **3.4 cm median on frame 20** of the sample.

## A5. Build the full coloured room

Projects every frame onto the mesh and averages per-vertex colour, keeping only
views where a vertex is visible, front-facing and unoccluded.

```bash
python3 scripts/colorize_room_mesh.py data/lidar-data/<scan>
```

```bash
# fast preview (~5s)
python3 scripts/colorize_room_mesh.py data/lidar-data/<scan> --every 4

# the app's own texture atlas instead, for comparison
python3 scripts/colorize_room_mesh.py data/lidar-data/<scan> --source texture

# save the coloured mesh
python3 scripts/colorize_room_mesh.py data/lidar-data/<scan> \
    --save-mesh out/room.ply --no-show
```

| Flag | Purpose |
|---|---|
| `--source frames\|texture` | Project the frames (default), or load the app's atlas |
| `--every N` | Use every Nth frame — higher is faster, more gaps |
| `--image-scale F` | Sampling resolution (default 0.5) |
| `--show-box` | Draw the scan's room bounding box |
| `--save-mesh PATH`, `--no-show` | Write output / skip the window |

**98.4% of 642k vertices in ~18s** on the sample scan, visually on par with the
app's own atlas.

That is Route A complete — a coloured, metric 3D model of the room.

---
---

# Route B — Monocular video (any iPhone)

No depth sensor and no measuring tape. Everything is solved from pixels.

## B1. Record the video

Use the stock Camera app. The capture advice from A1 applies and matters *more*
here, because camera poses have to be solved from the images alone:

- **Walk, don't pivot** — without parallax COLMAP cannot triangulate at all.
- Move slowly with heavy overlap between views.
- Bright, even lighting. Blurry frames get discarded in B3.
- Textured surfaces help enormously. Bare painted walls are the hard case.
- 30–60 seconds.

Copy it to `data/raw/` inside WSL.

## B2. Extract sharp keyframes

Samples the video at ~2.5 fps and drops blurry frames (Laplacian variance).

```bash
python3 scripts/extract_frames.py data/raw/home-video.mov data/frames
```

```bash
# sample more densely
python3 scripts/extract_frames.py data/raw/home-video.mov data/frames --fps 4
```

On a 38.5 s test clip this kept **69 of ~96** frames.

## B3. Solve camera poses with COLMAP

Runs feature extraction, sequential matching and incremental mapping through
`pycolmap` — the COLMAP CLI is not needed.

```bash
python3 -m roommodel.monocular.colmap_pipeline
```

Reads `data/frames/`, writes `workspace/`. Took **~3 minutes** for 69 frames.

## B4. Check how many frames registered

**Do this before going further.** It is the single number that decides whether
the rest is worth running.

```bash
python3 -c "
from roommodel.monocular.colmap_frames import ColmapScan
s = ColmapScan('workspace/sparse', 'data/frames')
print(f'{len(s)} of {s.num_input_images} images registered ({s.registration_rate:.0%})')
"
```

```
46 of 69 images registered (67%)
```

Below roughly 70%, the reconstruction covers only part of the room and no
downstream setting will fix it — reshoot more slowly with more overlap.

To look at the recovered trajectory:

```bash
python3 scripts/inspect_reconstruction.py workspace/sparse
```

## B5. Build the room map

Predicts depth per frame, recovers scale, and fuses everything into one coloured
mesh.

```bash
python3 scripts/map_room_monocular.py --colmap workspace/sparse --images data/frames
```

COLMAP solves shape but never size, so scale is recovered from the floor plus an
assumed camera height. That is switched on automatically here. If you filmed
from an unusual height, say so:

```bash
python3 scripts/map_room_monocular.py --colmap workspace/sparse --images data/frames \
    --camera-height 1.6
```

```bash
# finer voxels, and save the result
python3 scripts/map_room_monocular.py --colmap workspace/sparse --images data/frames \
    --voxel-size 0.03 --save-mesh out/room.ply --no-show
```

| Flag | Purpose |
|---|---|
| `--colmap DIR --images DIR` | The COLMAP model, and the images it was run on |
| `--camera-height M` | Assumed camera height above the floor (default 1.5) |
| `--every N` | Frame stride (defaults to 1 for COLMAP) |
| `--voxel-size M` | TSDF resolution in metres (default 0.04) |
| `--depth-trunc M` | Ignore depth beyond this (default 6) |
| `--encoder vits\|vitb\|vitl` | Depth model size (needs the matching checkpoint) |
| `--save-mesh PATH`, `--no-show` | Write output / skip the window |

That is Route B complete. Expect a correctly shaped, correctly sized, visibly
rough model — see below.

---
---

## Benchmarking the monocular route against LiDAR

If you have a Route A scan, you can run the Route B *algorithm* on it and score
the result, because the LiDAR mesh gives ground truth. This is how the numbers
below were produced.

```bash
# monocular reconstruction using the scan's ARKit poses, scored against LiDAR
python3 scripts/map_room_monocular.py data/lidar-data/<scan> --compare-lidar
```

```bash
# same, but also discarding the metric poses and deriving size from the floor
python3 scripts/map_room_monocular.py data/lidar-data/<scan> \
    --scale-from floor --compare-lidar
```

No LiDAR is read unless `--compare-lidar` is passed.

## What to expect

Measured on the sample scan (iPhone 15 Pro, 144 frames, 4.19 × 3.42 m room):

| Method | Median error vs LiDAR |
|---|---|
| Single frame, scaled on LiDAR | 3.4 cm |
| Monocular map, metric poses | 6.0 cm |
| Monocular map, floor-derived scale | 7.9 cm |

Scale recovery, not depth, is the weak link. Feature triangulation lands at
12–20% error; the floor-height assumption gets within a few percent.

**Route B on a real video is the roughest by a wide margin.** On a 38.5 s
handheld clip only 67–71% of frames registered, and the fused mesh — while
correctly shaped and correctly sized — is blobby and holey. The plumbing works;
the output is not yet a usable room model. Slower capture with more overlap is
the single biggest improvement available.

## Layout

```
roommodel/
├── core/        # sensor-agnostic: image IO, frames protocol, back-projection,
│                #   ray-casting, vertex colouring, plane fitting, Open3D display
├── lidar/       # scanner_app.py — the only module needing a LiDAR mesh
└── monocular/   # RGB-only: depth, metric alignment, sparse anchors, gravity
                 #   scale, TSDF fusion, COLMAP adapter, frame extraction
```

`core/frames.py` defines the seam. Anything exposing `cam_to_world`,
`world_to_cam`, `intrinsics_for`, `load_image` and `position` can drive the whole
pipeline — which is how the same code runs on both ARKit and COLMAP poses.

## Gotchas worth knowing

- **iPhone frames carry an EXIF orientation tag.** `cv2.imread` silently rotates
  them, while ARKit intrinsics describe the *unrotated* sensor buffer. Always
  load through `roommodel/core/image_io.py`.
- **Photographic colour must be rendered unlit.** Open3D's default shading adds
  highlights on top of colour that already contains the room's real lighting, and
  bright surfaces blow out to white. `core/o3d_scene.show()` handles this.
- **Open3D's legacy `ScalableTSDFVolume` is broken** in 0.20 — it allocates a
  single volume unit. `monocular/fusion.py` uses the tensor `VoxelBlockGrid`
  instead, which also runs on the GPU.
- **More frames is not always better.** On a scan-app capture each extra frame
  brings another chance of a bad scale estimate, and `--every 3` beats
  `--every 2`. On COLMAP the opposite holds — use every frame.

Longer-form notes, including how each of these was diagnosed, live in
`.vscode/notes.md`.
