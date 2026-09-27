# Step 2: camera pose recovery via COLMAP/GLOMAP.
import os
import shutil

import pycolmap


def run_colmap(image_dir: str, workspace_dir: str) -> pycolmap.Reconstruction:
    os.makedirs(workspace_dir, exist_ok=True)

    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")

    # Start clean so re-runs (e.g. after re-extracting frames with a
    # different blur threshold) don't mix with a stale database/reconstruction.
    if os.path.exists(database_path):
        os.remove(database_path)
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir)

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    pycolmap.match_sequential(database_path=database_path)
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir,
    )

    if not reconstructions:
        raise RuntimeError("Incremental mapping failed to register any images")

    # incremental_mapping can split into multiple disconnected sub-models if
    # frames don't have enough overlap to link into one; keep the largest.
    recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())

    total_images = len(
        [f for f in os.listdir(image_dir) if not f.startswith(".")]
    )
    print(recon.summary())
    print(f"{recon.num_reg_images()} / {total_images} input images registered")

    recon.write(sparse_dir)
    recon.export_PLY(os.path.join(sparse_dir, "points.ply"))

    return recon


def main() -> None:
    run_colmap("data/frames", "workspace")


if __name__ == "__main__":
    main()
