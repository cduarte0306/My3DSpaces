# Step 5: fix scale from user-entered wall measurements (ratio, or least-squares
# for multiple measurements). Flag >~10% disagreement as likely bad input/reconstruction.


def solve_scale(reconstructed_lengths: list[float], measured_lengths: list[float]) -> float:
    raise NotImplementedError
