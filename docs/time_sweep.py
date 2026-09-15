"""Render a trained 4D model from one fixed *test* camera while sweeping time 0 -> 1.

The built-in trajectories synthesise their own camera path and derive an up-axis from the
training poses, which rolls the view on D-NeRF scenes. Reusing a dataset camera verbatim
keeps the framing identical to the evaluation renders, so only the scene motion changes.

usage: time_sweep.py <training_output_dir> <test_view_index> <n_frames>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'scripts'))  # nerficg's scripts/utils.py sets up the source path
import utils
with utils.DiscoverSourcePath():
    import Framework
    from Datasets.utils import View
    from Implementations import Datasets as DI
    from Implementations import Methods as MI
    from Visual.Trajectories.utils import CameraTrajectory

base_dir, view_index, n_frames = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


class time_sweep(CameraTrajectory):
    """Holds a single test camera fixed and replays the full time range."""

    def __init__(self, reference: View, n_frames: int) -> None:
        super().__init__()
        self.reference = reference
        self.n_frames = n_frames

    def _generate(self, _default_camera, _reference_views) -> list[View]:
        reference = self.reference
        return [
            View(
                camera=reference.camera, camera_index=reference.camera_index,
                frame_idx=reference.frame_idx, global_frame_idx=reference.global_frame_idx,
                c2w=reference.c2w_numpy, timestamp=frame / (self.n_frames - 1), exif=reference.exif,
            )
            for frame in range(self.n_frames)
        ]


Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
model = MI.get_model(method=Framework.config.GLOBAL.METHOD_TYPE,
                     checkpoint=str(base_dir / 'checkpoints' / 'final.pt')).eval()
renderer = MI.get_renderer(method=Framework.config.GLOBAL.METHOD_TYPE, model=model)

trajectory = time_sweep(dataset.data['test'][view_index].to_simple(), n_frames)
trajectory.add_to_dataset(dataset, reference_set='test')
dataset.set_mode(trajectory.name)
renderer.render_subset(output_directory=base_dir / 'inference', dataset=dataset, verbose=True, save_gt=False)
