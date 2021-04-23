from typing import List

from albumentations import BasicTransform


class BaseTransformScheduler:
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.cur_transform(**kwargs)

    def step(self, **kwargs):
        pass


class TransformStepScheduler(BaseTransformScheduler):
    def __init__(
        self,
        transforms: List[BasicTransform],
        milestones: List[int],
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if len(milestones) > len(transforms):
            raise ValueError(
                "Length of milestones can't be greater than number of transforms"
            )
        self.transforms = {
            epoch_num: aug for epoch_num, aug in zip(milestones, transforms)
        }
        self._step = 0
        self.cur_transform = self.transforms[0]
        self.verbose = verbose

    def step(self, **kwargs) -> None:
        self._step += 1
        if self._step in self.transforms:
            self.cur_transform = self.transforms[self._step]
            if self.verbose:
                print(f"Changing aug to {self._step}")


class TransformSchedulerOnPlateau(BaseTransformScheduler):
    def __init__(
        self,
        transforms: List[BasicTransform],
        patience: int,
        mode: str = "min",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.patience = patience
        self.verbose = verbose

        self._step = 0
        self._cur_transform_ind = 0
        self.cur_transform = self.transforms[self._cur_transform_ind]

        self.best = 0.0 if self.mode == "max" else float("inf")
        self.num_bad_epochs = 0

    def is_better(self, left, right):
        if self.mode == "max":
            return left > right
        if self.mode == "min":
            return left < right

    def step(self, metric, **kwargs) -> None:
        current = float(metric)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if (
            self.num_bad_epochs > self.patience
            and self._cur_transform_ind < len(self.transforms) - 1
        ):
            self._cur_transform_ind += 1
            self.cur_transform = self.transforms[self._cur_transform_ind]
            self.num_bad_epochs = 0
            if self.verbose:
                print(f"Changing aug to {self._cur_transform_ind}")
