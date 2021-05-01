from typing import Dict, List

from albumentations import BasicTransform, NoOp


class BaseTransformScheduler:
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.cur_transform(**kwargs)

    def step(self, **kwargs):
        pass


class TransformMultiStepScheduler(BaseTransformScheduler):
    """Selects matching transform once the number of epoch reaches
    one of the milestones.

    Args:
        transforms (list): Transforms to schedule.
        milestones (list): List of epoch indices.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # transform = A.NoOp()     if 0 <= epoch < 5
        >>> # transform = transform_1  if 5 <= epoch < 30
        >>> # transform = transform_2  if 30 <= epoch < 80
        >>> # transform = transform_3  if epoch >= 80
        >>>
        >>> scheduled_transform = TransformMultiStepScheduler(transforms=[transform_1, transform_2, transform_3],
        >>>                                                   milestones=[5, 30, 80])
        >>> train_dataset = Dataset(transform=scheduled_transform)
        >>> val_dataset = Dataset()
        >>>
        >>> for epoch in range(100):
        >>>     train(train_dataset)
        >>>     validate(val_dataset)
        >>>     scheduled_transform.step()
    """

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
        self.epoch_to_transform: Dict[int, BasicTransform] = {
            epoch_num: aug for epoch_num, aug in zip(milestones, transforms)
        }
        if 0 not in self.epoch_to_transform:
            self.epoch_to_transform[0] = NoOp()
        self._step = 0
        self.cur_transform: BasicTransform = self.epoch_to_transform[0]
        self.verbose: bool = verbose

    def step(self, **kwargs) -> None:
        self._step += 1
        if self._step in self.epoch_to_transform:
            self.cur_transform = self.epoch_to_transform[self._step]
            if self.verbose:
                print(f"Changing aug at epoch={self._step}")


class TransformSchedulerOnPlateau(BaseTransformScheduler):
    """Selects next transform when a metric has stopped improving.
    This scheduler reads a metrics quantity and if no improvement
    is seen for a 'patience' number of epochs, next transform in list is selected.

    Args:
        transforms (list): Transforms to schedule.
        patience (int): Number of epochs with no improvement after
            which next transform will be chosen (if there is). For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only switch transforms after the
            3rd epoch if the loss still hasn't improved then.
            Default: 5.
        mode (str): One of `min`, `max`. In `min` mode, transform
            will be switched when the quantity monitored has stopped
            decreasing; in `max` mode it will be switched when the
            quantity monitored has stopped increasing. Default: 'min'.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>>
        >>> scheduled_transform = TransformSchedulerOnPlateau(transforms=[transform_1, transform_2, transform_3],
        >>>                                                   mode="max",
        >>>                                                   plateau=10)
        >>> train_dataset = Dataset(transform=scheduled_transform)
        >>> val_dataset = Dataset()
        >>>
        >>> for epoch in range(100):
        >>>     train(dataset)
        >>>     val_score = validate(val_dataset)
        >>>     # Note that step should be called after validate()
        >>>     scheduled_transform.step(val_score)
    """

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
                print(f"Changing aug to transforms[{self._cur_transform_ind}]")
