from unittest import mock

import pytest

from albu_scheduler import TransformMultiStepScheduler, TransformSchedulerOnPlateau


class TestTransformStepScheduler:
    def test_ok(self, image):
        transforms = [mock.MagicMock() for _ in range(4)]

        scheduled_transform = TransformMultiStepScheduler(
            transforms=transforms, milestones=[0, 5, 10]
        )
        scheduled_transform(image=image)
        transforms[0].assert_called_with(image=image)

        for _ in range(5):
            scheduled_transform.step()

        scheduled_transform(image=image)
        transforms[1].assert_called_with(image=image)

        transforms[2].assert_not_called()
        transforms[3].assert_not_called()

    def test_no_zero_milestone(self):
        transforms = [mock.MagicMock() for _ in range(4)]

        scheduled_transform = TransformMultiStepScheduler(
            transforms=transforms, milestones=[5, 10]
        )
        assert scheduled_transform.cur_transform.__class__.__name__ == "NoOp"

    def test_too_much_milestones_fails(self):
        transforms = [mock.MagicMock()]
        milestones = [i for i in range(100)]
        with pytest.raises(ValueError):
            TransformMultiStepScheduler(transforms=transforms, milestones=milestones)


class TestTransformSchedulerOnPlateau:
    @pytest.mark.parametrize(
        "mode, metric_values",
        [("max", [1, 2, 3, 3, 3, 1]), ("min", [10, 9, 8, 8, 8, 100])],
    )
    def test_ok(self, image, mode, metric_values):
        transforms = [mock.MagicMock() for _ in range(4)]

        scheduled_transform = TransformSchedulerOnPlateau(
            transforms=transforms, mode=mode, patience=2
        )
        scheduled_transform(image=image)
        transforms[0].assert_called_with(image=image)

        for metric_value in metric_values[:-1]:
            scheduled_transform.step(metric_value)
            scheduled_transform(image=image)
            transforms[0].assert_called_with(image=image)
            transforms[0].reset_mock()

        scheduled_transform.step(metric_values[-1])
        scheduled_transform(image=image)
        transforms[1].assert_called_with(image=image)

        for _ in range(100):
            scheduled_transform(image=image)
        transforms[2].assert_not_called()
