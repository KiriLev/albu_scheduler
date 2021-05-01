# albu_scheduler
Scheduler for [albumentations](https://github.com/albumentations-team/albumentations) transforms based on [PyTorch schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) interface

# Usage
## TransformMultiStepScheduler
```python
import albumentations as A

from albu_scheduler import TransformMultiStepScheduler

transform_1 = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transform_2 = A.Compose([
    A.RandomCrop(width=128, height=128),
    A.VerticalFlip(p=0.5),
])

scheduled_transform = TransformMultiStepScheduler(transforms=[transform_1, transform_2], 
                                                  milestones=[0, 10])
dataset = Dataset(transform=scheduled_transform)

for epoch in range(100):
    train(...)
    validate(...)
    scheduled_transform.step()
```
## TransformSchedulerOnPlateau
```python
from albu_scheduler import TransformSchedulerOnPlateau

scheduled_transform = TransformSchedulerOnPlateau(transforms=[transform_1, transform_2], 
                                                  mode="max",
                                                  patience=5)

dataset = Dataset(transform=scheduled_transform)
for epoch in range(100):
    train(...)
    score = validate(...)
    scheduled_transform.step(score)
```

# Installation
```bash
git clone https://github.com/KiriLev/albu_scheduler
cd albu_scheduler
make install
```