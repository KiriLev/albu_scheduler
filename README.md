# albu_scheduler
Scheduler for albumentations transforms

# Usage
## TransformStepScheduler
```python
import albumentations as A

from albu_scheduler import TransformStepScheduler

transform_1 = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transform_2 = A.Compose([
    A.RandomCrop(width=128, height=128),
    A.VerticalFlip(p=0.5),
])

scheduled_transform = TransformStepScheduler(transforms=[transform_1, transform_2], 
                                             milestones=[0, 10])

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

for epoch in range(100):
    train(...)
    score = validate(...)
    scheduled_transform.step(score)

```




    

```
