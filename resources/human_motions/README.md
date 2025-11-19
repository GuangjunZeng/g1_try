# Human Motion Data

This directory contains SMPLX human motion sequences for retargeting to the G1 robot.

## Data Format

Motion files should be in `.npz` format with the following keys:

```python
{
    'body_pose': (T, 63),      # Body joint parameters (21 joints × 3)
    'global_orient': (T, 3),   # Root orientation (axis-angle)
    'transl': (T, 3),          # Root translation (x, y, z)
    'betas': (10,),            # Shape parameters (optional)
    'fps': 30,                 # Frames per second
}
```

Where `T` is the number of frames.

## Obtaining Motion Data

### 1. From Motion Capture

Convert mocap data to SMPLX format:

```python
import smplx
import numpy as np

# Fit SMPLX to mocap markers
smplx_model = smplx.create('path/to/smplx', model_type='smplx')

# ... perform fitting ...

# Save to npz
np.savez('motion.npz',
    body_pose=body_pose_seq,
    global_orient=global_orient_seq,
    transl=transl_seq,
    betas=betas,
    fps=30
)
```

### 2. From Existing Datasets

- **AMASS**: Large-scale human motion database in SMPLX format
  - Download from: https://amass.is.tue.mpg.de/
  
- **GRAB**: Whole-body human grasping motions
  - Download from: https://grab.is.tue.mpg.de/

- **EgoBody**: Egocentric human body motion
  - Download from: https://sanweiliti.github.io/egobody/egobody.html

### 3. From Video

Use video-based human pose estimation:

- **VIBE**: Video inference for body pose
- **PIXIE**: Expressive pose and shape from video
- **PyMAF**: Multi-hypothesis 3D human mesh from video

## Example Motion Sequences

Sample motions to get started:

1. **Walking**: Basic locomotion pattern
2. **Reaching**: Upper body interaction with objects
3. **Sitting/Standing**: Whole-body motion with contact changes
4. **Dancing**: Complex full-body coordination

## Preprocessing

Before using motion sequences:

1. **Smooth trajectories**: Remove noise from pose estimates
2. **Resample to target FPS**: Match simulation timestep
3. **Center and orient**: Align starting pose
4. **Check joint limits**: Ensure poses are physically feasible

Example preprocessing:

```python
import numpy as np
from scipy.signal import savgol_filter

# Load motion
data = np.load('raw_motion.npz')
body_pose = data['body_pose']

# Smooth with Savitzky-Golay filter
body_pose_smooth = savgol_filter(body_pose, 
                                  window_length=11, 
                                  polyorder=3, 
                                  axis=0)

# Resample
from scipy.interpolate import interp1d
t_old = np.linspace(0, 1, len(body_pose_smooth))
t_new = np.linspace(0, 1, int(len(body_pose_smooth) * 60/30))  # 30->60 fps
interp = interp1d(t_old, body_pose_smooth, axis=0)
body_pose_resampled = interp(t_new)

# Save processed
np.savez('processed_motion.npz',
    body_pose=body_pose_resampled,
    global_orient=data['global_orient'],
    transl=data['transl'],
    fps=60
)
```

## Directory Structure

Organize motions by category:

```
human_motions/
├── locomotion/
│   ├── walk_forward.npz
│   ├── walk_backward.npz
│   ├── run.npz
│   └── turn.npz
├── manipulation/
│   ├── reach_high.npz
│   ├── pick_up.npz
│   └── push.npz
├── interaction/
│   ├── sit_down.npz
│   ├── stand_up.npz
│   └── climb.npz
└── complex/
    ├── dance.npz
    └── martial_arts.npz
```

## Quality Checklist

Before training with a motion:

- [ ] Motion is smooth (no sudden jumps)
- [ ] Frame rate is appropriate (30-60 fps)
- [ ] Poses are within SMPLX valid range
- [ ] Contact information is available or estimatable
- [ ] Duration is sufficient (>100 frames recommended)
- [ ] Motion exhibits desired behavior for robot

## Contact Information

For motion data questions or sharing cleaned datasets:
- Email: data@sailrolab.org
- GitHub Issues: Report dataset problems

---

**Note**: When using publicly available datasets, please comply with their respective licenses and citation requirements.

