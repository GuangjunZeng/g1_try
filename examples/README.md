# Examples

This directory contains example scripts demonstrating different aspects of the G1 retargeting system.

## Available Examples

### 1. Quick Start (`quickstart.py`)

Basic usage demonstration:
```bash
python examples/quickstart.py
```

Shows:
- Initializing the SMPLX retargeter
- Loading human motion data
- Retargeting to robot joint angles
- Computing retargeting losses

### 2. Custom Motion (`custom_motion.py`)

How to use custom human motion data:
```bash
python examples/custom_motion.py --motion_file=/path/to/motion.npz
```

### 3. Visualize Retargeting (`visualize.py`)

Visualize retargeting in Isaac Gym:
```bash
python examples/visualize.py
```

### 4. Batch Processing (`batch_retarget.py`)

Process multiple motions in batch:
```bash
python examples/batch_retarget.py --motion_dir=/path/to/motions/
```

## Creating Your Own Examples

Template structure for custom examples:

```python
#!/usr/bin/env python3

import torch
from g1_interaction.retargeting.smplx_retarget import SMPLXRetargeter
from g1_interaction import G1_INTERACTION_ROOT_DIR

def main():
    # Initialize retargeter
    retargeter = SMPLXRetargeter(
        smplx_model_path=f"{G1_INTERACTION_ROOT_DIR}/resources/smplx_models",
        robot_urdf_path=f"{G1_INTERACTION_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf",
        device='cuda'
    )
    
    # Load motion
    retargeter.load_motion_sequence('path/to/motion.npz')
    
    # Your custom code here
    # ...

if __name__ == '__main__':
    main()
```

## Tips

1. **Start with quickstart.py** to understand basic workflow
2. **Use smaller num_envs** when testing to reduce memory usage
3. **Enable visualization** during development (`--headless=False`)
4. **Check logs** in `logs/` directory for debugging

## Common Issues

- **SMPLX model not found**: Download from https://smpl-x.is.tue.mpg.de/
- **G1 URDF missing**: See `resources/robots/g1/README.md` for instructions
- **Out of memory**: Reduce number of environments or use CPU

For more help, see the main [README.md](../README.md).

