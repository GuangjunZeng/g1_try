# G1 Robot Assets

This directory should contain the G1 humanoid robot URDF and mesh files.

## Required Files

```
g1/
├── urdf/
│   └── g1.urdf          # G1 robot URDF file
└── meshes/              # Visual and collision meshes
    ├── torso.stl
    ├── hip_*.stl
    ├── thigh_*.stl
    ├── calf_*.stl
    ├── foot_*.stl
    ├── shoulder_*.stl
    ├── arm_*.stl
    └── hand_*.stl
```

## Obtaining G1 Robot Model

The G1 robot URDF can be obtained from:

1. **Unitree Robotics Official**: Contact Unitree for G1 CAD/URDF files
2. **Open Source Repositories**: Check for community-shared G1 models
3. **Generate from CAD**: Convert G1 CAD files to URDF using tools like `phobos` or `sw2urdf`

## URDF Requirements

The URDF should include:

### Joints (20 DOF minimum for humanoid):

**Lower Body:**
- `left_hip_yaw`, `left_hip_roll`, `left_hip_pitch`
- `left_knee`
- `left_ankle_pitch`, `left_ankle_roll`
- `right_hip_yaw`, `right_hip_roll`, `right_hip_pitch`
- `right_knee`
- `right_ankle_pitch`, `right_ankle_roll`

**Upper Body:**
- `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`
- `left_elbow`
- `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`
- `right_elbow`

### Key Bodies:
- `torso` (base)
- `left_foot`, `right_foot` (end effectors)
- `left_hand`, `right_hand` (end effectors)

## Placeholder URDF

If you don't have the G1 URDF yet, you can:

1. Use a similar humanoid robot (e.g., Cassie, ATLAS) as a placeholder
2. Create a simplified kinematic model matching G1's structure
3. Wait to obtain official G1 model from Unitree

## Notes

- Ensure mass and inertia properties are realistic
- Set appropriate joint limits
- Include collision geometry for contact detection
- Add visual meshes for rendering (optional)

---

For support, contact: robotics@sailrolab.org

