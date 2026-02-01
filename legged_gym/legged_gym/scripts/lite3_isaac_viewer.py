import os
from pathlib import Path
import threading
import tkinter as tk
import numpy as np
from isaacgym import gymapi, gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR

KP = 30.0
KD = 1.0
DEFAULT_JOINT_ANGLES = {
    "FL_HipX_joint": 0.0,
    "HL_HipX_joint": 0.0,
    "FR_HipX_joint": 0.0,
    "HR_HipX_joint": 0.0,
    "FL_HipY_joint": -0.8,
    "HL_HipY_joint": -0.8,
    "FR_HipY_joint": -0.8,
    "HR_HipY_joint": -0.8,
    "FL_Knee_joint": 1.6,
    "HL_Knee_joint": 1.6,
    "FR_Knee_joint": 1.6,
    "HR_Knee_joint": 1.6,
}


def start_sliders(joint_names):
    """Create Tk sliders (one per joint) in a background thread."""
    slider_state = {"values": {name: 0.0 for name in joint_names}}

    def make_on_change(name):
        def on_change(val):
            try:
                slider_state["values"][name] = float(val)
            except ValueError:
                pass
        return on_change

    def slider_loop():
        root = tk.Tk()
        root.title("Lite3 Joint Offsets (rad)")
        for i, name in enumerate(joint_names):
            scale = tk.Scale(
                root,
                from_=-1.8,
                to=1.8,
                resolution=0.02,
                orient=tk.HORIZONTAL,
                length=260,
                label=name,
                command=make_on_change(name),
            )
            scale.set(0.0)
            scale.grid(row=i, column=0, padx=8, pady=4, sticky="ew")
        root.mainloop()

    t = threading.Thread(target=slider_loop, daemon=True)
    t.start()
    return slider_state


def main():
    """Simple Isaac Gym viewer for the Lite3 URDF."""
    args = gymutil.parse_arguments(description="Visualize Lite3 model in Isaac Gym")

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.friction_offset_threshold = 0.05
    sim_params.physx.friction_correlation_distance = 0.01

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    joint_names = list(DEFAULT_JOINT_ANGLES.keys())
    slider_state = start_sliders(joint_names)

    asset_root = Path(LEGGED_GYM_ROOT_DIR) / "resources" / "robots" / "lite3" / "urdf"
    asset_file = "Lite3.urdf"

    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = False
    asset_opts.disable_gravity = False
    asset_opts.collapse_fixed_joints = True
    asset_opts.use_mesh_materials = True
    asset_opts.armature = 0.01

    asset = gym.load_asset(sim, str(asset_root), asset_file, asset_opts)

    env_spacing = 5.0
    lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, lower, upper, 1)

    init_pose = gymapi.Transform()
    init_pose.p = gymapi.Vec3(0.0, 0.0, 0.35)
    init_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, init_pose, "lite3", 0, 1)

    dof_props = gym.get_asset_dof_properties(asset)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(KP)
    dof_props["damping"].fill(KD)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    dof_count = gym.get_asset_dof_count(asset)
    dof_dict = gym.get_asset_dof_dict(asset)
    dof_states = np.zeros(dof_count, dtype=gymapi.DofState.dtype)
    pos_targets_base = np.zeros(dof_count, dtype=np.float32)
    for name, angle in DEFAULT_JOINT_ANGLES.items():
        idx = dof_dict.get(name)
        if idx is not None:
            dof_states[idx]["pos"] = angle
            pos_targets_base[idx] = angle

    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
    pos_targets = pos_targets_base.copy()
    gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)

    cam_pos = gymapi.Vec3(2.0, 1.5, 1.2)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    while not gym.query_viewer_has_closed(viewer):
        pos_targets[:] = pos_targets_base
        for name, offset in slider_state.get("values", {}).items():
            idx = dof_dict.get(name)
            if idx is not None:
                pos_targets[idx] = pos_targets_base[idx] + offset
        gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
