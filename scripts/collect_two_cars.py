import os
import json
import queue
import time
from pathlib import Path

import numpy as np
import carla


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def lidar_to_bin(measurement: carla.LidarMeasurement, out_path: Path):
    arr = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    arr.tofile(str(out_path))


def radar_to_npy(measurement: carla.RadarMeasurement, out_path: Path):
    arr = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    np.save(str(out_path), arr)


def transform_to_dict(tf: carla.Transform):
    return {
        "location": {
            "x": tf.location.x,
            "y": tf.location.y,
            "z": tf.location.z
        },
        "rotation": {
            "pitch": tf.rotation.pitch,
            "yaw": tf.rotation.yaw,
            "roll": tf.rotation.roll
        }
    }


def actor_bbox_dict(actor: carla.Actor):
    tf = actor.get_transform()
    bb = actor.bounding_box
    return {
        "id": actor.id,
        "type_id": actor.type_id,
        "transform": transform_to_dict(tf),
        "bbox_extent": {
            "x": bb.extent.x,
            "y": bb.extent.y,
            "z": bb.extent.z
        },
        "bbox_location": {
            "x": bb.location.x,
            "y": bb.location.y,
            "z": bb.location.z
        }
    }


def get_weather_preset(world, mode: str):
    if mode == "fog":
        weather = carla.WeatherParameters(
            cloudiness=60.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=10.0,
            sun_altitude_angle=20.0,
            fog_density=60.0,
            fog_distance=10.0,
            fog_falloff=1.0,
            wetness=0.0
        )
    else:
        weather = carla.WeatherParameters.ClearNoon
    return weather


def build_lidar_bp(bp_lib, cfg):
    bp = bp_lib.find("sensor.lidar.ray_cast")
    bp.set_attribute("channels", str(cfg["channels"]))
    bp.set_attribute("range", str(cfg["range"]))
    bp.set_attribute("points_per_second", str(cfg["points_per_second"]))
    bp.set_attribute("rotation_frequency", str(cfg["rotation_frequency"]))
    bp.set_attribute("upper_fov", str(cfg["upper_fov"]))
    bp.set_attribute("lower_fov", str(cfg["lower_fov"]))
    bp.set_attribute("sensor_tick", str(cfg["sensor_tick"]))
    bp.set_attribute("noise_stddev", str(cfg["noise_stddev"]))
    return bp


def build_radar_bp(bp_lib, cfg):
    bp = bp_lib.find("sensor.other.radar")
    bp.set_attribute("horizontal_fov", str(cfg["horizontal_fov"]))
    bp.set_attribute("vertical_fov", str(cfg["vertical_fov"]))
    bp.set_attribute("range", str(cfg["range"]))
    bp.set_attribute("points_per_second", str(cfg["points_per_second"]))
    bp.set_attribute("sensor_tick", str(cfg["sensor_tick"]))
    return bp


def make_sensor_dirs(root: Path, ego_name: str):
    for sub in ["lidar", "radar_front", "radar_back", "pose", "calib", "labels", "camera_front"]:
        ensure_dir(root / ego_name / sub)


def write_calib(root: Path, ego_name: str, sensor_name: str, sensor_actor: carla.Actor):
    save_json(
        root / ego_name / "calib" / f"{sensor_name}.json",
        {
            "sensor_name": sensor_name,
            "type_id": sensor_actor.type_id,
            "transform": transform_to_dict(sensor_actor.get_transform())
        }
    )

def build_camera_bp(bp_lib):
   bp = bp_lib.find("sensor.camera.rgb")
   bp.set_attribute("image_size_x", "1280")
   bp.set_attribute("image_size_y", "720")
   bp.set_attribute("fov", "90")
   bp.set_attribute("sensor_tick", "0.0")
   return bp


def main():
    project_root = Path("/home/meet/projects/RAWAP_carla_l4dr/carla_l4dr")
    cfg_path = project_root / "configs" / "sim_config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    out_root = Path(cfg["output_root"])
    ensure_dir(out_root / "labels")

    client = carla.Client(cfg["host"], cfg["port"])
    client.set_timeout(20.0)
    world = client.get_world()

    if world.get_map().name.split("/")[-1] != cfg["town"]:
        world = client.load_world(cfg["town"])

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg["fixed_delta_seconds"]
    world.apply_settings(settings)

    world.set_weather(get_weather_preset(world, cfg["weather"]))

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    ego1_tf = spawn_points[cfg["vehicle_1_spawn_index"]]
    ego2_tf = spawn_points[cfg["vehicle_2_spawn_index"]]

    ego1 = world.spawn_actor(vehicle_bp, ego1_tf)
    ego2 = world.spawn_actor(vehicle_bp, ego2_tf)

    ego1.set_autopilot(True)
    ego2.set_autopilot(True)

    lidar_bp = build_lidar_bp(bp_lib, cfg["lidar"])
    radar_bp = build_radar_bp(bp_lib, cfg["radar"])
    camera_bp = build_camera_bp(bp_lib)

    lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.2))
    radar_front_tf = carla.Transform(carla.Location(x=2.2, y=0.0, z=0.8), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    radar_back_tf = carla.Transform(carla.Location(x=-2.2, y=0.0, z=0.8), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))
    camera_front_tf = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=1.6),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )


    lidar1 = world.spawn_actor(lidar_bp, lidar_tf, attach_to=ego1, attachment_type=carla.AttachmentType.Rigid)
    radar1f = world.spawn_actor(radar_bp, radar_front_tf, attach_to=ego1, attachment_type=carla.AttachmentType.Rigid)
    radar1b = world.spawn_actor(radar_bp, radar_back_tf, attach_to=ego1, attachment_type=carla.AttachmentType.Rigid)
    cam1 = world.spawn_actor(camera_bp, camera_front_tf, attach_to=ego1, attachment_type=carla.AttachmentType.Rigid)

    lidar2 = world.spawn_actor(lidar_bp, lidar_tf, attach_to=ego2, attachment_type=carla.AttachmentType.Rigid)
    radar2f = world.spawn_actor(radar_bp, radar_front_tf, attach_to=ego2, attachment_type=carla.AttachmentType.Rigid)
    radar2b = world.spawn_actor(radar_bp, radar_back_tf, attach_to=ego2, attachment_type=carla.AttachmentType.Rigid)
    cam2 = world.spawn_actor(camera_bp, camera_front_tf, attach_to=ego2, attachment_type=carla.AttachmentType.Rigid)


    make_sensor_dirs(out_root, "ego_1")
    make_sensor_dirs(out_root, "ego_2")

    write_calib(out_root, "ego_1", "lidar_top", lidar1)
    write_calib(out_root, "ego_1", "radar_front", radar1f)
    write_calib(out_root, "ego_1", "radar_back", radar1b)
    write_calib(out_root, "ego_1", "camera_front", cam1)
    write_calib(out_root, "ego_2", "lidar_top", lidar2)
    write_calib(out_root, "ego_2", "radar_front", radar2f)
    write_calib(out_root, "ego_2", "radar_back", radar2b)
    write_calib(out_root, "ego_1", "camera_front", cam2)

    q_lidar1, q_r1f, q_r1b = queue.Queue(), queue.Queue(), queue.Queue()
    q_lidar2, q_r2f, q_r2b = queue.Queue(), queue.Queue(), queue.Queue()
    q_cam1, q_cam2 = queue.Queue(), queue.Queue()


    lidar1.listen(q_lidar1.put)
    radar1f.listen(q_r1f.put)
    radar1b.listen(q_r1b.put)
    cam1.listen(q_cam1.put)
    lidar2.listen(q_lidar2.put)
    radar2f.listen(q_r2f.put)
    radar2b.listen(q_r2b.put)
    cam2.listen(q_cam2.put)

    actors_all = [ego1, ego2, lidar1, radar1f, radar1b, cam1, lidar2, radar2f, radar2b, cam2]
    frame_index_records = []

    try:
        for _ in range(cfg["num_frames"]):
            frame_id = world.tick()

            m_l1 = q_lidar1.get(timeout=5.0)
            m_r1f = q_r1f.get(timeout=5.0)
            m_r1b = q_r1b.get(timeout=5.0)
            m_l2 = q_lidar2.get(timeout=5.0)
            m_r2f = q_r2f.get(timeout=5.0)
            m_r2b = q_r2b.get(timeout=5.0)
            m_c1 = q_cam1.get(timeout=5.0)
            m_c2 = q_cam2.get(timeout=5.0)


            fid = f"{frame_id:06d}"

            lidar_to_bin(m_l1, out_root / "ego_1" / "lidar" / f"{fid}.bin")
            radar_to_npy(m_r1f, out_root / "ego_1" / "radar_front" / f"{fid}.npy")
            radar_to_npy(m_r1b, out_root / "ego_1" / "radar_back" / f"{fid}.npy")
            lidar_to_bin(m_l2, out_root / "ego_2" / "lidar" / f"{fid}.bin")
            radar_to_npy(m_r2f, out_root / "ego_2" / "radar_front" / f"{fid}.npy")
            radar_to_npy(m_r2b, out_root / "ego_2" / "radar_back" / f"{fid}.npy")
            m_c1.save_to_disk(str(out_root / "ego_1" / "camera_front" / f"{fid}.png"))
            m_c2.save_to_disk(str(out_root / "ego_2" / "camera_front" / f"{fid}.png"))


            save_json(out_root / "ego_1" / "pose" / f"{fid}.json", transform_to_dict(ego1.get_transform()))
            save_json(out_root / "ego_2" / "pose" / f"{fid}.json", transform_to_dict(ego2.get_transform()))

            nearby_actors = world.get_actors().filter("vehicle.*")
            labels = [actor_bbox_dict(a) for a in nearby_actors]
            save_json(out_root / "labels" / f"{fid}.json", {"frame": frame_id, "vehicles": labels})

            frame_index_records.append({
                "frame": frame_id,
                "timestamp": world.get_snapshot().timestamp.elapsed_seconds
            })

        save_json(out_root / "frames.json", {"frames": frame_index_records})

    finally:
        for actor in actors_all:
            if actor.is_alive:
                actor.destroy()

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
