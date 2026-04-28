import sys
from pathlib import Path

# Add the project root to the python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import os
import json
import queue
import time

import numpy as np
import carla
import configs

def transform_to_matrix(transform):
    loc = transform.location
    rot = transform.rotation

    pitch = np.deg2rad(rot.pitch)
    yaw   = np.deg2rad(rot.yaw)
    roll  = np.deg2rad(rot.roll)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [loc.x, loc.y, loc.z]

    return T


def build_camera_calib(sensor):
    attrs = sensor.attributes

    width = int(attrs.get("image_size_x"))
    height = int(attrs.get("image_size_y"))
    fov = float(attrs.get("fov"))

    f = width / (2 * np.tan(np.deg2rad(fov / 2)))

    K = [
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ]

    return {
        "width": width,
        "height": height,
        "fov": fov,
        "K": K,
        "T_world_sensor": transform_to_matrix(sensor.get_transform()).tolist()
    }


def build_sensor_calib(sensor):
    return {
        "T_world_sensor": transform_to_matrix(sensor.get_transform()).tolist()
    }

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def radar_to_ply(measurement, out_path):
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)

    # [velocity, azimuth, altitude, depth]
    v = raw[:, 0]
    az = raw[:, 1]
    alt = raw[:, 2]
    d = raw[:, 3]

    # convert to XYZ
    x = d * np.cos(alt) * np.cos(az)
    y = d * np.cos(alt) * np.sin(az)
    z = d * np.sin(alt)

    pts = np.stack([x, y, z], axis=1)

    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for px, py, pz in pts:
            f.write(f"{px} {py} {pz}\n")


def radar_to_npy(measurement: carla.RadarMeasurement, out_path: Path):
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)

    # CARLA radar format: [velocity, azimuth, altitude, depth]
    arr = {
        "velocity": raw[:, 0],
        "azimuth": raw[:, 1],
        "altitude": raw[:, 2],
        "depth": raw[:, 3],
    }

    np.save(str(out_path), arr)


def transform_to_dict(tf: carla.Transform):
    return {
        "location": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z},
        "rotation": {
            "pitch": tf.rotation.pitch,
            "yaw": tf.rotation.yaw,
            "roll": tf.rotation.roll,
        },
    }


def actor_bbox_dict(actor: carla.Actor):
    tf = actor.get_transform()
    bb = actor.bounding_box
    return {
        "id": actor.id,
        "type_id": actor.type_id,
        "transform": transform_to_dict(tf),
        "bbox_extent": {"x": bb.extent.x, "y": bb.extent.y, "z": bb.extent.z},
        "bbox_location": {"x": bb.location.x, "y": bb.location.y, "z": bb.location.z},
    }


def get_weather(cfg_weather):
    if cfg_weather["mode"] == "custom":
        p = cfg_weather["params"]
        return carla.WeatherParameters(
            cloudiness=p["cloudiness"],
            precipitation=p["precipitation"],
            precipitation_deposits=p["precipitation_deposits"],
            wind_intensity=p["wind_intensity"],
            sun_altitude_angle=p["sun_altitude_angle"],
            fog_density=p["fog_density"],
            fog_distance=p["fog_distance"],
            fog_falloff=p["fog_falloff"],
            wetness=p["wetness"],
        )
    return carla.WeatherParameters.ClearNoon


def build_radar_bp(bp_lib, cfg):
    bp = bp_lib.find("sensor.other.radar")
    bp.set_attribute("horizontal_fov", str(cfg["horizontal_fov"]))
    bp.set_attribute("vertical_fov", str(cfg["vertical_fov"]))
    bp.set_attribute("range", str(cfg["range"]))
    bp.set_attribute("points_per_second", str(cfg["points_per_second"]))
    bp.set_attribute("sensor_tick", str(cfg["sensor_tick"]))
    return bp


def make_sensor_dirs(root: Path, ego_name: str):
    for sub in [
        "radar_front",
        "radar_back",
        "pose",
        "calib",
        "labels",
        "camera_front",
    ]:
        ensure_dir(root / ego_name / sub)


def write_calib(root: Path, ego_name: str, sensor_name: str, sensor_actor: carla.Actor, extraData: dict = None):
    save_json(
        root / ego_name / "calib" / f"{sensor_name}.json",
        {
            "sensor_name": sensor_name,
            "type_id": sensor_actor.type_id,
            # "transform": transform_to_dict(sensor_actor.get_transform()),
            **(extraData if extraData is not None else {})
        },
    )


def build_camera_bp(bp_lib):
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "1280")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "90")
    bp.set_attribute("sensor_tick", "0.0")
    return bp


def main():
    project_root = Path(configs.project_config["project_root"])
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

    world.set_weather(get_weather(cfg["weather"]))

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    ego_tf = spawn_points[cfg["vehicle_1_spawn_index"]]

    ego = world.spawn_actor(vehicle_bp, ego_tf)

    ego.set_autopilot(True)

    radar_bp = build_radar_bp(bp_lib, cfg["radar"])
    camera_bp = build_camera_bp(bp_lib)

    radar_transform_fl = carla.Transform(
        carla.Location(x=1.5, y=-0.8, z=1.0),
        carla.Rotation(yaw=-45)
    )

    radar_transform_fr = carla.Transform(
        carla.Location(x=1.5, y=0.8, z=1.0),
        carla.Rotation(yaw=45)
    )

    radar_transform_bl = carla.Transform(
        carla.Location(x=-1.5, y=-0.8, z=1.0),
        carla.Rotation(yaw=-135)
    )

    radar_transform_br = carla.Transform(
        carla.Location(x=-1.5, y=0.8, z=1.0),
        carla.Rotation(yaw=135)
    )

    camera_front_tf = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=1.6),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    radar1fl = world.spawn_actor(radar_bp, radar_transform_fl, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)

    radar1fr = world.spawn_actor(radar_bp, radar_transform_fr, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)

    radar1bl = world.spawn_actor(radar_bp, radar_transform_bl, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)

    radar1br = world.spawn_actor(radar_bp, radar_transform_br, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)

    cam1 = world.spawn_actor(
        camera_bp,
        camera_front_tf,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )

    make_sensor_dirs(out_root, "ego")

    ensure_dir(out_root / "previews" / "ego" / "radar")

    write_calib(out_root, "ego", "radar_front_left", radar1fl, extraData=build_sensor_calib(radar1fl))
    write_calib(out_root, "ego", "radar_front_right", radar1fr, extraData=build_sensor_calib(radar1fr))
    write_calib(out_root, "ego", "radar_back_left", radar1bl, extraData=build_sensor_calib(radar1bl))
    write_calib(out_root, "ego", "radar_back_right", radar1br, extraData=build_sensor_calib(radar1br))
    write_calib(out_root, "ego", "camera_front", cam1, extraData=build_camera_calib(cam1))

    q_r1fl, q_r1fr, q_r1bl, q_r1br = queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()
    q_cam1 = queue.Queue()

    radar1fl.listen(q_r1fl.put)
    radar1fr.listen(q_r1fr.put)
    radar1bl.listen(q_r1bl.put)
    radar1br.listen(q_r1br.put)
    cam1.listen(q_cam1.put)

    actors_all = [
        ego,
        radar1fl,
        radar1fr,
        radar1bl,
        radar1br,
        cam1,
    ]
    frame_index_records = []

    for tl in world.get_actors().filter('*traffic_light*'):
        tl.set_green_time(1.0)
        tl.set_red_time(1.0)
        tl.set_yellow_time(0.5)
    
    tm = client.get_trafficmanager()
    tm.ignore_lights_percentage(ego, 50.0)  # partial ignore
    try:
        for _ in range(cfg["num_frames"]):
            frame_id = world.tick()

            m_r1fl = q_r1fl.get(timeout=5.0)
            m_r1fr = q_r1fr.get(timeout=5.0)
            m_r1bl = q_r1bl.get(timeout=5.0)
            m_r1br = q_r1br.get(timeout=5.0)
            m_c1 = q_cam1.get(timeout=5.0)

            fid = f"{frame_id:06d}"

            radar_to_npy(m_r1fl, out_root / "ego" / "radar_front" / f"left_{fid}.npy")
            radar_to_npy(m_r1fr, out_root / "ego" / "radar_front" / f"right_{fid}.npy")
            radar_to_npy(m_r1bl, out_root / "ego" / "radar_back" / f"left_{fid}.npy")
            radar_to_npy(m_r1br, out_root / "ego" / "radar_back" / f"right_{fid}.npy")
            m_c1.save_to_disk(str(out_root / "ego" / "camera_front" / f"{fid}.png"))

            radar_to_ply(m_r1fl, out_root / "previews" / "ego" / "radar" / f"front_left_{fid}.ply")
            radar_to_ply(m_r1fr, out_root / "previews" / "ego" / "radar" / f"front_right_{fid}.ply")
            radar_to_ply(m_r1bl, out_root / "previews" / "ego" / "radar" / f"back_left_{fid}.ply")
            radar_to_ply(m_r1br, out_root / "previews" / "ego" / "radar" / f"back_right_{fid}.ply")

            save_json(
                out_root / "ego" / "pose" / f"{fid}.json",
                {
                    # "transform": transform_to_dict(ego.get_transform()),
                    "T_world_ego": transform_to_matrix(ego.get_transform()).tolist()
                },
            )

            nearby_actors = world.get_actors().filter("vehicle.*")
            labels = [actor_bbox_dict(a) for a in nearby_actors]
            weather = world.get_weather()

            weather_dict = {
                "cloudiness": weather.cloudiness,
                "precipitation": weather.precipitation,
                "precipitation_deposits": weather.precipitation_deposits,
                "wind_intensity": weather.wind_intensity,
                "fog_density": weather.fog_density,
                "fog_distance": weather.fog_distance,
                "fog_falloff": weather.fog_falloff,
                "sun_azimuth_angle": weather.sun_azimuth_angle,
                "sun_altitude_angle": weather.sun_altitude_angle,
            }

            ego_pose = transform_to_matrix(ego.get_transform())

            save_json(
                out_root / "labels" / f"{fid}.json",
                {
                    "frame": frame_id,
                    "timestamp": world.get_snapshot().timestamp.elapsed_seconds,

                    # global ground truth
                    "weather": weather_dict,
                    "vehicles": labels,

                    # ego-specific poses (CRITICAL)
                    "ego": {
                        "T_world_ego": ego_pose.tolist()
                    }
                },
            )

            frame_index_records.append(
                {
                    "frame": frame_id,
                    "timestamp": world.get_snapshot().timestamp.elapsed_seconds,
                }
            )

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
