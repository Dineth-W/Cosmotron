import numpy as np
import cv2
from pupil_apriltags import Detector
from controller import Robot, Camera, Motor, RangeFinder
import math
import sys

# GLOBAL CONSTANTS
TIME_STEP = 640
VELOCITY = 800             # logical speed, will be clamped to motor max
MAX_SPEED = 2000.0         # logical max for turns, also clamped
TAG_FAMILY = "tag25h9"
TAG_SIZE = 0.20            # meters
TARGET_TAG_IDS = [0, 1, 2]
WAYPOINT_TOLERANCE = 0.08  # meters
OBSTACLE_DIST = 0.25       # meters for a collision detection (front)
SAFE_BACKUP_DIST = 0.5     # distance to move back (in meters) to go around
OBSTACLE_TIMEOUT = 30      # cycles before giving up on obstacle avoidance

# MOTOR NAMES
JOINT_NAMES = [
    "BackLeftBogie", "FrontLeftBogie", "FrontLeftArm", "BackLeftArm",
    "FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel",
    "BackRightBogie", "FrontRightBogie", "FrontRightArm", "BackRightArm",
    "FrontRightWheel", "MiddleRightWheel", "BackRightWheel"
]

joints = {}
MAX_WHEEL_VELOCITY = 1.0   # will be overwritten from motors


def move_wheels(left_speed, right_speed):
    """
    Set wheel speeds, clamped so we never exceed Webots 'maxVelocity'.
    This prevents console warnings like 'requested velocity exceeds maxVelocity'.
    """
    global MAX_WHEEL_VELOCITY
    ls = float(np.clip(left_speed, -MAX_WHEEL_VELOCITY, MAX_WHEEL_VELOCITY))
    rs = float(np.clip(right_speed, -MAX_WHEEL_VELOCITY, MAX_WHEEL_VELOCITY))

    joints["FrontLeftWheel"].setVelocity(ls)
    joints["MiddleLeftWheel"].setVelocity(ls)
    joints["BackLeftWheel"].setVelocity(ls)
    joints["FrontRightWheel"].setVelocity(rs)
    joints["MiddleRightWheel"].setVelocity(rs)
    joints["BackRightWheel"].setVelocity(rs)


def wheels_straight():
    joints["FrontLeftArm"].setPosition(0.0)
    joints["FrontRightArm"].setPosition(0.0)
    joints["BackRightArm"].setPosition(0.0)
    joints["BackLeftArm"].setPosition(0.0)


def webots_to_opencv(image_data, width, height):
    if image_data:
        np_array = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        image_bgr = np_array[:, :, :3]
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return gray_image
    return None


def select_nearest_tag(detections, camera_params, tag_size=TAG_SIZE):
    min_dist = float('inf')
    selected_tag = None
    selected_id = None
    for d in detections:
        tag_dist = float('inf')
        # If pose is available, use it
        if hasattr(d, 'pose_t') and d.pose_t is not None and len(d.pose_t) == 3:
            try:
                tag_dist = abs(float(d.pose_t[2]))
            except Exception:
                tag_dist = float('inf')
        # Otherwise, approximate from pixel width
        elif len(d.corners) == 4:
            c = np.array(d.corners)
            tag_width_px = np.linalg.norm(c[0] - c[1])
            fx = camera_params[0]
            if tag_width_px > 0:
                tag_dist = (tag_size * fx) / tag_width_px

        if tag_dist < min_dist:
            min_dist = tag_dist
            selected_tag = d
            selected_id = d.tag_id

    return selected_tag, selected_id, min_dist


def rotate_to_center(robot, tag, camera, tolerance=10):
    width = camera.getWidth()
    center_screen = width / 2.0
    tag_x = tag.center[0]
    offset = tag_x - center_screen

    if abs(offset) < tolerance:
        move_wheels(0, 0)
        print("Tag centered.")
        return True
    else:
        # logical speed, will be clamped by move_wheels()
        turn_speed = np.clip(0.004 * offset, -MAX_SPEED, MAX_SPEED)
        move_wheels(-turn_speed, turn_speed)
        robot.step(TIME_STEP)
        return False


def rangefinder_distance(range_finder):
    """
    Returns a robust front distance in meters.
    Fixes:
      - Uses width & height to sample the true center (3x3 window).
      - Uses median of that window to reduce noise.
      - Uses getMaxRange() to clamp invalid readings.
    """
    width = range_finder.getWidth()
    height = range_finder.getHeight()
    max_range = range_finder.getMaxRange()

    image = range_finder.getRangeImage()
    if image is None or len(image) == 0:
        return max_range

    # Webots returns width*height floats in row-major order.
    cx = width // 2
    cy = height // 2

    vals = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x = min(max(cx + dx, 0), width - 1)
            y = min(max(cy + dy, 0), height - 1)
            idx = y * width + x
            try:
                v = float(image[idx])
            except Exception:
                continue
            if np.isfinite(v) and v > 0.0:
                vals.append(v)

    if not vals:
        return max_range

    val = float(np.median(vals))

    if not np.isfinite(val) or val <= 0.0:
        return max_range
    if val > max_range:
        val = max_range

    return val


def turn_to_heading(robot, current_yaw, target_yaw, turn_gain=1800, tol=0.1, max_steps=40):
    """
    Turn in-place to approximately reach target_yaw from current_yaw.
    Includes guards against NaN / inf so int() never crashes.
    Speeds are clamped to MAX_WHEEL_VELOCITY via move_wheels().
    """
    dtheta = ((target_yaw - current_yaw + np.pi) % (2 * np.pi)) - np.pi

    # Safety: avoid non-finite angles
    if not np.isfinite(dtheta):
        print(f"[WARN] Non-finite dtheta in turn_to_heading "
              f"(current={current_yaw}, target={target_yaw}). Skipping turn.")
        move_wheels(0, 0)
        robot.step(TIME_STEP)
        return

    if abs(dtheta) < tol:
        move_wheels(0, 0)
        return

    # Compute approximate number of step cycles, with safety clamps
    raw_steps = abs(dtheta) / 0.15
    if not np.isfinite(raw_steps) or raw_steps <= 0:
        steps = 1
    else:
        steps = int(raw_steps) + 1

    steps = max(1, min(steps, max_steps))
    sign = np.sign(dtheta) if dtheta != 0 else 0

    print(f"Turning: dtheta={dtheta:.3f} rad, steps={steps}, sign={sign}")

    for _ in range(steps):
        # turn_gain is logical, move_wheels will clamp to MAX_WHEEL_VELOCITY
        move_wheels(sign * turn_gain, -sign * turn_gain)
        robot.step(TIME_STEP)

    move_wheels(0, 0)
    robot.step(TIME_STEP)


def go_to_waypoint(robot, range_finder, start, goal, heading, use_obstacle_avoidance=True):
    rover_x, rover_y = start
    goal_x, goal_y = goal
    dx = goal_x - rover_x
    dy = goal_y - rover_y
    target_dist = np.hypot(dx, dy)
    target_heading = math.atan2(dx, dy)

    cur_heading = heading
    # Align initial heading
    turn_to_heading(robot, cur_heading, target_heading)
    cur_heading = target_heading

    cycles = 0
    while target_dist > WAYPOINT_TOLERANCE and cycles < 1500:
        dx = goal_x - rover_x
        dy = goal_y - rover_y
        target_dist = np.hypot(dx, dy)
        target_heading = math.atan2(dx, dy)

        heading_err = ((target_heading - cur_heading + np.pi) % (2 * np.pi)) - np.pi

        if abs(heading_err) > 0.06:
            # Align heading in-place, then move
            move_wheels(0, 0)
            robot.step(TIME_STEP)
            turn_to_heading(robot, cur_heading, target_heading)
            cur_heading = target_heading

        # Obstacle check
        rf_front = rangefinder_distance(range_finder)
        print(f"Waypoint nav: dist={target_dist:.2f}, heading_err={heading_err:.2f}, rf_front={rf_front:.2f}")

        if use_obstacle_avoidance and rf_front < OBSTACLE_DIST:
            print("Obstacle detected! Backing up and turning to avoid.")
            # Backup a bit
            move_wheels(-VELOCITY, -VELOCITY)
            for _ in range(int(SAFE_BACKUP_DIST * 5)):  # empirical "backup time"
                robot.step(TIME_STEP)
            move_wheels(0, 0)

            # Turn 45 deg right to try to go around
            for _ in range(8):
                move_wheels(VELOCITY, -VELOCITY)
                robot.step(TIME_STEP)
            move_wheels(0, 0)

            # Update heading estimate
            cur_heading -= np.pi / 4.0
            continue

        # Move forward towards goal
        move_wheels(VELOCITY, VELOCITY)
        for _ in range(3):
            robot.step(TIME_STEP)

        # Dead-reckoning (approx position update)
        rover_x += np.sin(cur_heading) * 0.03  # ~3 cm per cycle
        rover_y += np.cos(cur_heading) * 0.03
        cycles += 1

    move_wheels(0, 0)
    robot.step(TIME_STEP)
    print("Reached waypoint.")

    return rover_x, rover_y, cur_heading


def run_robot():
    global MAX_WHEEL_VELOCITY

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # Device Initialization
    for name in JOINT_NAMES:
        try:
            joints[name] = robot.getDevice(name)
        except Exception as e:
            sys.stderr.write(f"Error: Could not find device '{name}'. {e}\n")
            return

    try:
        camera = robot.getDevice("camera")
        camera.enable(time_step)
    except Exception:
        sys.stderr.write("Error: Could not find device 'camera'.\n")
        return

    try:
        range_finder = robot.getDevice("range-finder")
        range_finder.enable(time_step)
    except Exception:
        sys.stderr.write("Error: Could not find device 'range-finder'.\n")
        return

    # Set wheel motors to velocity control and detect maxVelocity
    wheel_names = [
        "FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel",
        "FrontRightWheel", "MiddleRightWheel", "BackRightWheel"
    ]

    max_vals = []
    for wheel in wheel_names:
        motor = joints[wheel]
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
        try:
            mv = motor.getMaxVelocity()
            max_vals.append(mv)
        except Exception:
            pass

    if max_vals:
        MAX_WHEEL_VELOCITY = min(max_vals)
    else:
        MAX_WHEEL_VELOCITY = 1.0

    print(f"[INFO] Detected MAX_WHEEL_VELOCITY = {MAX_WHEEL_VELOCITY}")

    camera_params = (600, 600, camera.getWidth() / 2.0, camera.getHeight() / 2.0)
    at_detector = Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,
        refine_edges=1
    )

    # --- COORDINATE MAPPING STATE ---
    rover_x, rover_y, rover_yaw = 0.0, 0.0, 0.0  # (x, y, heading(rad))
    visited_tags = set()

    print("Starting multi-tag mission...")

    # Try to handle up to len(TARGET_TAG_IDS) tags
    for mission_step in range(len(TARGET_TAG_IDS)):
        print(f"\n=== Mission step {mission_step + 1} ===")
        found_tag = False
        tag_id = None
        goal_x = None
        goal_y = None

        # --- SEARCH FOR NEXT UNVISITED TAG ---
        search_loops = 0
        while robot.step(time_step) != -1 and not found_tag:
            search_loops += 1
            if search_loops > 2000:
                print("Search timeout for next tag. Ending mission.")
                move_wheels(0, 0)
                return

            image_data = camera.getImage()
            grayscale_frame = webots_to_opencv(
                image_data,
                camera.getWidth(),
                camera.getHeight()
            )

            selected_tag = None
            selected_id = None
            if grayscale_frame is not None:
                detections = at_detector.detect(
                    grayscale_frame,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=TAG_SIZE
                )
                # Only consider desired IDs and not-yet-visited tags
                detections = [
                    d for d in detections
                    if d.tag_id in TARGET_TAG_IDS and d.tag_id not in visited_tags
                ]

                if detections:
                    selected_tag, selected_id, selected_dist = select_nearest_tag(
                        detections,
                        camera_params,
                        tag_size=TAG_SIZE
                    )

            if selected_tag:
                centered = rotate_to_center(robot, selected_tag, camera)
                if not centered:
                    continue

                # Get tag distance from range finder (front distance)
                measured_dist = rangefinder_distance(range_finder)
                if not np.isfinite(measured_dist):
                    measured_dist = range_finder.getMaxRange()

                tag_coord_x = rover_x + measured_dist * math.sin(rover_yaw)
                tag_coord_y = rover_y + measured_dist * math.cos(rover_yaw)
                tag_id = selected_id

                print(
                    f"\n--- Tag acquired ---\n"
                    f"Rover at (x={rover_x:.2f}, y={rover_y:.2f}), heading={rover_yaw:.2f} rad.\n"
                    f"Tag_id: {tag_id}, tag_coord=({tag_coord_x:.2f},{tag_coord_y:.2f}), "
                    f"measured_dist={measured_dist:.2f} m\n"
                )

                # Calculate waypoint distance based on tag behavior
                if tag_id == 0:
                    goal_dist = measured_dist - 1.5
                else:
                    goal_dist = measured_dist - 1.0

                if not np.isfinite(goal_dist):
                    goal_dist = 1.0

                goal_x = rover_x + goal_dist * math.sin(rover_yaw)
                goal_y = rover_y + goal_dist * math.cos(rover_yaw)

                print(
                    f"Navigation goal: {goal_dist:.2f} m in front of tag: "
                    f"({goal_x:.2f}, {goal_y:.2f})\n"
                )

                found_tag = True
            else:
                wheels_straight()
                move_wheels(VELOCITY * 0.3, -VELOCITY * 0.3)
                print("No new tag found. Rotating to search...")

        if not found_tag or goal_x is None or goal_y is None:
            print("No further tags found. Ending mission.")
            move_wheels(0, 0)
            break

        # --- NAVIGATE TO TARGET COORD ---
        rover_x, rover_y, rover_yaw = go_to_waypoint(
            robot,
            range_finder,
            (rover_x, rover_y),
            (goal_x, goal_y),
            rover_yaw,
            use_obstacle_avoidance=True
        )

        # --- Alignment at Goal: Turn 180 from approach heading ---
        print("At goal, aligning 180° from approach direction...")
        new_yaw = (rover_yaw + np.pi) % (2 * np.pi)
        turn_to_heading(robot, rover_yaw, new_yaw)
        rover_yaw = new_yaw

        # --- Perform tag-based action ---
        if tag_id == 0:
            print("ID 0: STOPPING at 1.5 m before tag, facing opposite direction. Mission ends here.")
            move_wheels(0, 0)
            robot.step(TIME_STEP)
            visited_tags.add(tag_id)
            break  # final tag behavior
        elif tag_id == 1:
            print("ID 1: Performing 90° right turn.")
            target_yaw = (rover_yaw - np.pi / 2.0) % (2 * np.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw
        elif tag_id == 2:
            print("ID 2: Performing 90° left turn.")
            target_yaw = (rover_yaw + np.pi / 2.0) % (2 * np.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw
        else:
            print(f"Tag {tag_id} has no special behavior. Continuing.")

        visited_tags.add(tag_id)
        print(f"Finished behavior for tag {tag_id}. Looking for next tag...")

    print("Mission complete. No more tags or mission steps.")
    move_wheels(0, 0)


if __name__ == '__main__':
    run_robot()
