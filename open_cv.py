import numpy as np
import cv2
from pupil_apriltags import Detector
from controller import Robot, Camera, Motor, InertialUnit
import math
import sys

# GLOBAL CONSTANTS
TIME_STEP = 640
VELOCITY = 800             # logical speed, will be clamped to motor max
MAX_SPEED = 2000.0         # logical max for turns, also clamped
TAG_FAMILY = "tag25h9"
TAG_SIZE = 0.50            # meters
TARGET_TAG_IDS = [0, 1, 2]
WAYPOINT_TOLERANCE = 0.02  # meters
MAX_TAG_VISITS = 5         # visit ~5 tags total (IDs can repeat)

# MOTOR NAMES
JOINT_NAMES = [
    "BackLeftBogie", "FrontLeftBogie", "FrontLeftArm", "BackLeftArm",
    "FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel",
    "BackRightBogie", "FrontRightBogie", "FrontRightArm", "BackRightArm",
    "FrontRightWheel", "MiddleRightWheel", "BackRightWheel"
]

joints = {}
MAX_WHEEL_VELOCITY = 1.0   # will be overwritten from motors
imu = None                 # InertialUnit (for yaw)


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
    """
    Picks the nearest tag using either:
      - pose_t[2] (Z distance) if pose is estimated, or
      - pixel width with pinhole camera geometry.
    Returns: (selected_tag, selected_id, distance_m)
    """
    min_dist = float('inf')
    selected_tag = None
    selected_id = None

    fx = camera_params[0]

    for d in detections:
        tag_dist = float('inf')

        # 1) If pose is available, use the Z distance from the camera
        if hasattr(d, 'pose_t') and d.pose_t is not None:
            try:
                t = np.array(d.pose_t).reshape(-1)  # handle (3,1) or (3,)
                if len(t) >= 3:
                    tag_dist = abs(float(t[2]))
            except Exception:
                tag_dist = float('inf')

        # 2) Otherwise, estimate from pixel width of the tag
        if (not np.isfinite(tag_dist) or tag_dist == float('inf')) and hasattr(d, 'corners') and len(d.corners) == 4:
            c = np.array(d.corners)
            tag_width_px = np.linalg.norm(c[0] - c[1])
            if tag_width_px > 0:
                tag_dist = (tag_size * fx) / tag_width_px

        if tag_dist < min_dist:
            min_dist = tag_dist
            selected_tag = d
            selected_id = d.tag_id

    return selected_tag, selected_id, min_dist


def rotate_to_center(robot, tag, camera, tolerance=10):
    """
    Align robot so the detected tag is horizontally centered in the CAMERA image.
    """
    width = camera.getWidth()
    center_screen = width / 2.0
    tag_x = tag.center[0]
    offset = tag_x - center_screen

    if abs(offset) < tolerance:
        move_wheels(0, 0)
        print("Tag centered in camera.")
        return True
    else:
        # logical speed, will be clamped by move_wheels()
        turn_speed = np.clip(0.004 * offset, -MAX_SPEED, MAX_SPEED)
        move_wheels(-turn_speed, turn_speed)
        robot.step(TIME_STEP)
        return False


def get_yaw():
    """
    Return current yaw (heading around vertical axis) from the InertialUnit.
    If IMU is missing, returns 0.0 so code still runs (turns will be approximate).
    """
    global imu
    if imu is None:
        return 0.0
    try:
        roll, pitch, yaw = imu.getRollPitchYaw()
        return float(yaw)
    except Exception:
        return 0.0


def turn_to_heading(robot, target_yaw, tol=0.1, kp=1.0, max_steps=200):
    """
    Turn in-place until IMU yaw is close to target_yaw (in radians).
    Uses a simple P-controller on heading error.
    """
    global MAX_WHEEL_VELOCITY

    for step in range(max_steps):
        current_yaw = get_yaw()

        # Wrap smallest angular difference into [-pi, pi]
        dtheta = ((target_yaw - current_yaw + math.pi) % (2 * math.pi)) - math.pi

        if abs(dtheta) < tol:
            print(f"[turn_to_heading] Reached target yaw. dtheta={dtheta:.3f} rad")
            break

        # Proportional control for turn speed
        turn_speed = kp * dtheta  # sign of dtheta controls direction

        # Turn in place: (your robot config) left = turn_speed, right = -turn_speed
        left = turn_speed
        right = -turn_speed

        # Clamp to motor limits
        left = float(np.clip(left, -MAX_WHEEL_VELOCITY, MAX_WHEEL_VELOCITY))
        right = float(np.clip(right, -MAX_WHEEL_VELOCITY, MAX_WHEEL_VELOCITY))

        move_wheels(left, right)
        robot.step(TIME_STEP)

    move_wheels(0, 0)
    robot.step(TIME_STEP)


def turn_relative(robot, delta_angle, tol=0.5):
    """
    Turn the rover by delta_angle (radians) relative to current yaw.
    Positive -> one turn direction, Negative -> opposite.
    """
    current_yaw = get_yaw()
    target_yaw = (current_yaw + delta_angle) % (2 * math.pi)
    turn_to_heading(robot, target_yaw, tol=tol)


def approach_tag(robot, camera, at_detector, camera_params, target_id, desired_stop_dist):
    """
    Move towards the tag using CAMERA distance in a feedback loop.

    Key behavior for partial / blocked tags:
    - While the tag is visible: keep recentering and using its distance.
    - Once we've seen the tag at least once, if it later becomes partially
      blocked or disappears (close to it), we:
        * do NOT spin to search again immediately,
        * if it stays lost for several frames in a row, then we use the last
          known distance to drive forward in open loop to roughly reach
          desired_stop_dist.

    While we are in that "calculated steps" open-loop, we DO NOT re-center or
    track other tags – we just finish the steps and exit.
    """
    lost_counter = 0
    seen_once = False
    last_dist = None

    # Approximate distance the rover moves in one "open-loop" step (meters)
    approx_step_dist = 0.6
    # How many consecutive lost frames we tolerate before switching to open-loop
    LOST_GRACE_FRAMES = 0

    while robot.step(TIME_STEP) != -1:
        image_data = camera.getImage()
        gray = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
        if gray is None:
            move_wheels(0, 0)
            print("No camera image while approaching.")
            break

        detections = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )

        detections = [d for d in detections if d.tag_id == target_id]

        # --- CASE 1: Tag visible in this frame ---
        if detections:
            lost_counter = 0          # reset lost counter (tag came back)
            seen_once = True

            selected_tag, _, dist = select_nearest_tag(detections, camera_params, TAG_SIZE)
            if selected_tag is None or not np.isfinite(dist):
                print("[Approach] Invalid distance from tag, skipping step.")
                continue

            last_dist = dist  # remember the most recent valid distance

            # Keep it centered
            centered = rotate_to_center(robot, selected_tag, camera)
            if not centered:
                # rotate_to_center already stepped the robot
                continue

            print(f"[Approach] Tag {target_id}: dist={dist:.3f} m, target={desired_stop_dist:.3f} m")

            # Check if we reached the desired distance
            if dist <= desired_stop_dist + WAYPOINT_TOLERANCE:
                print("[Approach] Reached desired stop distance from tag.")
                move_wheels(0, 0)
                robot.step(TIME_STEP)
                break

            # Still far: move a bit forward under feedback
            move_wheels(VELOCITY, VELOCITY)
            for _ in range(2):
                robot.step(TIME_STEP)

        # --- CASE 2: Tag NOT visible in this frame ---
        else:
            # Never seen the tag at all in this approach -> classic search
            if not seen_once:
                lost_counter += 1
                print(f"[Approach] Tag {target_id} not yet seen (lost_count={lost_counter}). Searching...")
                if lost_counter > 10:
                    print("[Approach] Could not see tag at all, aborting approach.")
                    move_wheels(0, 0)
                    break
                # Slow rotate to search again
                move_wheels(0.2 * MAX_WHEEL_VELOCITY, -0.2 * MAX_WHEEL_VELOCITY)
                continue

            # We HAVE seen the tag before: now it's partially blocked or out of FOV.
            lost_counter += 1
            print(f"[Approach] Tag {target_id} temporarily lost after being seen. lost_count={lost_counter}")

            # If it's a short glitch, just creep forward a bit and hope it reappears.
            if lost_counter <= LOST_GRACE_FRAMES:
                move_wheels(VELOCITY * 0.5, VELOCITY * 0.5)
                robot.step(TIME_STEP)
                continue

            # Lost for longer than grace window and we have a last valid distance:
            # -> finish open-loop (calculated steps) and DO NOT track other tags
            if last_dist is not None:
                remaining = max(0.0, last_dist - desired_stop_dist)
                if remaining > 0.0:
                    steps = int(remaining / approx_step_dist)
                    print(f"[Approach] Tag {target_id} lost for long; finishing in open-loop "
                          f"for ~{remaining:.2f} m ({steps} steps) using last_dist={last_dist:.2f} m.")
                    for _ in range(steps):
                        move_wheels(VELOCITY, VELOCITY)
                        robot.step(TIME_STEP)

            # After open-loop finish, stop the approach
            move_wheels(0, 0)
            robot.step(TIME_STEP)
            break

    move_wheels(0, 0)


def run_robot():
    global MAX_WHEEL_VELOCITY, imu

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # --- InertialUnit (IMU) ---
    try:
        imu = robot.getDevice("inertial unit")  # name must match your Webots world
        imu.enable(time_step)
        print("[INFO] InertialUnit found and enabled.")
    except Exception as e:
        imu = None
        print(f"[WARN] InertialUnit not found: {e}. Using fallback yaw=0.")

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

    # Camera intrinsics: (fx, fy, cx, cy)
    # Compute fx, fy from Webots FOV
    width = camera.getWidth()
    height = camera.getHeight()
    fov = camera.getFov()  # horizontal FOV in radians

    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = fx  # assuming square pixels
    cx = width / 2.0
    cy = height / 2.0

    camera_params = (fx, fy, cx, cy)
    print(f"[INFO] Camera params: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    at_detector = Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,
        refine_edges=1
    )

    # --- STATE (optional, just for logging) ---
    rover_x, rover_y = 0.0, 0.0

    print("Starting multi-tag mission...")

    # Visit up to MAX_TAG_VISITS tags (IDs can repeat)
    for mission_step in range(MAX_TAG_VISITS):
        print(f"\n=== Mission step {mission_step + 1} ===")
        found_tag = False
        tag_id = None
        measured_dist = None

        # --- SEARCH FOR NEXT TAG (ID 0, 1, or 2) ---
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
            selected_dist = None

            if grayscale_frame is not None:
                detections = at_detector.detect(
                    grayscale_frame,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=TAG_SIZE
                )
                # Only consider desired IDs (allow repeated IDs; no visited filter)
                detections = [
                    d for d in detections
                    if d.tag_id in TARGET_TAG_IDS
                ]

                if detections:
                    selected_tag, selected_id, selected_dist = select_nearest_tag(
                        detections,
                        camera_params,
                        tag_size=TAG_SIZE
                    )

            if selected_tag is not None and selected_dist is not None and np.isfinite(selected_dist):
                # 1) Rotate so the tag is in the center of the camera
                centered = rotate_to_center(robot, selected_tag, camera)
                if not centered:
                    continue

                measured_dist = float(selected_dist)
                tag_id = selected_id

                current_yaw = get_yaw()
                print(
                    f"\n--- Tag acquired (camera-based distance) ---\n"
                    f"Rover approx pose (x={rover_x:.2f}, y={rover_y:.2f}), "
                    f"heading={current_yaw:.2f} rad.\n"
                    f"Tag_id: {tag_id}, measured_dist={measured_dist:.2f} m\n"
                )

                found_tag = True
            else:
                wheels_straight()
                move_wheels(VELOCITY * 0.3, -VELOCITY * 0.3)
                print("No new tag found. Rotating to search...")

        if not found_tag or measured_dist is None:
            print("No further tags found. Ending mission.")
            move_wheels(0, 0)
            break

        # --- APPROACH TAG USING CAMERA DISTANCE FEEDBACK + OPEN-LOOP NEARBY ---
        if tag_id == 0:
            desired_stop_dist = 1.4
        else:
            desired_stop_dist = 1.0

        print(f"Approaching tag {tag_id} to stop at ~{desired_stop_dist} m.")
        approach_tag(robot, camera, at_detector, camera_params, tag_id, desired_stop_dist)

        # ========== SIMPLE TAG-BASED ROTATIONS (NO FINE-ALIGN) ==========
        if tag_id == 0:
            # 180 degrees -> π radians
            print("ID 0: Turning 180 degrees (π radians).")
            turn_relative(robot, math.pi)

            print("ID 0: STOPPING at ~desired distance before tag, facing opposite direction. Mission ends here.")
            move_wheels(0, 0)
            robot.step(TIME_STEP)
            break  # final tag behavior (arrival marker)

        elif tag_id == 1:
            # Right 90 degrees or left 90 depending on your observed direction.
            print("ID 1: Turning right 90 degrees (-π/2 radians).")
            turn_relative(robot, -math.pi / 2.0)

        elif tag_id == 2:
            # Left 90 degrees or right 90 depending on your observed direction.
            print("ID 2: Turning left 90 degrees (+π/2 radians).")
            turn_relative(robot, math.pi / 2.0)

        else:
            print(f"Tag {tag_id} has no special behavior. Continuing.")

        print(f"Finished behavior for tag {tag_id}. Looking for next tag...")

    print("Mission complete. No more tags or mission steps.")
    move_wheels(0, 0)


if __name__ == '__main__':
    run_robot()
