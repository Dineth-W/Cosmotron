import numpy as np
import cv2
from pupil_apriltags import Detector
from controller import Robot, Camera, Motor
import math
import sys
import time

# GLOBAL CONSTANTS
TIME_STEP = 640
VELOCITY = 800             # logical speed, will be clamped to motor max
MAX_SPEED = 2000.0         # logical max for turns, also clamped
TAG_FAMILY = "tag25h9"
TAG_SIZE = 0.20            # meters
TARGET_TAG_IDS = [0, 1, 2]
WAYPOINT_TOLERANCE = 0.02  # meters

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
    This prevents console warnings like  'requested velocity exceeds maxVelocity'.
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


def approach_tag(robot, camera, at_detector, camera_params, target_id, desired_stop_dist):
    """
    Move towards the tag using CAMERA distance in a feedback loop.
    - Re-detects the tag each cycle.
    - Keeps it centered.
    - Stops when measured distance ~= desired_stop_dist (within WAYPOINT_TOLERANCE).
    """
    last_offset = None
    last_move_time = time.time()
    stuck_threshold = 4.0  # seconds
    epsilon = 0.03         # minimal change in offset to count as moving
    lost_counter = 0

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

        if not detections:
            lost_counter += 1
            print(f"[Approach] Tag {target_id} lost ({lost_counter}).")
            if lost_counter > 10:
                print("[Approach] Tag lost for too long, stopping.")
                move_wheels(0, 0)
                break

            # Slow rotate to search again
            move_wheels(0.2 * MAX_WHEEL_VELOCITY, -0.2 * MAX_WHEEL_VELOCITY)
            continue

        lost_counter = 0

        selected_tag, _, dist = select_nearest_tag(detections, camera_params, TAG_SIZE)
        if selected_tag is None or not np.isfinite(dist):
            print("[Approach] Invalid distance from tag, skipping step.")
            continue
        
        
        # ---- STUCK DETECTION BLOCK ----
        if last_offset is None or abs(dist - last_offset) > epsilon:
            last_offset = dist
            last_move_time = time.time()
        else:
            if time.time() - last_move_time > stuck_threshold:
                print("Stuck detected! Running complex unstuck maneuver.")
                complex_unstuck(robot)
                last_offset = None
                last_move_time = time.time()
                continue
# ---- END STUCK DETECTION BLOCK ----
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

        # Move a bit forward
        move_wheels(VELOCITY, VELOCITY)
        for _ in range(2):
            robot.step(TIME_STEP)

    move_wheels(0, 0)
    
def complex_unstuck(robot, duration=1.0, wiggle_times=4):
    print("Attempting complex unstuck maneuver...")
    # Move backward for 'duration' seconds
    move_wheels(-VELOCITY, -VELOCITY)
    for _ in range(int(duration * 1000 / TIME_STEP)):
        robot.step(TIME_STEP)
    move_wheels(0, 0)
    robot.step(TIME_STEP)
    # Rotate 360° slowly
    segment = 12  # split 360 into small segments
    for _ in range(segment):
        move_wheels(VELOCITY*0.5, -VELOCITY*0.5)
        for _ in range(3):
            robot.step(TIME_STEP)
    move_wheels(0, 0)
    robot.step(TIME_STEP)
    # Wiggle left and right
    for i in range(wiggle_times):
        speed = VELOCITY*0.5
        factor = (-1) ** i
        move_wheels(factor*speed, -factor*speed)
        for _ in range(4):
            robot.step(TIME_STEP)
        move_wheels(0, 0)
        robot.step(TIME_STEP)
    print("Complex unstuck maneuver complete.")

def detect_fence_close(robot, fence_camera, fence_distance=0.5):
    """
    Returns True if a fence is detected within 0.5m in front of the camera.
    This function can be replaced with image thresholding, texture/color detection, or distance sensor logic.
    """

    image_data = fence_camera.getImage()
    gray = webots_to_opencv(image_data, fence_camera.getWidth(), fence_camera.getHeight())

    # Replace below with better detection (e.g. color/texture segmentation matching fence)
    # Example: count white pixels if fence is painted white and camera sees fence at center of frame
    if gray is not None:
        height, width = gray.shape
        roi = gray[int(0.3*height):int(0.7*height), int(0.4*width):int(0.6*width)]
        mean_val = np.mean(roi)
        # Assume fence gives a bright/dark return (tune this threshold)
        if mean_val > 180:  # If fence is white; adjust for your environment
            print("Fence detected ahead!")
            return True
    return False
 
def detect_red_square(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Adjust HSV ranges for your arena lighting
    lower_red1 = np.array([0, 80, 20])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 20])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:  # adjust min size
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 4:  # square or rectangle
                # Check if it's roughly square
                x, y, w, h = cv2.boundingRect(approx)
                if 0.75 < w/h < 1.25:
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return (cX, cY)
    return None

def detect_white_circle(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Option 1: HoughCircles for circles
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=50, param2=30, minRadius=15, maxRadius=80)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the largest circle/furthest from edge
        for circle in circles[0, :]:
            cX, cY, radius = circle
            return (cX, cY)
    # Option 2: Contour approximation for case where Hough fails
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # adjust threshold as needed
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if 0.75 < (cv2.arcLength(cnt, True)/(2*np.pi*radius)) < 1.25:
                return (int(x), int(y))
    return None
  
def move_to_ground_object(robot, camera, object_pixel_pos, tolerance=10):
    width = camera.getWidth()
    center_x = width / 2.0
    obj_x = object_pixel_pos[0]
    offset = obj_x - center_x
    while abs(offset) > tolerance:
        turn_speed = np.clip(0.004 * offset, -MAX_SPEED, MAX_SPEED)
        move_wheels(-turn_speed, turn_speed)
        robot.step(TIME_STEP)
        image_data = camera.getImage()
        gray = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Re-detect to get new location
        # (insert your detection logic again here)
        # offset = ...
    move_wheels(VELOCITY, VELOCITY)
    for _ in range(20):  # Adjust steps as needed
        robot.step(TIME_STEP)
    move_wheels(0, 0)
    return True
      
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
        fence_camera = robot.getDevice("fence_camera")
        fence_camera.enable(time_step)
    except Exception:
        sys.stderr.write("Error: Could not find device 'fence_camera'.\n")
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
    # Better: compute fx, fy from Webots FOV instead of hard-coded 600
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
    rover_x, rover_y, rover_yaw = 0.0, 0.0, 0.0
    visited_tags = set()

    print("Starting multi-tag mission...")

    

    # Try to handle up to len(TARGET_TAG_IDS) tags
    for mission_step in range(len(TARGET_TAG_IDS)):
        print(f"\n=== Mission step {mission_step + 1} ===")
        found_tag = False
        tag_id = None
        measured_dist = None

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
            selected_dist = None

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

            if selected_tag is not None and selected_dist is not None and np.isfinite(selected_dist):
                # 1) Rotate so the tag is in the center of the camera
                centered = rotate_to_center(robot, selected_tag, camera)
                if not centered:
                    continue

                measured_dist = float(selected_dist)
                tag_id = selected_id

                print(
                    f"\n--- Tag acquired (camera-based distance) ---\n"
                    f"Rover approx pose (x={rover_x:.2f}, y={rover_y:.2f}), heading={rover_yaw:.2f} rad.\n"
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
        if detect_fence_close(robot, fence_camera, fence_distance=0.5):
            print("Turning around to avoid fence!")
            target_yaw = (rover_yaw + math.pi) % (2 * math.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw
    # Optionally stop after turning, or resume other logic
        # --- APPROACH TAG USING CAMERA DISTANCE FEEDBACK ---
        if tag_id == 0:
            desired_stop_dist = 0.9
        else:
            desired_stop_dist = 0.9

        print(f"Approaching tag {tag_id} to stop at ~{desired_stop_dist} m.")
        approach_tag(robot, camera, at_detector, camera_params, tag_id, desired_stop_dist)

        image_data = camera.getImage()
        gray = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
        detections = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )
        detections = [d for d in detections if d.tag_id == tag_id]
        if not detections:
            print("Tag not found for precise alignment")
        else:
            selected_tag, _, dist = select_nearest_tag(detections, camera_params, TAG_SIZE)
            if selected_tag and getattr(selected_tag, "pose_t", None) is not None:
                tx, ty, tz = selected_tag.pose_t
                theta = math.atan2(tx, tz)
                print(f"Aligning, theta={math.degrees(theta):.2f}°")
                turn_to_heading(robot, rover_yaw, rover_yaw - theta)
                rover_yaw = (rover_yaw - theta) % (2 * math.pi)

                r = math.sqrt(tx**2 + tz**2)
                lateral_offset = r * math.sin(theta)
                print(f"Lateral offset: {lateral_offset:.3f} m")
                if abs(lateral_offset) > 0.01:
                    turn_sign = np.sign(theta)
                    turn90 = math.pi/2 * turn_sign
                    target_yaw = (rover_yaw + turn90) % (2 * math.pi)
                    turn_to_heading(robot, rover_yaw, target_yaw)
                    rover_yaw = target_yaw

                    move_wheels(VELOCITY, VELOCITY)
                    drive_time = abs(lateral_offset) / (VELOCITY * 0.01)
                    for _ in range(int(drive_time)):
                        robot.step(TIME_STEP)
                    move_wheels(0, 0)

                    target_yaw = (rover_yaw - turn90) % (2 * math.pi)
                    turn_to_heading(robot, rover_yaw, target_yaw)
                    rover_yaw = target_yaw

                # Optional: face the tag again (may not be needed if previous step did it)
                print("Final 90° turn to face the tag")
                target_yaw = (rover_yaw + math.pi/2) % (2 * math.pi)
                turn_to_heading(robot, rover_yaw, target_yaw)
                rover_yaw = target_yaw
            # --- Tag-based rotations (in radians) ---
            if tag_id == 0:
                # 180 degrees -> π radians
                print("ID 0: Turning 180 degrees (π radians).")
                target_yaw = (rover_yaw + math.pi) % (2 * math.pi)
                turn_to_heading(robot, rover_yaw, target_yaw)
                rover_yaw = target_yaw

                print("ID 0: STOPPING at 0.3 m before tag, facing opposite direction. Mission ends here.")
                move_wheels(0, 0)
                robot.step(TIME_STEP)
                visited_tags.add(tag_id)
                #break  # final tag behavior
                
                image_data = camera.getImage()
                ground_img = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
            
                if ground_img is not None:
                    # Convert back to BGR for color detection
                    color_img = cv2.cvtColor(ground_img, cv2.COLOR_GRAY2BGR)
                    # Red square
                    red_square_pos = detect_red_square(color_img)
                    # White circle
                    white_circle_pos = detect_white_circle(color_img)
            
                    if red_square_pos:
                        print("Red square detected at:", red_square_pos)
                        # Turn towards and approach red square
                        success = move_to_ground_object(robot, camera, red_square_pos)
                        if success:
                            print("Arrived at red square!")
                    elif white_circle_pos:
                        print("White circle detected at:", white_circle_pos)
                        success = move_to_ground_object(robot, camera, white_circle_pos)
                        if success:
                            print("Arrived at white circle!")
                    else:
                        print("No red square or white circle detected.")
            elif tag_id == 1:
                # Right 90 degrees -> -π/2 radians
                print("ID 1: Turning right 90 degrees (-π/2 radians).")
                target_yaw = (rover_yaw - (math.pi / 2.0)) % (2 * math.pi)
                turn_to_heading(robot, rover_yaw, target_yaw)
                rover_yaw = target_yaw

            elif tag_id == 2:
                # Left 90 degrees -> +π/2 radians
                print("ID 2: Turning left 90 degrees (+π/2 radians).")
                target_yaw = (rover_yaw + (math.pi / 2.0)) % (2 * math.pi)
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
