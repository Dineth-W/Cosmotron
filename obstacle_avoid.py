import numpy as np
import cv2
from pupil_apriltags import Detector
from controller import Robot, Camera, Motor
import math
import sys
import time

# GLOBAL CONSTANTS
TIME_STEP = 640
VELOCITY = 800             
MAX_SPEED = 2000.0         
TAG_FAMILY = "tag25h9"
TAG_SIZE = 0.20            
TARGET_TAG_IDS = [0, 1, 2]
WAYPOINT_TOLERANCE = 0.02 

# MOTOR NAMES
JOINT_NAMES = [
    "BackLeftBogie", "FrontLeftBogie", "FrontLeftArm", "BackLeftArm",
    "FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel",
    "BackRightBogie", "FrontRightBogie", "FrontRightArm", "BackRightArm",
    "FrontRightWheel", "MiddleRightWheel", "BackRightWheel"
]

joints = {}
MAX_WHEEL_VELOCITY = 1.0  

def move_wheels(left_speed, right_speed):
    """
    Set wheel speeds, clamped so we never exceed Webots 'maxVelocity'.
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
    """Returns Grayscale image for AprilTags"""
    if image_data:
        np_array = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        image_bgr = np_array[:, :, :3]
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return gray_image
    return None

def webots_to_opencv_bgr(image_data, width, height):
    """Returns BGR image for Color Detection (Red/White)"""
    if image_data:
        np_array = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        return np_array[:, :, :3]
    return None

def select_nearest_tag(detections, camera_params, tag_size=TAG_SIZE):
    min_dist = float('inf')
    selected_tag = None
    selected_id = None

    fx = camera_params[0]

    for d in detections:
        tag_dist = float('inf')

        # 1) If pose is available, use the Z distance from the camera
        if hasattr(d, 'pose_t') and d.pose_t is not None:
            try:
                t = np.array(d.pose_t).reshape(-1)  
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
    width = camera.getWidth()
    center_screen = width / 2.0
    tag_x = tag.center[0]
    offset = tag_x - center_screen

    if abs(offset) < tolerance:
        move_wheels(0, 0)
        return True
    else:
        turn_speed = np.clip(0.004 * offset, -MAX_SPEED, MAX_SPEED)
        move_wheels(-turn_speed, turn_speed)
        robot.step(TIME_STEP)
        return False


def turn_to_heading(robot, current_yaw, target_yaw, turn_gain=1800, tol=0.1, max_steps=40):
    dtheta = ((target_yaw - current_yaw + np.pi) % (2 * np.pi)) - np.pi

    if not np.isfinite(dtheta):
        move_wheels(0, 0)
        robot.step(TIME_STEP)
        return

    if abs(dtheta) < tol:
        move_wheels(0, 0)
        return

    raw_steps = abs(dtheta) / 0.15
    if not np.isfinite(raw_steps) or raw_steps <= 0:
        steps = 1
    else:
        steps = int(raw_steps) + 1

    steps = max(1, min(steps, max_steps))
    sign = np.sign(dtheta) if dtheta != 0 else 0

    print(f"Turning: dtheta={dtheta:.3f} rad, steps={steps}, sign={sign}")

    for _ in range(steps):
        move_wheels(sign * turn_gain, -sign * turn_gain)
        robot.step(TIME_STEP)

    move_wheels(0, 0)
    robot.step(TIME_STEP)


def approach_tag(robot, camera, at_detector, camera_params, target_id, desired_stop_dist):
    last_offset = None
    last_move_time = time.time()
    stuck_threshold = 4.0  
    epsilon = 0.03         
    lost_counter = 0

    while robot.step(TIME_STEP) != -1:
        image_data = camera.getImage()
        gray = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
        if gray is None:
            move_wheels(0, 0)
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
            if lost_counter > 10:
                move_wheels(0, 0)
                break

            move_wheels(0.2 * MAX_WHEEL_VELOCITY, -0.2 * MAX_WHEEL_VELOCITY)
            continue

        lost_counter = 0

        selected_tag, _, dist = select_nearest_tag(detections, camera_params, TAG_SIZE)
        if selected_tag is None or not np.isfinite(dist):
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

        centered = rotate_to_center(robot, selected_tag, camera)
        if not centered:
            continue

        print(f"[Approach] Tag {target_id}: dist={dist:.3f} m, target={desired_stop_dist:.3f} m")

        if dist <= desired_stop_dist + WAYPOINT_TOLERANCE:
            move_wheels(0, 0)
            robot.step(TIME_STEP)
            break

        move_wheels(VELOCITY, VELOCITY)
        for _ in range(2):
            robot.step(TIME_STEP)

    move_wheels(0, 0)
    
def complex_unstuck(robot, duration=1.0, wiggle_times=4):
    print("Attempting complex unstuck maneuver...")
    move_wheels(-VELOCITY, -VELOCITY)
    for _ in range(int(duration * 1000 / TIME_STEP)):
        robot.step(TIME_STEP)
    move_wheels(0, 0)
    robot.step(TIME_STEP)
    
    segment = 12  
    for _ in range(segment):
        move_wheels(VELOCITY*0.5, -VELOCITY*0.5)
        for _ in range(3):
            robot.step(TIME_STEP)
    move_wheels(0, 0)
    robot.step(TIME_STEP)
    
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
    image_data = fence_camera.getImage()
    gray = webots_to_opencv(image_data, fence_camera.getWidth(), fence_camera.getHeight())
    if gray is not None:
        height, width = gray.shape
        roi = gray[int(0.3*height):int(0.7*height), int(0.4*width):int(0.6*width)]
        mean_val = np.mean(roi)
        if mean_val > 180: 
            print("Fence detected ahead!")
            return True
    return False
 
def detect_red_square(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
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
        if area > 150: 
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 4: 
                x, y, w, h = cv2.boundingRect(approx)
                if 0.75 < w/h < 1.25:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        return (cX, cY)
    return None

def detect_white_circle(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200: 
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2: # Loose tolerance for oblique angles
                return (int(x), int(y))
    return None
  
def move_to_ground_object(robot, bottom_camera, target_type="white"):
    """
    Visual Servo: Centers the robot over the Red Square or White Circle.
    """
    print(f"Starting move_to_ground_object for {target_type}...")
    
    width = bottom_camera.getWidth()
    height = bottom_camera.getHeight()
    center_x = width / 2.0
    center_y = height / 2.0
    
    # Assuming 60 degree FOV downwards for distance calc
    fov = 1.0 
    cam_height = 0.3 
    
    lost_count = 0
    
    while robot.step(TIME_STEP) != -1:
        image_data = bottom_camera.getImage()
        img_bgr = webots_to_opencv_bgr(image_data, width, height)
        
        target_pos = None
        if target_type == "white":
            target_pos = detect_white_circle(img_bgr)
        elif target_type == "red":
            target_pos = detect_red_square(img_bgr)
            
        if target_pos:
            lost_count = 0
            obj_x, obj_y = target_pos
            
            # Calculate Pixel Errors
            # X error: Steering
            # Y error: Forward/Backward distance
            error_x = obj_x - center_x
            error_y = center_y - obj_y  # Positive means object is "above" center (Forward)

            # Calculate approximate distance from center in meters
            fy = (width/2) / math.tan(fov/2)
            dist_x = (error_x * cam_height) / fy
            dist_y = (error_y * cam_height) / fy
            total_dist = math.sqrt(dist_x**2 + dist_y**2)
            
            print(f"Tracking {target_type}: Center Dist={total_dist:.3f}m")

            # Stop condition
            if abs(error_x) < 20 and abs(error_y) < 20:
                print("Target Centered directly underneath.")
                move_wheels(0, 0)
                return True
            
            # Control Logic
            turn_val = np.clip(0.005 * error_x, -0.5, 0.5)
            forward_val = np.clip(0.008 * error_y, -0.8, 0.8)
            
            # Differential Drive Mixing
            move_wheels((forward_val + turn_val)*VELOCITY, (forward_val - turn_val)*VELOCITY)
            
        else:
            lost_count += 1
            print("Object lost...")
            # Search behavior: Move forward slowly
            move_wheels(VELOCITY * 0.2, VELOCITY * 0.2)
            if lost_count > 30:
                print("Object completely lost. Aborting ground move.")
                move_wheels(0, 0)
                return False
    return False
      
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
        
    try:
        camera = robot.getDevice("camera")
        camera.enable(time_step)
    except Exception:
        sys.stderr.write("Error: Could not find device 'camera'.\n")
        return

    # --- NEW: Initialize Bottom Camera ---
    try:
        bottom_camera = robot.getDevice("bottom_camera")
        bottom_camera.enable(time_step)
        print("Bottom camera enabled.")
    except Exception:
        sys.stderr.write("Warning: Could not find 'bottom_camera'.\n")
        bottom_camera = None

    # Set wheel motors
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

    # Camera params
    width = camera.getWidth()
    height = camera.getHeight()
    fov = camera.getFov() 

    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = fx  
    cx = width / 2.0
    cy = height / 2.0

    camera_params = (fx, fy, cx, cy)

    at_detector = Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,
        refine_edges=1
    )

    rover_x, rover_y, rover_yaw = 0.0, 0.0, 0.0
    visited_tags = set()

    print("Starting multi-tag mission...")

    for mission_step in range(len(TARGET_TAG_IDS)):
        print(f"\n=== Mission step {mission_step + 1} ===")
        found_tag = False
        tag_id = None
        measured_dist = None

        # --- SEARCH FOR NEXT UNVISITED TAG ---
        search_loops = 0
        while robot.step(time_step) != -1 and not found_tag:
            search_loops += 1
            if search_loops > 3000:
                print("Search timeout. Ending mission.")
                move_wheels(0, 0)
                return

            image_data = camera.getImage()
            grayscale_frame = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())

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
                detections = [
                    d for d in detections
                    if d.tag_id in TARGET_TAG_IDS and d.tag_id not in visited_tags
                ]

                if detections:
                    selected_tag, selected_id, selected_dist = select_nearest_tag(detections, camera_params)

            if selected_tag is not None and selected_dist is not None:
                centered = rotate_to_center(robot, selected_tag, camera)
                if not centered:
                    continue

                measured_dist = float(selected_dist)
                tag_id = selected_id
                found_tag = True
            else:
                wheels_straight()
                move_wheels(VELOCITY * 0.3, -VELOCITY * 0.3)

        if not found_tag:
            break
        
        if detect_fence_close(robot, fence_camera, fence_distance=0.5):
            print("Fence detected! Turning around.")
            target_yaw = (rover_yaw + math.pi) % (2 * math.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw

        # --- APPROACH TAG ---
        desired_stop_dist = 0.9
        approach_tag(robot, camera, at_detector, camera_params, tag_id, desired_stop_dist)

        # --- ALIGNMENT (Lateral Offset Math from your original code) ---
        image_data = camera.getImage()
        gray = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
        detections = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE)
        detections = [d for d in detections if d.tag_id == tag_id]
        
        if detections:
            selected_tag, _, dist = select_nearest_tag(detections, camera_params, TAG_SIZE)
            if selected_tag and getattr(selected_tag, "pose_t", None) is not None:
                tx, ty, tz = selected_tag.pose_t
                theta = math.atan2(tx, tz)
                print(f"Aligning angle, theta={math.degrees(theta):.2f}°")
                turn_to_heading(robot, rover_yaw, rover_yaw - theta)
                rover_yaw = (rover_yaw - theta) % (2 * math.pi)

                # Lateral shift calculation
                r = math.sqrt(tx**2 + tz**2)
                lateral_offset = r * math.sin(theta)
                
                if abs(lateral_offset) > 0.05:
                    print(f"Correcting Lateral offset: {lateral_offset:.3f} m")
                    turn_sign = np.sign(theta)
                    turn90 = math.pi/2 * turn_sign
                    target_yaw = (rover_yaw + turn90) % (2 * math.pi)
                    turn_to_heading(robot, rover_yaw, target_yaw)
                    rover_yaw = target_yaw

                    move_wheels(VELOCITY, VELOCITY)
                    # Simple time-based drive for lateral correction
                    drive_time = abs(lateral_offset) / (VELOCITY * 0.0005) # Tuned const
                    for _ in range(int(drive_time)):
                        robot.step(time_step)
                    move_wheels(0, 0)

                    target_yaw = (rover_yaw - turn90) % (2 * math.pi)
                    turn_to_heading(robot, rover_yaw, target_yaw)
                    rover_yaw = target_yaw
                
                # Final face-up
                target_yaw = (rover_yaw + math.pi/2) % (2 * math.pi) # Assuming tag was 90deg off? Adjust if needed
                # Usually simple realignment to tag is enough:
                # approach_tag handles the final centering.

        # --- TAG SPECIFIC BEHAVIOR ---
        if tag_id == 0:
            # 1. Turn 180
            print("ID 0 Detected: Turning 180 degrees.")
            target_yaw = (rover_yaw + math.pi) % (2 * math.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw
            
            print("ID 0: Stopped. Activating Bottom Camera for Ground Objects.")
            move_wheels(0, 0)
            for _ in range(10): robot.step(time_step) # Wait a moment

            if bottom_camera:
                # 2. Search Logic
                # We scan for White Circle first, then Red Square
                # We might need to wiggle or drive forward slightly if it's not immediately visible
                found_obj = False
                
                # Quick check without moving
                img_data = bottom_camera.getImage()
                bgr = webots_to_opencv_bgr(img_data, bottom_camera.getWidth(), bottom_camera.getHeight())
                
                if detect_white_circle(bgr):
                    print("White Circle detected immediately.")
                    move_to_ground_object(robot, bottom_camera, "white")
                    found_obj = True
                elif detect_red_square(bgr):
                    print("Red Square detected immediately.")
                    move_to_ground_object(robot, bottom_camera, "red")
                    found_obj = True
                
                # If not found immediately, drive forward slowly looking for it
                if not found_obj:
                    print("Ground object not in view. Driving forward to scan...")
                    scan_steps = 0
                    while robot.step(time_step) != -1 and scan_steps < 100: # Drive for ~3 seconds
                        scan_steps += 1
                        move_wheels(VELOCITY*0.3, VELOCITY*0.3)
                        
                        img_data = bottom_camera.getImage()
                        bgr = webots_to_opencv_bgr(img_data, bottom_camera.getWidth(), bottom_camera.getHeight())
                        
                        if detect_white_circle(bgr):
                            print("Found White Circle while scanning!")
                            move_to_ground_object(robot, bottom_camera, "white")
                            break
                        elif detect_red_square(bgr):
                            print("Found Red Square while scanning!")
                            move_to_ground_object(robot, bottom_camera, "red")
                            break
                    move_wheels(0,0)

            visited_tags.add(tag_id)

        elif tag_id == 1:
            print("ID 1: Turning right 90 degrees.")
            target_yaw = (rover_yaw - (math.pi / 2.0)) % (2 * math.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw
            visited_tags.add(tag_id)

        elif tag_id == 2:
            print("ID 2: Turning left 90 degrees.")
            target_yaw = (rover_yaw + (math.pi / 2.0)) % (2 * math.pi)
            turn_to_heading(robot, rover_yaw, target_yaw)
            rover_yaw = target_yaw
            visited_tags.add(tag_id)

        print(f"Finished behavior for tag {tag_id}.")

    print("Mission complete.")
    move_wheels(0, 0)


if __name__ == '__main__':
    run_robot()
