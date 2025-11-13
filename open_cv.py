import numpy as np
import cv2
from pupil_apriltags import Detector
from controller import Robot, Camera, Motor, RangeFinder
import math
import sys

# GLOBAL CONSTANTS
TIME_STEP = 640
VELOCITY = 800
MAX_SPEED = 2000.0
TAG_FAMILY = "tag25h9"
TAG_SIZE = 0.20 # meters
TARGET_TAG_IDS = [0, 1, 2]
WAYPOINT_TOLERANCE = 0.08 # meters
OBSTACLE_DIST = 0.25      # meters for a collision detection (front)
SAFE_BACKUP_DIST = 0.5    # distance to move back (in meters) to go around
OBSTACLE_TIMEOUT = 30     # cycles before giving up on obstacle avoidance

# MOTOR NAMES
JOINT_NAMES = [
    "BackLeftBogie", "FrontLeftBogie", "FrontLeftArm", "BackLeftArm",
    "FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel",
    "BackRightBogie", "FrontRightBogie", "FrontRightArm", "BackRightArm",
    "FrontRightWheel", "MiddleRightWheel", "BackRightWheel"
]

joints = {}

def move_wheels(left_speed, right_speed):
    joints["FrontLeftWheel"].setVelocity(left_speed)
    joints["MiddleLeftWheel"].setVelocity(left_speed)
    joints["BackLeftWheel"].setVelocity(left_speed)
    joints["FrontRightWheel"].setVelocity(right_speed)
    joints["MiddleRightWheel"].setVelocity(right_speed)
    joints["BackRightWheel"].setVelocity(right_speed)

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
        if hasattr(d, 'pose_t') and d.pose_t is not None and len(d.pose_t) == 3:
            tag_dist = abs(float(d.pose_t[2]))
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
    center_screen = width / 2
    tag_x = tag.center[0]
    offset = tag_x - center_screen
    if abs(offset) < tolerance:
        move_wheels(0,0)
        print("Tag centered.")
        return True
    else:
        turn_speed = np.clip(0.004 * offset, -MAX_SPEED, MAX_SPEED)
        move_wheels(-turn_speed, turn_speed)
        robot.step(TIME_STEP)
        return False

def rangefinder_distance(range_finder):
    image = range_finder.getRangeImage()
    if image is not None and len(image) > 0:
        idx = len(image) // 2
        val = float(image[idx])
        if val < 0.03:
            return 100.0
    else:
        val = 100.0
    return val

def turn_to_heading(robot, current_yaw, target_yaw, turn_gain=1800, tol=0.1):
    # returns after executing turn in-place
    dtheta = ((target_yaw - current_yaw + np.pi) % (2 * np.pi)) - np.pi
    if abs(dtheta) < tol:
        move_wheels(0,0)
        return
    steps = int(abs(dtheta) / 0.15) + 1
    sign = np.sign(dtheta)
    for _ in range(steps):
        move_wheels(sign*turn_gain,-sign*turn_gain)
        robot.step(TIME_STEP)
    move_wheels(0,0)
    robot.step(TIME_STEP)

def go_to_waypoint(robot, range_finder, start, goal, heading, use_obstacle_avoidance=True):
    rover_x, rover_y = start
    goal_x, goal_y = goal
    dx = goal_x - rover_x
    dy = goal_y - rover_y
    target_dist = np.hypot(dx, dy)
    target_heading = math.atan2(dx, dy)

    cur_heading = heading
    # align initial
    turn_to_heading(robot, cur_heading, target_heading)
    cur_heading = target_heading

    cycles = 0
    while target_dist > WAYPOINT_TOLERANCE and cycles < 1500:
        # Simple heading control towards goal
        dx = goal_x - rover_x
        dy = goal_y - rover_y
        target_dist = np.hypot(dx, dy)
        target_heading = math.atan2(dx, dy)
        heading_err = ((target_heading - cur_heading + np.pi) % (2 * np.pi)) - np.pi

        if abs(heading_err) > 0.06:
            # align heading in-place, then move
            move_wheels(0,0)
            robot.step(TIME_STEP)
            turn_to_heading(robot, cur_heading, target_heading)
            cur_heading = target_heading

        # If close obstacle, back up and turn to go around (simulate simple avoidance)
        rf_front = rangefinder_distance(range_finder)
        print(f"Waypoint nav: dist={target_dist:.2f}, heading_err={heading_err:.2f}, rf_front={rf_front:.2f}")
        if use_obstacle_avoidance and rf_front < OBSTACLE_DIST:
            print("Obstacle detected! Backing up and turning to avoid.")
            # Backup a bit
            move_wheels(-VELOCITY, -VELOCITY)
            for _ in range(int(SAFE_BACKUP_DIST * 5)): # empirical "backup time"
                robot.step(TIME_STEP)
            move_wheels(0,0)
            # Turn 45 deg right to try to go "around"
            for _ in range(8):
                move_wheels(VELOCITY,-VELOCITY)
                robot.step(TIME_STEP)
            move_wheels(0,0)
            # update heading estimate
            cur_heading -= np.pi/4
            continue

        # Move forward towards goal
        move_wheels(VELOCITY,VELOCITY)
        for _ in range(3):
            robot.step(TIME_STEP)
        # (No perfect odometry, so dead reckoning: just simulate we moved forward)
        # Here, you could update (rover_x, rover_y), but w/o odometry, we just count steps for now.
        # This could be improved if you have wheel encoders.
        rover_x += np.sin(cur_heading) * 0.03  # assume ~3cm per cycle
        rover_y += np.cos(cur_heading) * 0.03
        cycles += 1

    move_wheels(0,0)
    robot.step(TIME_STEP)
    print("Reached waypoint.")

    return (rover_x, rover_y, cur_heading)

def run_robot():
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
    except Exception as e:
        sys.stderr.write("Error: Could not find device 'camera'.\n")
        return

    try:
        range_finder = robot.getDevice("range-finder")
        range_finder.enable(time_step)
    except Exception as e:
        sys.stderr.write("Error: Could not find device 'range-finder'.\n")
        return

    for wheel in ["FrontLeftWheel", "MiddleLeftWheel", "BackLeftWheel",
                  "FrontRightWheel", "MiddleRightWheel", "BackRightWheel"]:
        joints[wheel].setPosition(float('inf'))
        joints[wheel].setVelocity(0.0)

    camera_params = (600, 600, camera.getWidth() / 2, camera.getHeight() / 2)
    at_detector = Detector(families=TAG_FAMILY, nthreads=1, quad_decimate=1.0, refine_edges=1)

    # --- COORDINATE MAPPING STATE ---
    rover_x, rover_y, rover_yaw = 0., 0., 0.   # (x, y, heading(rad)), start at (0,0), heading=0

    # FIND AND MAP FIRST TAG
    found_tag = False
    tag_coord_x, tag_coord_y = None, None
    tag_id = None
    tag_dist = None

    while robot.step(time_step) != -1 and not found_tag:
        image_data = camera.getImage()
        grayscale_frame = webots_to_opencv(image_data, camera.getWidth(), camera.getHeight())
        selected_tag, selected_id, selected_dist = None, None, None
        if grayscale_frame is not None:
            detections = at_detector.detect(
                grayscale_frame,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=TAG_SIZE
            )
            detections = [d for d in detections if d.tag_id in TARGET_TAG_IDS]
            if detections:
                selected_tag, selected_id, selected_dist = select_nearest_tag(detections, camera_params, tag_size=TAG_SIZE)

        if selected_tag:
            centered = rotate_to_center(robot, selected_tag, camera)
            if not centered:
                continue
            # Get tag distance from range finder
            measured_dist = rangefinder_distance(range_finder)
            tag_coord_x = rover_x + measured_dist * math.sin(rover_yaw)
            tag_coord_y = rover_y + measured_dist * math.cos(rover_yaw)
            tag_id = selected_id
            tag_dist = measured_dist

            print(f"\n---Starting Tag info---\n"
                  f"Rover at (x={rover_x:.2f}, y={rover_y:.2f}), heading={rover_yaw:.2f}rad.\n"
                  f"First tag_id: {tag_id}, tag_coord=({tag_coord_x:.2f},{tag_coord_y:.2f}), measured_dist={measured_dist:.2f}m\n")
            found_tag = True

            # Calculate waypoint: 1 meter in front of the tag (in rover's reference)
            if tag_id == 0:
                goal_dist = measured_dist - 1.5
            else:
                goal_dist = measured_dist - 1.0
            goal_x = rover_x + goal_dist * math.sin(rover_yaw)
            goal_y = rover_y + goal_dist * math.cos(rover_yaw)
            print(f"Navigation goal: 1.0m in front of tag: ({goal_x:.2f}, {goal_y:.2f})\n")
        else:
            wheels_straight()
            move_wheels(VELOCITY * 0.3, -VELOCITY * 0.3)
            print("No tag found. Rotating to search...")

    # --- NAVIGATE TO TARGET COORD ---
    rover_x, rover_y, rover_yaw = go_to_waypoint(
        robot, range_finder, (rover_x, rover_y), (goal_x, goal_y), rover_yaw,
        use_obstacle_avoidance=True
    )

    # --- Alignment at Goal: Turn 180 from approach heading ---
    print("At goal, aligning 180deg from initial approach direction...")
    new_yaw = (rover_yaw + np.pi) % (2*np.pi)
    turn_to_heading(robot, rover_yaw, new_yaw)
    rover_yaw = new_yaw

    # --- Perform tag-based action (right/left/stop) ---
    if tag_id == 0:
        print("ID 0: STOPPING at 1.5m before tag, facing opposite direction.")
        move_wheels(0, 0)
        robot.step(TIME_STEP)
    elif tag_id == 1:
        print("ID 1: Performing 90 deg right turn.")
        turn_to_heading(robot, rover_yaw, (rover_yaw - np.pi/2) % (2*np.pi))
    elif tag_id == 2:
        print("ID 2: Performing 90 deg left turn.")
        turn_to_heading(robot, rover_yaw, (rover_yaw + np.pi/2) % (2*np.pi))

    print("Mission step complete (1 tag demo). Extend logic for next tag if needed.")
    move_wheels(0, 0)

if __name__ == '__main__':
    run_robot()