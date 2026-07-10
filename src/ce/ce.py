import time
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

config = SOFollowerRobotConfig(
    port="/dev/ttyACM0",
    id="my_awesome_follower_arm",
)

with SO101Follower(config) as robot:
    action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 10,
    }
    robot.send_action(action)
    
    # 关键：等待机械臂运动到位（2~3秒）
    time.sleep(3)
    
    obs = robot.get_observation()
    print(obs["shoulder_pan.pos"])