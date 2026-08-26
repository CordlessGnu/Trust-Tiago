import base64
import cv2
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import requests
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TiagoSmolVLALoop(Node):
  def __init__(self):
    super().__init__('tiago_smolvla_loop')
    self.bridge = CvBridge()

    self.latest_top = None
    self.latest_head = None
    self.latest_side = None
    self.current_joints = {}

    # Tiagos arm joints
    self.arm_joint_names = [
        'arm_1_joint',
        'arm_2_joint',
        'arm_3_joint',
        'arm_4_joint',
        'arm_5_joint',
        'arm_6_joint',
        'arm_7_joint',
    ]

    self.gripper_joint_names = ['gripper_right_finger_joint', 'gripper_left_finger_joint']

    # Subscribing to relevant topics for the cameras
    self.create_subscription(
        JointState, '/joint_states', self.joint_cb, 10
    )
    self.create_subscription(
        Image,
        '/overhead_camera/overhead_camera_sensor/image_raw', self.top_cb, 10,
    )
    self.create_subscription(
        Image, '/side_camera/scene_camera_sensor/image_raw', self.side_cb, 10
    )
    self.create_subscription(
        Image, '/head_front_camera/rgb/image_raw', self.head_cb, 10
    )

    # Creating publishers for outputting movements to controllers for execution
    self.arm_pub = self.create_publisher(
        JointTrajectory, '/arm_controller/joint_trajectory', 10
    )
    self.gripper_pub = self.create_publisher(
        JointTrajectory, '/gripper_controller/joint_trajectory', 10
    )

    # Set timer interval
    self.timer_period = 0.5
    self.create_timer(self.timer_period, self.control_loop)

    # api url for the lerobot server
    self.api_url = ''

  # Callbacks for processing img msgs in ros format to OpenCV
  def top_cb(self, msg):
    self.latest_top = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
  def head_cb(self, msg):
    self.latest_head = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
  def side_cb(self, msg):
    self.latest_side = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

  # Processing joint messages and pairing values to joint names
  def joint_cb(self, msg):
    for name, pos in zip(msg.name, msg.position):
      self.current_joints[name] = pos

  # Conversion of image opencv format to base 64
  def cv_to_b64(self, img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

  # Function to get the right joints to match so100 output of smolvla to tiago's
  def get_6dof_so100_state(self):
    return [
        # shoulder_pan
        self.current_joints.get('arm_1_joint', 0.0),
        # shoulder_lift
        self.current_joints.get('arm_2_joint', 0.0),
        # elbow_flex  
        self.current_joints.get('arm_4_joint', 0.0), 
        # wrist_flex 
        self.current_joints.get('arm_6_joint', 0.0),
        # wrist_roll
        self.current_joints.get('arm_7_joint', 0.0), 
        # right gripper finger
        self.current_joints.get('gripper_right_finger_joint', 0.0),
    ]

  def publish_trajectory(
      self, publisher, joint_names: list[str], positions: list[float]
  ):
    msg = JointTrajectory()
    msg.joint_names = joint_names

    point = JointTrajectoryPoint()
    point.positions = positions

    sec = int(self.timer_period)
    nanosec = int((self.timer_period - sec) * 1e9)
    point.time_from_start = Duration(sec=sec, nanosec=nanosec)

    msg.points = [point]
    publisher.publish(msg)

  def control_loop(self):
    if (
        self.latest_top is None
        or self.latest_side is None
        or self.latest_head is None
        or not self.current_joints
    ):
      return

    # Getting robot joint states in 6dof format
    so100_states = self.get_6dof_so100_state()

    payload = {
        'image_top64': self.cv_to_b64(self.latest_top),
        'image_head64': self.cv_to_b64(self.latest_head),
        'image_side64': self.cv_to_b64(self.latest_side),
        'joint_states': so100_states,
        # 'task': 'pick up the object',
    }

    try:
      res = requests.post(self.api_url, json=payload, timeout=1.0)
      if res.status_code == 200:
        action = res.json().get('action', [])

        if not action:
          self.get_logger().info('No action received from the API.')
          return
        
        tiago_arm_action = [
            action[0],  # arm_1_joint
            action[1],  # arm_2_joint
            self.current_joints.get('arm_3_joint', 0.0),  # arm_3_joint (held)
            action[2],  # arm_4_joint
            self.current_joints.get('arm_5_joint', 0.0),  # arm_5_joint (held)
            action[3],
            action[4],
            # self.current_joints.get('arm_6_joint', 0.0),  # arm_6_joint (held)
            # self.current_joints.get('arm_7_joint', 0.0),  # arm_7_joint (held)
        ]

        tiago_gripper_action = [action[5], action[5]]  # gripper_finger_joint

        # Publish the actions to controllers
        self.publish_trajectory(
            self.arm_pub, self.arm_joint_names, tiago_arm_action
        )
        self.publish_trajectory(
            self.gripper_pub, self.gripper_joint_names, tiago_gripper_action
        )

    except Exception as e:
      self.get_logger().error(f'Inference failed: {e}')


def main():
  rclpy.init()
  node = TiagoSmolVLALoop()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()