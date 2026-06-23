from std_msgs.msg import String
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
import os
from mistralai.client import Mistral
from cv_bridge import CvBridge
import base64
import cv2

# VLM api stuff
key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"


class cameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_sub')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/head_front_camera/rgb/image_raw',
            self.listener_callback,
            10)
        self.client = Mistral(api_key = key)
        self.latest_img = None

        timer_period = 10.0
        self.time = self.create_timer(timer_period, self.timer_callback)


        self.subscription

    def listener_callback(self, img):
        self.latest_img = img

    def timer_callback(self):

        if self.latest_img is not None:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")

            success, encoded_image = cv2.imencode('.jpg', cv_image)

            b64_img = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image very concise"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{b64_img}"
                        }
                    ]
                }
            ]
            response = self.client.chat.complete(
                model =  "ministral-3b-2512",
                messages=message
            )

            response_text = response.choices[0].message.content
            self.get_logger().info(f"Mistral: {response_text}")
        else:
            self.get_logger().warning('No image yet')

def main(args=None):
    rclpy.init(args=args)
    camera_sub = cameraSubscriber()
    rclpy.spin(camera_sub)

    camera_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()