import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import redis
import json
import threading

class RedisToROS2Bridge(Node):
    def __init__(self):
        super().__init__('redis_to_ros2_bridge')
        self.publisher_ = self.create_publisher(String, 'redis_messages', 10)
        
        # Connect to Redis
        self.redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)  # 'redis' is the service name in docker-compose
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('object_detection')

        # Start Redis listener in a separate thread
        self.redis_thread = threading.Thread(target=self.redis_listener)
        self.redis_thread.start()

    def redis_listener(self):
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'].decode('utf-8'))
                ros_msg = String()
                ros_msg.data = json.dumps(data)
                self.publisher_.publish(ros_msg)
                self.get_logger().info(f'Published to ROS2: {ros_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    bridge = RedisToROS2Bridge()
    rclpy.spin(bridge)
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()