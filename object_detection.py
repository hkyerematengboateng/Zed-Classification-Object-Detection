import redis
from flask import Flask, jsonify
import json
from flask_socketio import SocketIO, disconnect
import threading
from flask_cors import CORS
import logging
app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins="http://localhost:3000")
CORS(app, resources={"*": {"origins": "http://localhost:3000"}})

connector = None

class Redis_Listener(threading.Thread):
    def __init__(self,state=True):
        super.__init__()
        self.state = state
    
    def run(self):
        while self.state:
            pass
class RedisConnector:
    def __init__(self, socketio=None):
        self.server = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.channel = "object_detection"
        self.socket = socketio
        self.listen_channel = threading.Event()
    def subscribe(self, channel:str = None):
        if channel is not None:
            self.channel = channel
        self.pubsub = self.server.pubsub()
        self.pubsub.subscribe(self.channel)
            # Start Redis listener in a separate thread
        self.redis_thread = threading.Thread(target=self.redis_listener)
        self.redis_thread.start()

    def stop_listener(self, listen=False):
        self.listen_channel.set()
        print('Here',self.listen_channel.is_set())
        self.redis_thread.join(1)
    def redis_listener(self):
        print('Here',self.listen_channel.is_set())
        while not self.listen_channel.is_set():
            print('Listening')
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    data = json.loads(message['data'].decode('utf-8'))
                    socketio.emit("message", data)
                    logging.info(f'Published to ROS2: {data}')
        
        print('Stopped')

socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
@app.route("/")
def hello_world():
    return {"status":"working",
            "app_name": "Zed Object Detection Middleware server"}

@app.route('/stop',methods=['GET'])
def stop_streaming():
    global connector
    connector.stop_listener()
    disconnect()
    return {'status': 'listening stopped'}

@app.route("/start", methods=['POST','GET'])
def detection_data():
    global connector
    connector = RedisConnector(socketio)
    connector.subscribe()
    socketio.emit("connection","Streaming starting")
    return "test"


if __name__ == '__main__':
    socketio.run(app)