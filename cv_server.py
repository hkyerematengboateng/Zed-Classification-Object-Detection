from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from typing import List, Dict
app = FastAPI()
ws_connected = False

class ConnectionManager:
    def __init__(self):
        self.active_connections = Dict[str, List[WebSocket]] = {}

    async def connect(self, topic:str, websocket: WebSocket):
        await websocket.accept()
        if topic not in self.active_connections:
            self.active_connections[topic] = []
        self.active_connections[topic].append(websocket)

    def disconnect(self,topic:str, websocket: WebSocket):
        if topic in self.active_connections:
            self.active_connections[topic].remove(websocket)

    async def broadcast(self, topic:str, message: str):
        if topic in self.active_connections:
            for connection in self.active_connections[topic]:
                await connection.send_text(message)

ws_connection = ConnectionManager()

@app.post("/start")
def start_camera():
    pass

@app.post("/stop")
def stop_camera():
    pass