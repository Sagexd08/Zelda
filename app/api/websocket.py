
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import json
import base64
from datetime import datetime

from app.core.config import settings
from app.core.database import db_manager
from app.models.face_detector import get_face_detector
from app.models.face_aligner import get_face_aligner
from app.models.embedding_extractor import get_embedding_extractor, cosine_similarity
from app.models.fusion_model import get_fusion_model
from app.models.liveness_detector import get_liveness_detector
from app.models.temporal_liveness import get_temporal_liveness_detector
from app.core.security import security_manager
from app.core.database import User, Embedding

class WebSocketConnectionManager:

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

        self.face_detector = get_face_detector()
        self.face_aligner = get_face_aligner()
        self.embedding_extractor = get_embedding_extractor()
        self.fusion_model = get_fusion_model()
        self.liveness_detector = get_liveness_detector()
        self.temporal_liveness = get_temporal_liveness_detector()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)

    async def process_frame(
        self,
        websocket: WebSocket,
        frame_data: str,
        user_id: str,
        session_id: str
    ):
        try:
            frame = self._decode_frame(frame_data)

            if frame is None:
                await self.send_message(websocket, {
                    'status': 'error',
                    'message': 'Invalid frame data'
                })
                return

            detection = self.face_detector.detect_largest(frame)

            if detection is None:
                await self.send_message(websocket, {
                    'status': 'no_face',
                    'message': 'No face detected'
                })
                return

            await self.send_message(websocket, {
                'status': 'face_detected',
                'confidence': float(detection.confidence),
                'bbox': detection.bbox.tolist()
            })

            if settings.ENABLE_TEMPORAL_LIVENESS:
                self.temporal_liveness.add_frame(frame, detection.landmarks)

            face_region = frame[
                int(detection.bbox[1]):int(detection.bbox[3]),
                int(detection.bbox[0]):int(detection.bbox[2])
            ]
            is_live, liveness_score = self.liveness_detector.predict(face_region)

            if not is_live:
                await self.send_message(websocket, {
                    'status': 'liveness_failed',
                    'liveness_score': float(liveness_score),
                    'message': 'Liveness check failed'
                })
                return

            await self.send_message(websocket, {
                'status': 'liveness_passed',
                'liveness_score': float(liveness_score)
            })

            temporal_live = True
            temporal_confidence = 1.0

            if settings.ENABLE_TEMPORAL_LIVENESS:
                if len(self.temporal_liveness.frame_buffer) >= settings.MIN_VIDEO_FRAMES:
                    temporal_live, temporal_confidence = self.temporal_liveness.predict()

                    await self.send_message(websocket, {
                        'status': 'temporal_liveness',
                        'temporal_live': temporal_live,
                        'temporal_confidence': float(temporal_confidence)
                    })

            if not temporal_live:
                await self.send_message(websocket, {
                    'status': 'temporal_liveness_failed',
                    'message': 'Temporal liveness check failed'
                })
                return

            face_160 = self.face_aligner.align(frame, detection, output_size=160)
            face_224 = self.face_aligner.align(frame, detection, output_size=224)

            test_embeddings = self.embedding_extractor.extract_all_embeddings(face_160, face_224)

            if not test_embeddings:
                await self.send_message(websocket, {
                    'status': 'error',
                    'message': 'Failed to extract embeddings'
                })
                return

            test_fusion = self.fusion_model.fuse_embeddings(test_embeddings)
            test_embeddings['fusion'] = test_fusion

            db = db_manager.get_session()
            try:
                user = db.query(User).filter(User.user_id == user_id).first()

                if not user:
                    await self.send_message(websocket, {
                        'status': 'error',
                        'message': 'User not found'
                    })
                    return

                stored_embedding = db.query(Embedding).filter(
                    Embedding.user_id == user.id,
                    Embedding.is_primary == True
                ).first()

                if not stored_embedding:
                    await self.send_message(websocket, {
                        'status': 'error',
                        'message': 'No enrolled embeddings'
                    })
                    return

                stored_embeddings = {}
                if stored_embedding.fusion_embedding:
                    stored_embeddings['fusion'] = security_manager.decrypt_embedding(
                        stored_embedding.fusion_embedding
                    )

                if 'fusion' in stored_embeddings:
                    similarity = cosine_similarity(
                        test_embeddings['fusion'],
                        stored_embeddings['fusion']
                    )
                else:
                    similarity = 0.0

                threshold = user.custom_threshold if user.custom_threshold else settings.VERIFICATION_THRESHOLD

                authenticated = similarity >= threshold

                await self.send_message(websocket, {
                    'status': 'authenticated' if authenticated else 'not_authenticated',
                    'authenticated': authenticated,
                    'confidence': float(similarity),
                    'threshold': float(threshold),
                    'liveness_score': float(liveness_score),
                    'temporal_confidence': float(temporal_confidence)
                })

            finally:
                db.close()

        except Exception as e:
            print(f"Error processing frame: {e}")
            await self.send_message(websocket, {
                'status': 'error',
                'message': str(e)
            })

    def _decode_frame(self, frame_data: str) -> np.ndarray:
        try:
            if 'base64,' in frame_data:
                frame_data = frame_data.split('base64,')[1]

            img_bytes = base64.b64decode(frame_data)

            nparr = np.frombuffer(img_bytes, np.uint8)

            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return frame

        except Exception as e:
            print(f"Frame decode error: {e}")
            return None

connection_manager = WebSocketConnectionManager()

async def websocket_endpoint(websocket: WebSocket, client_id: str):
    if not settings.WEBSOCKET_ENABLED:
        await websocket.close(code=1008, reason="WebSocket disabled")
        return

    await connection_manager.connect(websocket, client_id)

    connection_manager.temporal_liveness.reset()

    try:
        while True:
            data = await websocket.receive_json()

            message_type = data.get('type')

            if message_type == 'ping':
                await connection_manager.send_message(websocket, {
                    'type': 'pong',
                    'timestamp': datetime.utcnow().isoformat()
                })

            elif message_type == 'frame':
                frame_data = data.get('frame')
                user_id = data.get('user_id')
                session_id = data.get('session_id', 'default')

                if not frame_data or not user_id:
                    await connection_manager.send_message(websocket, {
                        'status': 'error',
                        'message': 'Missing frame or user_id'
                    })
                    continue

                await connection_manager.process_frame(
                    websocket, frame_data, user_id, session_id
                )

            elif message_type == 'reset':
                connection_manager.temporal_liveness.reset()
                await connection_manager.send_message(websocket, {
                    'status': 'reset',
                    'message': 'Temporal buffer reset'
                })

            else:
                await connection_manager.send_message(websocket, {
                    'status': 'error',
                    'message': f'Unknown message type: {message_type}'
                })

    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)

    except Exception as e:
        print(f"WebSocket error: {e}")
        connection_manager.disconnect(client_id)
