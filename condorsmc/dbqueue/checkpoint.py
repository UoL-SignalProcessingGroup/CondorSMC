import json
import logging
import numpy as np
from time import time

from . import utils


class Checkpoint:
    def __init__(self, db, session_id: str, node_id: str, node_type: int):
        self.db = db
        self.session_id = session_id
        self.node_id = node_id
        self.node_type = node_type

    def exists(self, type):
        _exists = self.db.select(
            "checkpoint",
            "id",
            f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}' AND type = {type}",
        )
        if _exists:
            return True
        return False

    def create(self, type, payload):
        logging.info(f"Creating checkpoint for session {self.session_id}")
        payload = self._encode_payload(payload)
        _created = self.db.insert(
            "checkpoint",
            "(`session_id`, `node_id`, `node_type`, `created_at`, `type`, `payload`)",
            (
                self.session_id,
                self.node_id,
                self.node_type,
                time(),
                type,
                json.dumps(payload, cls=utils.NpEncoder),
            ),
        )
        if _created:
            logging.info(f"Checkpoint created for session {self.session_id}")
            return True
        return False

    def get(self, type):
        _checkpoint = self.db.select_one(
            "checkpoint",
            "payload",
            f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}' AND type = {type}",
        )
        if _checkpoint:
            payload = json.loads(_checkpoint[0])
            return self._decode_payload(payload)
        return None

    def update(self, type, payload):
        logging.info(f"Updating checkpoint for session {self.session_id}")
        payload = self._encode_payload(payload)
        _updated = self.db.update(
            "checkpoint",
            f"payload = '{json.dumps(payload, cls=utils.NpEncoder)}'",
            f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}' AND type = {type}",
        )
        if _updated:
            logging.info(f"Checkpoint updated for session {self.session_id}")
            return True
        return False

    def delete(self, type):
        logging.info(f"Deleting checkpoint for session {self.session_id}")
        _deleted = self.db.delete(
            "checkpoint",
            f"session_id = '{self.session_id}' AND `node_id` = '{self.node_id}' AND type = {type}",
        )
        if _deleted:
            logging.info(f"Checkpoint deleted for session {self.session_id}")
            return True
        return False

    def _encode_payload(self, payload):
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()

        for key, value in payload.items():            
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, list):
                        for j, vv in enumerate(v):
                            if np.isinf(vv) or np.isnan(vv):
                                payload[key][i][j] = None
                    else:
                        if np.isinf(v) or np.isnan(v):
                            payload[key][i] = None
            else:
                print(type(value))
                if np.isinf(value) or np.isnan(value):
                    payload[key] = None
        
        return payload

    def _decode_payload(self, payload):
        for key, value in payload.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, list):
                        for j, vv in enumerate(v):
                            if vv is None:
                                payload[key][i][j] = -np.inf
                    else:
                        if v is None:
                            payload[key][i] = -np.inf
            else:
                if value is None:
                    payload[key] = -np.inf

        for key, value in payload.items():
            if isinstance(value, list):
                payload[key] = np.array(value)

        return payload
