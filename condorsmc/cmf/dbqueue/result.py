import json
import logging
import uuid
from time import time

from . import utils


class Result:
    def __init__(
        self,
        db,
        session_id: str,
        node_id: str,
        role: int,
        results_id: str = None,
        attributes=json.dumps({}),
        payload=json.dumps({}),
    ):
        self.db = db
        self.session_id = session_id
        self.node_id = node_id
        self.role = role
        self.results_id = str(uuid.uuid4()) if results_id is None else results_id
        self.attributes = attributes
        self.payload = payload

    def insert(self):
        res = self.db.insert(
            "results",
            "(`session_id`, `node_id`, `role`, `job_id`, `attributes`, `payload`, `created_at`)",
            (self.session_id, self.node_id, self.role, self.results_id, json.dumps(self.attributes, cls=utils.NpEncoder), json.dumps(self.payload, cls=utils.NpEncoder), time()),
        )
        return res

    def fetch(self):
        results = self.db.select(
            "results",
            "*",
            "WHERE `results_id` = %s",
        )

    def delete(self):
        self.db.delete("results", "results_id", self.results_id)