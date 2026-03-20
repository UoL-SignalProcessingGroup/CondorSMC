import logging
import random
import time

import pymysql

"""
TO DO
    - Add docstrings
"""


class PyMySQLConnector:
    def __init__(self, host, user, password, database, poll_delay=5, max_retries=5):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.poll_delay = poll_delay
        self.max_retries = max_retries
        self.connection = None
        self.cursor = None
        self.query_queue = []

    def connect(self, limit=20):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            self.cursor = self.connection.cursor()
            logging.info("Connected to MySQL database")
        except pymysql.Error as error:
            error_code = error.args[0]
            sleep_time = random.randint(5, max(20, self.poll_delay))

            if error_code == 1040:
                logging.error(
                    f"Waiting {sleep_time} seconds before retrying connect {limit} attempts left..."
                )

                if limit > 1:
                    self.connection = None
                    time.sleep(sleep_time)
                    return self.connect(limit - 1)

            if "polling too quickly" in str(error):
                if limit > 1:
                    self.connection = None
                    time.sleep(sleep_time)
                    return self.connect(limit - 1)

            raise RuntimeError(f"""
                                    Exceeded maximum number of retries for query queue.
                                    Error: {error}
                                    Query Queue: {self.query_queue}
                                """)

            return False

    def disconnect(self):
        if self.connection is not None:
            self.cursor.close()
            self.connection.close()
            self.connection = None
            logging.info("Disconnected from MySQL database")

    def insert(self, table, columns, values, limit=10, queue_query=False):
        query = f"INSERT INTO {table} {columns} VALUES {values}"
        if queue_query:
            self.query_queue.append(("insert", query))
            return True
        res = self._execute_query(query, limit=limit)
        return res

    def select(
        self,
        table,
        columns,
        where_clause,
        orderby=None,
        limit=10,
        queue_query=False,
    ):
        query = f"SELECT {columns} FROM {table} WHERE {where_clause}"
        if orderby is not None:
            query += f" ORDER BY {orderby}"
        if queue_query:
            self.query_queue.append(("select", query))
            return True
        res = self._execute_query(query, limit=limit, select=True)
        return res

    def select_one(
        self,
        table,
        columns,
        where_clause,
        orderby=None,
        limit=10,
        queue_query=False,
    ):
        query = f"SELECT {columns} FROM {table} WHERE {where_clause}"
        if orderby is not None:
            query += f" ORDER BY {orderby}"
        if queue_query:
            self.query_queue.append(("select_one", query))
            return True
        res = self._execute_query(query, limit=limit, select_one=True)
        return res

    def update(
        self, table, set_values, where_clause, limit=10, queue_query=False
    ):
        query = f"UPDATE {table} SET {set_values} WHERE {where_clause}"
        if queue_query:
            self.query_queue.append(("update", query))
            return True
        res = self._execute_query(query, limit=limit)
        return res

    def delete(self, table, where_clause, limit=10, queue_query=False):
        query = f"DELETE FROM {table} WHERE {where_clause}"
        if queue_query:
            self.query_queue.append(("delete", query))
            return True
        res = self._execute_query(query, limit=limit)
        return res

    def _execute_query(self, query, select=False, select_one=False, limit=10):
        try:
            if select:
                self.connect()
                self.cursor.execute(query)
                res = self.cursor.fetchall()
                self.disconnect()
                if not res:
                    return None
                return res
            elif select_one:
                self.connect()
                self.cursor.execute(query)
                res = self.cursor.fetchone()
                self.disconnect()
                if not res:
                    return None
                return res
            else:
                self.connect()
                self.cursor.execute(query)
                self.connection.commit()
                self.disconnect()
                return True
        except pymysql.Error as error:
            error_code = error.args[0]
            sleep_time = random.randint(1, max(2, self.poll_delay))

            if error_code == 1040:
                if limit > 1:
                    self.connection = None
                    time.sleep(sleep_time)
                    return self._execute_query(query_queue, select, select_one, limit - 1)

            if "polling too quickly" in str(error):
                if limit > 1:
                    self.connection = None
                    time.sleep(sleep_time)
                    return self._execute_query(query_queue, select, select_one, limit - 1)

            raise RuntimeError(f"""
                                    Exceeded maximum number of retries for query queue.
                                    Error: {error}
                                    Query Queue: {self.query_queue}
                                """)

            return False

    def _execute_query_queue(self, limit=10):
        self.connect()

        try:
            results = []

            for query in self.query_queue:
                if query[0] == "select":
                    self.cursor.execute(query[1])
                    res = self.cursor.fetchall()
                    if not res:
                        results.append(None)
                    else:
                        results.append(res)
                elif query[0] == "select_one":
                    self.cursor.execute(query[1])
                    res = self.cursor.fetchone()
                    if not res:
                        results.append(None)
                    else:
                        results.append(res)
                else:
                    self.cursor.execute(query[1])
                    self.connection.commit()
                    results.append(True)

            self.disconnect()
            self.query_queue = []
            return results
        except pymysql.Error as error:
            error_code = error.args[0]
            sleep_time = random.randint(1, max(2, self.poll_delay))

            if error_code == 1040:
                logging.error(
                    f"Waiting {sleep_time} seconds before retrying {limit} attempts left..."
                )

                if limit > 1:
                    self.connection = None
                    time.sleep(sleep_time)
                    return self._execute_query(query_queue, select, select_one, limit - 1)

            if "polling too quickly" in str(error):
                if limit > 1:
                    time.sleep(sleep_time)
                    return self._execute_query(query_queue, select, select_one, limit - 1)

            raise RuntimeError(f"""
                                    Exceeded maximum number of retries for query queue.
                                    Error: {error}
                                    Query Queue: {self.query_queue}
                                """)

            return False
