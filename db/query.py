import os
from argparse import ArgumentParser, ArgumentTypeError
import psycopg2
from db.query_strings import *


class DatabaseQuery:
    def __init__(self, dbURL):
        self.dbURL = dbURL
        self._connect()

    def execute(self, command, commit=False, ret=False):
        """
        Execute a SQL command on the database.
        """
        if self.conn is None:
            raise ValueError("No connection established to the database")

        try:
            self.cur.execute(command)

            if commit:
                self.conn.commit()

            if ret:
                res = self.cur.fetchall()
                return res
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.close()

    def _connect(self):
        """
        Connect to the PostgreSQL database server
        """
        self.conn = None
        try:
            # Connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            self.conn = psycopg2.connect(self.dbURL)
            self.cur = self.conn.cursor()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            if self.conn is not None:
                self.close()

    def close(self):
        """
        Close the connection to the database
        """
        if self.conn is not None:
            # Close the communication with the PostgreSQL
            print("Closing database connection")
            self.conn.close()
            self.conn = None


class FilterNetQuery(DatabaseQuery):
    def __init__(self, dbURL):
        super().__init__(dbURL)

    def status(self):
        db_version = self.execute(VERSION, ret=True)
        if len(db_version) > 0:
            return 'PostgreSQL database version:' + db_version[0][0]

    def create_tables(self):
        self.execute(CREATE_TABLE, commit=True)




def parse_args():
    def valid_file(filepath):
        if os.path.exists(filepath):
            return os.path.normpath(filepath)
        else:
            raise ArgumentTypeError("Invalid file: {}".format(filepath))

    parser = ArgumentParser(
        prog="FilterNet db util",
        description="Postgres database management for FilterNet")

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    st = subparsers.add_parser(
        "status",
        description="Print current status of database")
    st.set_defaults(command="status")

    tb = subparsers.add_parser(
        "table",
        description="Create table structure in database")
    tb.set_defaults(command="table")

    return parser.parse_args()


ARGS = parse_args()


def main():
    dbURI = os.environ["DATABASE_URL"]

    query = FilterNetQuery(dbURI)
    if query.conn is None:
        exit(1)

    try:
        if ARGS.command == 'status':
            print(query.status())

        if ARGS.command == 'table':
            query.create_tables()

    except ValueError:
        print("ERROR: Database connection.")

    finally:
        if query is not None:
            query.close()
            print("Database connection closed")


if __name__ == "__main__":
    main()
