import os
import psycopg2

from argparse import ArgumentParser
from .query_strings import *


def parse_args():
    parser = ArgumentParser(
        prog="FilterNet db util",
        description="Postgres database management for FilterNet")

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "command"

    tb = subparsers.add_parser(
        "table",
        description="Create table structure in database")
    tb.set_defaults(command="table")

    ins = subparsers.add_parser(
        "insert",
        description="Insert checkpoint directory in database")
    ins.add_argument(
        "--generator",
        default='model_weights/generator.h5',
        help="Generator weights filepath")
    ins.add_argument(
        "--discriminator",
        default='model_weights/discriminator.h5',
        help="Discriminator weights filepath")
    ins.set_defaults(command="insert")

    dl = subparsers.add_parser(
        "delete")
    dl.add_argument(
        "--id", required=True,
        help="ID of model to delete from database")

    dl.set_defaults(command="delete")

    return parser.parse_args()


class DatabaseQuery:
    def __init__(self, dbURL):
        self.dbURL = dbURL

    def execute(self, command, args=None, commit=False, ret=False):
        """
        Executes given SQL command.
        """
        conn = None
        try:
            # Connect to the Postgres database
            print('Connecting to the PostgreSQL database')
            conn = psycopg2.connect(self.dbURL)
            # Create a new cursor object
            cur = conn.cursor()
            print('Executing command')
            cur.execute(command, args)

            if commit:
                conn.commit()

            if ret:
                res = cur.fetchone()
                return res, True
            return True

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            return False

        finally:
            if conn is not None:
                print('Closing connection')
                conn.close()


class FilterNetQuery(DatabaseQuery):
    def __init__(self, dbURL):
        super().__init__(dbURL)

    def create_tables(self):
        self.execute(CREATE_TABLE, commit=True)

    def insert_model(self, gen_path, disc_path):
        gen = open(gen_path, 'rb').read()
        disc = open(disc_path, 'rb').read()
        return self.execute(INSERT_MODEL,
                            (psycopg2.Binary(gen), psycopg2.Binary(disc),),
                            commit=True)

    def read_model(self, id):
        return self.execute(SELECT_MODEL, (id,), ret=True)

    def delete_model(self, id):
        return self.execute(DELETE_MODEL, (id,), commit=True, ret=True)


def main():
    ARGS = parse_args()

    dbURI = os.environ["DATABASE_URL"]
    query = FilterNetQuery(dbURI)

    if ARGS.command == 'table':
        query.create_tables()
        print("Tables created")

    elif ARGS.command == 'insert':
        query.insert_model(ARGS.generator, ARGS.discriminator)
        print("Checkpoint directory inserted")

    elif ARGS.command == 'delete':
        res = query.delete_model(ARGS.id)
        print(res)
        print("Model deleted")
