from google.cloud import datastore


class DatastoreQuery:
    def __init__(self):
        self.client = datastore.Client()
        self.key = self.client.key('EntityKind', 'images')

    def _get_latest_id(self):
        return self.client.get(self.key)["active"]

    def increment_latest_id(self):
        cur = self._get_latest_id()
        entity = datastore.Entity(key=self.key)
        entity.update({
            'active': cur+1
        })
        self.client.put(entity)
        return cur
