##########
# Author: Chaolun Xia, 2013-Jan-09#
#
# A basic and private interface to connect and test the mongodb
#
##########
#Edited by: (Please write your name here)#
import types
import pymongo
import time

class MongoDBInterface(object):
    #A basic interface#

    def __init__(self,
                 address='grande.rutgers.edu',
                 port=27018):
        try:
            self._connection = pymongo.Connection(address, port)
        except Exception, e:
            print 'cannot connected to the mongodb at (%s, %d) in %d seconds' % (address,port)
            assert 1 == 2

    def setDB(self, name):
        self._db = self._connection[name]

    def setCollection(self, name):
        self._collection = self._db[name]

    def saveDocument(self, document):
        # document must be a json or a class from {event, photo, prediction}
        if not type(document) is types.DictType:
            document = document.toDict()
        self._collection.save(document)

    def _deleteDocument(self, condition):
        assert condition is not None
        self._collection.remove(condition)

    def getOneDocument(self, condition={}):
        return self._collection.find_one(condition)

    def getOneDocumentWithMostValue(self, field, direction, condition={}):
        # direct = -1 means take tha largest, 1 means take the smallest
        assert direction in [1, -1]
        assert type(field) is types.StringType
        return self._collection.find_one(condition, sort=[(field, direction)])

    def getAllDocuments(self, condition={},
                        fields_to_select=[],
                        limit=0, sorting_info=[]):
        # sorting_info = [(field_1, direct_1), (field_2, direct_2)]
        # empty dict will not affect
        fields_filter = {}
        if type(fields_to_select) is types.StringType:
            fields_filter[fields_to_select] = True
        else:
            assert type(fields_to_select) is types.ListType
            for field in fields_to_select:
                fields_filter[field] = True

        if len(sorting_info) > 0:
            #assert len(sorting_info) == 2
            #assert type(sorting_info[0]) is types.StringType
            #assert sorting_info[1] in [-1, 1]
            # not appear, or appear as true
            assert fields_filter.get(sorting_info[0][0], True) is True

        # direct = -1 means take tha largest, 1 means take the smallest
        if len(fields_filter) == 0:
            return self._collection.find(condition, timeout=False, limit=limit, sort=sorting_info)
        else:
            return self._collection.find(condition, fields_filter, timeout=False, limit=limit, sort=sorting_info)

    def updateDocument(self, document):
        if not type(document) is types.DictionaryType:
            document = document.toDict()
        self._collection.update({'_id': document['_id']}, document, True)

if __name__ == '__main__':
    pass
