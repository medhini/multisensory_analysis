import json

ONTOLOGY_FILE = "ontology.json"

class Ontology:
    def __init__(self):
        records = []
        with open(ONTOLOGY_FILE) as f:
            records = json.load(f)
        
        self.record_for_id = {record["id"]: record for record in records}
        self.parent_id_for_id = {}
        for record in records:
            for child_id in record["child_ids"]:
                parent_for_id[child_id] = record["id"]

    def get_record_for_id(self, id):
        return self.record_for_id[id]

    def get_most_general_id(self, id):
        answer_id = id
        while answer_id in self.parent_id_for_id:
            answer_id = self.parent_id_for_id[answer_id]
        return answer_id

def get_records(whitelist):
    """Gets AudioSet ontology details for names of sounds that interest the programmer.
    
    Arguments:
        whitelist {string[]} -- names of sounds that interest the programmer

    Returns:
        {dict[]} -- records for the names in whitelist.  Might be longer than whitelist, as some sounds have child
        sounds that are more specific.
    
    Raises:
        IOError -- if the ontology file cannot be opened
        ValueError -- if a sound in the whitelist is not in AudioSet
    """
    ontology = {}
    with open(ONTOLOGY_FILE) as f:
        ontology = json.load(f)

    # 1. Construct needed dicts
    ids_for_name = {}
    index_for_id = {}
    for (i, record) in enumerate(ontology):
        name = record["name"]
        id = record["id"]
        index_for_id[id] = i
        if name in ids_for_name:
            ids_for_name[name].append(id)
        else:
            ids_for_name[name] = [id]
    
    # Get all ids corresponding to the names in the whitelist
    whitelist_ids = []
    for name in whitelist:
        if name not in ids_for_name:
            raise ValueError("%s is not in AudioSet.  Double check spelling and caps." % name)
        for id in ids_for_name[name]:
            whitelist_ids.extend(get_child_ids(id, ontology, index_for_id))

    # Return records for the ids
    unique_ids = set(whitelist_ids)
    return [ontology[index_for_id[id]] for id in unique_ids]
            
def get_child_ids(id, ontology, index_for_id):
    """Gets a list of child records of an ontology item.  Note that the id passed to this function will appear as the
    first item in the list.
    
    Arguments:
        id {string} -- ontology record id
        ontology {dict[]} -- AudioSet ontology
        index_for_id {dict<string, int>} -- mapping from AudioSet id to index in the ontology
    
    Returns:
        string[] -- list of AudioSet ids that are children of the record
    """
    record = ontology[index_for_id[id]]
    # A list of just this record.  If you wanted a list of a particular attribute, you would specify it here.
    result = [record["id"]]
    for child_id in record["child_ids"]:
        result.extend(get_child_ids(child_id, ontology, index_for_id))
    return result
