import download
import process_ontology

class CategoryGetter:
    def __init__(self, original_video_list):
        """Initializes an instance
        
        Arguments:
            original_video_list {string} -- original csv file that will be queried for video ids
        """
        youtube_clips = download.read_clips(original_video_list)
        self.id_to_index = {clip.id: clip for clip in youtube_clips}
        self.ontology = process_ontology.Ontology()

    def get_general_categories_for_video(self, video_id):
        """Gets the most general ontology ids for a video.
        
        Arguments:
            video_id {string} -- YouTube video id
        
        Returns:
            Set<string> -- set of the most general onotology ids that the video has
        """
        clip = self.id_to_index[video_id]
        return set([self.ontology.get_most_general_id(label) for label in clip.labels])

    def get_specific_categories_for_video(self, video_id):
        """Gets a list of category ids for a video that have no children in the ontology; i.e. can't get more specific
        
        Arguments:
            video_id {string} -- YouTube video id
        
        Returns:
            string[] -- list of ontology ids
        """
        clip = self.id_to_index[video_id]
        return [label for label in clip.labels if self.ontology.is_specific(label)]

    def get_human_readable_category(self, category_id):
        """
        Arguments:
            category_id {string} -- ontology category id
        
        Returns:
            string -- human-readable translation of the ontology category id
        """
        return self.ontology.get_record_for_id(category_id)["name"]
