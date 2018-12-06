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
