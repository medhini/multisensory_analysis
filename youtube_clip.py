class YouTubeClip:
    def __init__(self, id, trim_start, trim_end, labels):
        """Makes a new YouTubeClip object
        
        Arguments:
            id {string} -- YouTube ID
            trim_start {float-like} -- start of the video clip, in seconds
            trim_end {float-like} -- end of the video clip, in seconds
            labels {string} -- comma-separated labels
        """
        self.id = id
        self.trim_start = float(trim_start)
        self.trim_end = float(trim_end)
        self.labels = [l.strip() for l in labels.split(",")]

    def get_duration(self):
        return self.trim_end - self.trim_start
    
    def to_string(self):
        return "%s_%d_%d" % (self.id, int(self.trim_start * 1000), int(self.trim_end * 1000))
