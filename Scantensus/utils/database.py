import os

class ReverseUnityVideoDatabase:
    def __init__(self, database_dir):

        i = 0
        reverse_location = {}

        for root, directories, filenames in os.walk(database_dir):
            for filename in filenames:
                if filename.endswith(".png"):
                    # Remove .png and -####
                    reverse_location[filename[:-9]] = root
                    i = i + 1
                    if i % 10000 == 0:
                        print(f"Found {i} files in database")

        self._reverse_db = reverse_location

    def reverse_dir(self, video):

        return self._reverse_db.get(video, None)