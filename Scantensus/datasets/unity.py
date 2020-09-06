import glob
import logging
from pathlib import Path

logger = logging.getLogger()

class UnityO:

    # Like with DICOM - for the word image read instance.

    def __init__(self, unity_code: str, png_cache_dir=None, server_url=None):
        self.png_cache_dir = png_cache_dir
        self.server_url = server_url

        unity_code = Path(unity_code)
        unity_code = unity_code.name.split('.')[0]
        unity_code = unity_code.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")

        self.unity_code = unity_code

        unity_code_len = len(self.unity_code)

        self._failed = False
        if unity_code_len == 67:
            self.code_type = 'video'
            self.unity_i_code = self.unity_code
        elif unity_code_len == 72:
            self.code_type = 'frame'
            self.frame_num = int(unity_code[-4:])
            self.unity_i_code = self.unity_code[:-5]
            self.unity_f_code = self.unity_code
        else:
            logger.warning(f"{unity_code} not a valid code")
            self._failed = True
            return

        self._sub_a = self.unity_i_code[:2]
        self._sub_b = self.unity_i_code[3:5]
        self._sub_c = self.unity_i_code[5:7]

    def get_frame_path(self, frame_offset=0):
        if self._failed:
            return ""

        if self.code_type == 'frame':
            return f"{self.png_cache_dir / self._sub_a / self._sub_b / self._sub_c / self.unity_i_code}-{(self.frame_num + frame_offset):04}.png"
        else:
            raise Exception

    def get_all_frames_path(self):
        search_string = f"{self.png_cache_dir / self._sub_a / self._sub_b / self._sub_c / self.unity_i_code}*.png"
        images_path = glob.glob(search_string)
        valid_images = sorted(images_path)

        return valid_images
