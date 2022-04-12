import pathlib
from os.path import join

path_constants_json = pathlib.Path(__file__).parent.absolute()

file_name = "constants_veg.json"

fpath_constants_file = join(path_constants_json, file_name)
