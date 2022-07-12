import os
import json
file = open('../code/test_suite.json')
data = json.load(file)
for entry in data:
 for val in range(50,entry["nfe"],entry["frame_extraction_step"]):
  if entry['runStatic']:
    os.system("python3 create_thumbnail.py %s -nfe %d -st" % ('../data/videos/' ,val))
  elif entry['runMlDefault']:
    os.system("python3 create_thumbnail.py %s -nfe %d" % ('../data/videos/' ,val))
  else:
    os.system("python3 create_thumbnail.py %s -nfe %d %s" % ('../data/videos/' ,val," ".join(entry["args"])))


