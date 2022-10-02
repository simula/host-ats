import os
import json
file = open('../code/test_suite.json')
data = json.load(file)
for entry in data:
  for iteration in range (1,entry["iterate"]+1):
    if entry['runStatic']:
      os.system("python3 create_thumbnail.py %s -nfe %d -iter %d -st" % ('../data/videos/',iteration))
    elif entry['runMlDefault']:
      os.system("python3 create_thumbnail.py %s -nfe %d -iter %d" % ('../data/videos/',iteration))
    else:
      for downSamplingVal in entry["downSampling"]['numberOfFramesPerSecond']:
        for downScalingonProcessingVal in entry["downScaling"]['downscaleOnProcessing']:
          for downScalingPostProcessingVal in entry["downScaling"]['downScalePostProcessing']:
                for annotationSecondVal in entry["annotation"]["annotationSecond"]:
                  for cutBeforeAnnotationVal in entry["annotation"]["cutBeforeAnnotation"]:
                    for cutAfterAnnotationVal in entry["annotation"]["cutAfterAnnotation"]:
                     os.system("python3 create_thumbnail.py %s  -fpse %d -iter %d -ds %f -dso %f  -as %d -bac %d -aac %d %s" % ('../data/videos/',
                     downSamplingVal,iteration,
                     downScalingonProcessingVal,downScalingPostProcessingVal,
                     annotationSecondVal,
                     cutBeforeAnnotationVal,cutAfterAnnotationVal," ".join(entry["args"])))




    


