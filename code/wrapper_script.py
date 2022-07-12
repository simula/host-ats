import os
nfe_values=[50,75,100,150,175,200,225,265,300,350,400,450,500,550,650,700,850]
for val in nfe_values:
 os.system("python3 create_thumbnail.py %s -nfe %d" % ('../data/videos/sample.mp4' ,val))