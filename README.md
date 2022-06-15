# HOST-ATS Automatic Thumbnail Selection Pipeline

# Installation

Make sure you have Python version 3 before you start the installation. 
You can check your Python version by running the following command:

```
python --version
```

After cloning this repo, the packages and Python libraries needed for running the pipeline can be installed by running the following commands: 

```
apt-get update
xargs apt-get install -y <packages.txt
pip install -r requirements.txt
```

## Versions

The pipeline was tested using the following versions in a Ubuntu 20.04.4 LTS environment with Python 3.8.13 and pip 20.2.4*:

Packages (versions not specified in `packages.txt`**):

```
ffmpeg=7:4.2.4-1ubuntu0.1
libsm6=2:1.2.3-1
libxext6=2:1.3.4-0ubuntu1
```

_(*) This version of pip (non-latest) is used to ensure comparability with the public [CodeOcean capsule](#CodeOcean-Capsule)._  
_(**) Package versions are not specified explicitly in `packages.txt`, in order to allow for the automatic installation of the corresponding latest versions in different OS flavors._

Python libraries (versions specified in `requirements.txt`):

```
cmake==3.22.3
dlib==19.23.1
image-quality==1.2.7
mtcnn==0.1.1
numpy==1.22.3
opencv-python==4.5.5.64
pillow==9.0.1
scikit-image==0.18.3
tensorflow-cpu==2.8.0
```


## Possible Installation Warnings

### `dlib`

There can be a wheel build issue while installing the `dlib` library, but it is possible to install it using the legacy `setup.py install` method. Sample warning messages below.

```
  Building wheel for dlib (setup.py): started
  Building wheel for dlib (setup.py): finished with status 'error'
  ERROR: Command errored out with exit status 1:
   command: /opt/conda/bin/python -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-w2fiwhkz/dlib/setup.py'"'"'; __file__='"'"'/tmp/pip-install-w2fiwhkz/dlib/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d /tmp/pip-wheel-80kgcg5h
       cwd: /tmp/pip-install-w2fiwhkz/dlib/
  Complete output (8 lines):
  running bdist_wheel
  running build
  running build_py
  package init file 'tools/python/dlib/__init__.py' not found (or not a regular file)
  running build_ext
  
  ERROR: CMake must be installed to build dlib
  
  ----------------------------------------
  ERROR: Failed building wheel for dlib
  Running setup.py clean for dlib
[...]
Failed to build dlib
Installing collected packages: cmake, dlib, pillow, libsvm, numpy, scipy, tifffile, networkx, imageio, PyWavelets, pyparsing, kiwisolver, python-dateutil, fonttools, packaging, cycler, matplotlib, scikit-image, image-quality, opencv-python, keras, mtcnn, flatbuffers, keras-preprocessing, libclang, tensorflow-io-gcs-filesystem, typing-extensions, tf-estimator-nightly, absl-py, astunparse, h5py, gast, werkzeug, oauthlib, requests-oauthlib, pyasn1, rsa, cachetools, pyasn1-modules, google-auth, google-auth-oauthlib, tensorboard-data-server, tensorboard-plugin-wit, protobuf, zipp, importlib-metadata, markdown, grpcio, tensorboard, opt-einsum, google-pasta, termcolor, wrapt, tensorflow-cpu
    Running setup.py install for dlib: started
    Running setup.py install for dlib: still running...
    Running setup.py install for dlib: finished with status 'done'

```

# Data

## Input Video Clips

Video clips for which thumbnails are to be generated should be placed under the `data/videos` folder in the local runtime directory. The pipeline can be run with any video (e.g., goal clips from the [SoccerNet](https://www.soccer-net.org/download) dataset, highlight galleries for the Norwegian [Eliteserien](https://highlights.eliteserien.no/) or the Swedish [Allsvenskan](https://highlights.allsvenskan.se/)).

## Models

Models to be used in the various modules of the pipeline should be placed under the `data/models` folder in the local runtime directory. Examples include:

| Corresponding Module | Model |
| ------------- | ------------- |
| Face detection  | [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades)  |
| Face detection  | [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/sr6033/face-detection-with-OpenCV-and-DNN) |
| Face detection  | [deploy.prototxt.txt](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)  |
| Logo detection  | logo_eliteserien.h5 (this repository)  |
| Logo detection  | logo_soccernet.h5 (this repository) |
| Close-up shot detection  | close_up_model.h5 (this repository) |


# Running

The automatic thumbnail selection pipeline can be run by calling the script `python create_thumbnail.py` in the terminal, where various configuration parameters can be specified as command line options.

To run the `create_thumbnail.py` script, first go into the `code/` directory:

```
cd code/
``` 

Then, run the pipeline with the path to a single video clip or a video clip folder as input:

```
python create_thumbnail.py <path_to_video_file_or_folder>
``` 

The above command will run the pipeline with the default configuration parameters, and output a thumbnail image per video clip (see [Output](#Output)).


## Configuration

It is possible to modify configuration parameters using the command line options listed below.

To list the available options, run:

```
python create_thumbnail.py -h
```

and the following will display in terminal:

```
create_thumbnail.py [-h] [-LEliteserien2019 | -LSoccernet | -xl] [-CSurma | -xc] [-IQAOcampo | -xi] [-BSVD | -BLaplacian | -xb] [-dlib | -haar | -mtcnn | -dnn | -xf] [-cuthr CLOSEUPTHRESHOLD]
                           [-brthr BRISQUETHRESHOLD] [-logothr LOGOTHRESHOLD] [-svdthr SVDTHRESHOLD] [-lapthr LAPLACIANTHRESHOLD] [-css CUTSTARTSECONDS] [-ces CUTENDSECONDS]
                           [-nfe NUMBEROFFRAMESTOEXTRACT | -fre FRAMERATETOEXTRACT | -fpse FPSEXTRACT] [-ds DOWNSCALEPROCESSINGIMAGES] [-dso DOWNSCALEOUTPUTIMAGE] [-as ANNOTATIONSECOND] [-bac BEFOREANNOTATIONSECONDSCUT]
                           [-aac AFTERANNOTATIONSECONDSCUT] [-st STATICTHUMBNAILSEC]
                           destination
                           
Thumbnail generator

positional arguments:
  destination           Destination of the input to be processed. Can be file or folder.

options:
  -h, --help            show this help message and exit
  -LEliteserien2019     Surma model used for logo detection, trained on Eliteserien 2019.
  -LSoccernet           Surma model used for logo detection, trained on Soccernet.
  -xl, --xLogoDetection
                        Don't run logo detection.
  -CSurma               Surma model used for close-up detection.
  -xc, --xCloseupDetection
                        Don't run close-up detection.
  -IQAOcampo            Ocampo model used for image quality assessment.
  -xi, --xIQA           Don't run image quality prediction.
  -BSVD                 SVD method used for blur detection.
  -BLaplacian           Laplacian method used for blur detection.
  -xb, --xBlurDetection
                        Don't run blur detection.
  -dlib                 Dlib detection model is slow, but presice.
  -haar                 Haar detection model is fast, but unprecise.
  -mtcnn                MTCNN detection model is slow, but precise.
  -dnn                  DNN detection model is fast and precise.
  -xf, --xFaceDetection
                        Don't run the face detection.
  -cuthr CLOSEUPTHRESHOLD, --closeUpThreshold CLOSEUPTHRESHOLD
                        The threshold value for the close-up detection model. The value must be between 0 and 1. The default is: 0.75
  -brthr BRISQUETHRESHOLD, --brisqueThreshold BRISQUETHRESHOLD
                        The threshold value for the image quality predictor model. The default is: 35
  -logothr LOGOTHRESHOLD, --logoThreshold LOGOTHRESHOLD
                        The threshold value for the logo detection model. The value must be between 0 and 1. The default value is: 0.1
  -svdthr SVDTHRESHOLD, --svdThreshold SVDTHRESHOLD
                        The threshold value for the SVD blur detection. The default value is: 0.65
  -lapthr LAPLACIANTHRESHOLD, --laplacianThreshold LAPLACIANTHRESHOLD
                        The threshold value for the Laplacian blur detection. The default value is: 1000
  -css CUTSTARTSECONDS, --cutStartSeconds CUTSTARTSECONDS
                        The number of seconds to cut from start of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: 0
  -ces CUTENDSECONDS, --cutEndSeconds CUTENDSECONDS
                        The number of seconds to cut from the end of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: 0
  -nfe NUMBEROFFRAMESTOEXTRACT, --numberOfFramesToExtract NUMBEROFFRAMESTOEXTRACT
                        Number of frames to be extracted from the video for the thumbnail selection process. The default is: 50
  -fre FRAMERATETOEXTRACT, --framerateToExtract FRAMERATETOEXTRACT
                        The framerate wanted to be extracted from the video for the thumbnail selection process.
  -fpse FPSEXTRACT, --fpsExtract FPSEXTRACT
                        Number of frames per second to extract from the video for the thumbnail selection process.
  -ds DOWNSCALEPROCESSINGIMAGES, --downscaleProcessingImages DOWNSCALEPROCESSINGIMAGES
                        The value deciding how much the images to be processed should be downscaled. The default value is: 0.5
  -dso DOWNSCALEOUTPUTIMAGE, --downscaleOutputImage DOWNSCALEOUTPUTIMAGE
                        The value deciding how much the output thumbnail image should be downscaled. The default value is: 1.0
  -as ANNOTATIONSECOND, --annotationSecond ANNOTATIONSECOND
                        The second the event is annotated to in the video.
  -bac BEFOREANNOTATIONSECONDSCUT, --beforeAnnotationSecondsCut BEFOREANNOTATIONSECONDSCUT
                        Seconds before the annotation to cut the frame extraction.
  -aac AFTERANNOTATIONSECONDSCUT, --afterAnnotationSecondsCut AFTERANNOTATIONSECONDSCUT
                        Seconds after the annotation to cut the frame extraction.
  -st STATICTHUMBNAILSEC, --staticThumbnailSec STATICTHUMBNAILSEC
                        To generate a static thumbnail from the video, this flag is used. The second the frame should be clipped from should follow as an argument. Running this flag ignores all the other flags.
```

## Possible Runtime Warnings

### `tensorflow`

```
I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

### `image-quality`

The image quality predictor in package `image-quality==1.2.7` is not compatible with `scikit-image>=0.19`. 
This is why the `requirements.txt` explicitly specifies an older version of `scikit-image`. 

During runtime, this warning message can be seen:

```
/opt/conda/lib/python3.8/site-packages/imquality/brisque.py:45: FutureWarning: The behavior of rgb2gray will change in scikit-image 0.19. Currently, rgb2gray allows 2D grayscale image to be passed as inputs and leaves them unmodified as outputs. Starting from version 0.19, 2D arrays will be treated as 1D images with 3 channels.
```

A future fix in the [`image-quality` package](https://pypi.org/project/image-quality/) can allow for using newer versions of `scikit-image`, thereby resolving this warning.

# Output

Running the pipeline by calling `create_thumbnail.py` outputs one image file (`jpg`) per input video clip, into the folder `results/`. 
This is the thumbnail selected for the particular video clip. 
If a complete folder is specified as input, the pipeline iterates through all video files in the folder, and outputs one image per video file. 

# Huldra

The user study survey framework Huldra is accessible on GitHub: https://github.com/simula/huldra 

# Code Ocean Capsule

A reproducible Code Ocean capsule is runnable at this link: https://doi.org/10.24433/CO.2648317.v1

# Dashboard Demonstration Videos

The dashboard with a graphical user interface is available at YouTube: https://www.youtube.com/watch?v=HHMCdMucorI and https://www.youtube.com/watch?v=VZQaEy2VauQ

