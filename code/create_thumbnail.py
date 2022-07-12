#Imports
import cv2
import os
import math
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
from os.path import isfile, isdir
import shutil
import imquality.brisque as brisque
import dlib
from mtcnn.mtcnn import MTCNN
import time
import platform
import csv
from datetime import datetime

#Paths
model_folder = "../data/models/"
frames_folder_outer = "../results/temp"
thumbnail_output = "../results/"
#Models
haarXml = model_folder + "haarcascade_frontalface_default.xml"
modelFile =  model_folder + "res10_300x300_ssd_iter_140000.caffemodel"
configFile = model_folder + "deploy.prototxt.txt"
eliteserien_logo_model = model_folder + "logo_eliteserien.h5"
soccernet_logo_model = model_folder + "logo_soccernet.h5"
surma_closeup_model = model_folder + "close_up_model.h5"

haarStr = "haar"
dlibStr = "dlib"
mtcnnStr = "mtcnn"
dnnStr = "dnn"
surmaStr = "surma"
svdStr = "svd"
laplacianStr = "laplacian"
ocampoStr = "ocampo"
eliteserienStr = "eliteserien"
soccernetStr = "soccernet"
filename_additional = "thumbnail"

#The probability score the image classifying model gives, is depending on which class it is basing the score on.
#It could be switched
close_up_model_inverted = False
#Execution time
frame_extraction=0
models_loading=0
logo_detection=0
closeup_detection=0
face_detection=0
blur_detection=0
iq_predicition=0
total=0
def main():
    #Default values
    close_up_threshold = 0.75
    totalFramesToExtract = 50
    faceDetModel = dnnStr
    framerateExtract = None
    fpsExtract = None
    cutStartSeconds = 0
    cutEndSeconds = 0
    downscaleOnProcessing = 0.5
    downscaleOutput = 1.0
    annotationSecond = None
    beforeAnnotationSecondsCut = 10
    afterAnnotationSecondsCut = 40
    staticThumbnailSec = None
    logo_model_name = eliteserienStr
    logo_detection_model = ""
    logo_threshold = 0.1
    close_up_model_name = surmaStr
    close_up_model = ""
    iqa_model_name = ocampoStr
    brisque_threshold = 35
    blur_model_name = laplacianStr
    svd_threshold = 0.60
    laplacian_threshold = 1000
    filename_output = ""
    
    parser = argparse.ArgumentParser(description="Thumbnail generator")
    parser.add_argument("destination", nargs=1, help="Destination of the input to be processed. Can be file or folder.")

    #Logo detection models
    logoGroup = parser.add_mutually_exclusive_group(required=False)
    logoGroup.add_argument("-LEliteserien2019", action='store_true', help="Surma model used for logo detection, trained on Eliteserien 2019.")
    logoGroup.add_argument("-LSoccernet", action='store_true', help="Surma model used for logo detection, trained on Soccernet.")
    logoGroup.add_argument("-xl", "--xLogoDetection", default=True, action="store_false", help="Don't run logo detection.")

    #Close-up detection models
    closeupGroup = parser.add_mutually_exclusive_group(required=False)
    closeupGroup.add_argument("-CSurma", action='store_true', help="Surma model used for close-up detection.")
    closeupGroup.add_argument("-xc", "--xCloseupDetection", default=True, action="store_false", help="Don't run close-up detection.")

    #IQA models
    iqaGroup = parser.add_mutually_exclusive_group(required=False)
    iqaGroup.add_argument("-IQAOcampo", action='store_true', help="Ocampo model used for image quality assessment.")
    iqaGroup.add_argument("-xi", "--xIQA", default=True, action="store_false", help="Don't run image quality prediction.")

    #Blur detection models
    blurGroup = parser.add_mutually_exclusive_group(required=False)
    blurGroup.add_argument("-BSVD", action='store_true', help="SVD method used for blur detection.")
    blurGroup.add_argument("-BLaplacian", action='store_true', help="Laplacian method used for blur detection.")
    blurGroup.add_argument("-xb", "--xBlurDetection", default=True, action="store_false", help="Don't run blur detection.")


    #Face models
    faceGroup = parser.add_mutually_exclusive_group(required = False)
    faceGroup.add_argument("-dlib", action='store_true', help="Dlib detection model is slow, but presice.")
    faceGroup.add_argument("-haar", action='store_true', help="Haar detection model is fast, but unprecise.")
    faceGroup.add_argument("-mtcnn", action='store_true', help="MTCNN detection model is slow, but precise.")
    faceGroup.add_argument("-dnn", action='store_true', help="DNN detection model is fast and precise.")
    faceGroup.add_argument("-xf", "--xFaceDetection", default=True, action="store_false", help="Don't run the face detection.")

    #Flags fixing default values
    parser.add_argument("-cuthr", "--closeUpThreshold", type=restricted_float, default=[close_up_threshold], nargs=1, help="The threshold value for the close-up detection model. The value must be between 0 and 1. The default is: " + str(close_up_threshold))
    parser.add_argument("-brthr", "--brisqueThreshold", type=float, default=[brisque_threshold], nargs=1, help="The threshold value for the image quality predictor model. The default is: " + str(brisque_threshold))
    parser.add_argument("-logothr", "--logoThreshold", type=restricted_float, default=[logo_threshold], nargs=1, help="The threshold value for the logo detection model. The value must be between 0 and 1. The default value is: " + str(logo_threshold))
    parser.add_argument("-svdthr", "--svdThreshold", type=restricted_float, default=[svd_threshold], nargs=1, help="The threshold value for the SVD blur detection. The default value is: " + str(svd_threshold))
    parser.add_argument("-lapthr", "--laplacianThreshold", type=float, default=[laplacian_threshold], nargs=1, help="The threshold value for the Laplacian blur detection. The default value is: " + str(laplacian_threshold))
    parser.add_argument("-css", "--cutStartSeconds", type=positive_int, default=[cutStartSeconds], nargs=1, help="The number of seconds to cut from start of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: " + str(cutStartSeconds))
    parser.add_argument("-ces", "--cutEndSeconds", type=positive_int, default=[cutEndSeconds], nargs=1, help="The number of seconds to cut from the end of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: " + str(cutEndSeconds))
    numFrameExtractGroup = parser.add_mutually_exclusive_group(required = False)
    numFrameExtractGroup.add_argument("-nfe", "--numberOfFramesToExtract", type=above_zero_int, default=[totalFramesToExtract], nargs=1, help="Number of frames to be extracted from the video for the thumbnail selection process. The default is: " + str(totalFramesToExtract))
    numFrameExtractGroup.add_argument("-fre", "--framerateToExtract", type=restricted_float, default=[framerateExtract], nargs=1, help="The framerate wanted to be extracted from the video for the thumbnail selection process.")
    numFrameExtractGroup.add_argument("-fpse", "--fpsExtract", type=above_zero_float, default=[fpsExtract], nargs=1, help="Number of frames per second to extract from the video for the thumbnail selection process.")
    parser.add_argument("-ds", "--downscaleProcessingImages", type=restricted_float, default=[downscaleOnProcessing], nargs=1, help="The value deciding how much the images to be processed should be downscaled. The default value is: " + str(downscaleOnProcessing))
    parser.add_argument("-dso", "--downscaleOutputImage", type=restricted_float, default=[downscaleOutput], nargs=1, help="The value deciding how much the output thumbnail image should be downscaled. The default value is: " + str(downscaleOutput))
    parser.add_argument("-as", "--annotationSecond", type=positive_int, default=[annotationSecond], nargs=1, help="The second the event is annotated to in the video.")
    parser.add_argument("-bac", "--beforeAnnotationSecondsCut", type=positive_int, default=[beforeAnnotationSecondsCut], nargs=1, help="Seconds before the annotation to cut the frame extraction.")
    parser.add_argument("-aac", "--afterAnnotationSecondsCut", type=positive_int, default=[afterAnnotationSecondsCut], nargs=1, help="Seconds after the annotation to cut the frame extraction.")
    parser.add_argument("-st", "--staticThumbnailSec", type=positive_int, default=[staticThumbnailSec], nargs=1, help="To generate a static thumbnail from the video, this flag is used. The second the frame should be clipped from should follow as an argument. Running this flag ignores all the other flags.")
    parser.add_argument("-fn", "--outputFilename", type=str, default=[filename_output], nargs=1, help="Filename for the output thumbnail instead of default.")

    args = parser.parse_args()
    destination = args.destination[0]
    staticThumbnailSec = args.staticThumbnailSec[0]
    filename_output = args.outputFilename[0]

    #Trimming
    annotationSecond = args.annotationSecond[0]
    beforeAnnotationSecondsCut = args.beforeAnnotationSecondsCut[0]
    afterAnnotationSecondsCut = args.afterAnnotationSecondsCut[0]
    cutStartSeconds = args.cutStartSeconds[0]
    cutEndSeconds = args.cutEndSeconds[0]
    #Down-sampling
    totalFramesToExtract = args.numberOfFramesToExtract[0]
    framerateExtract = args.framerateToExtract[0]
    fpsExtract = args.fpsExtract[0]
    if fpsExtract:
        totalFramesToExtract = None
        framerateExtract = None
    if framerateExtract:
        totalFramesToExtract = None
        fpsExtract = None
    if totalFramesToExtract:
        framerateExtract = None
        fpsExtract = None
    #Down-scaling
    downscaleOnProcessing = args.downscaleProcessingImages[0]
    downscaleOutput = args.downscaleOutputImage[0]

    #Logo detection
    runLogoDetection = args.xLogoDetection
    if not runLogoDetection:
        logo_model_name = ""
    if args.LEliteserien2019:
        logo_model_name = eliteserienStr
    elif args.LSoccernet:
        logo_model_name = soccernetStr
    logo_threshold = args.logoThreshold[0]

    #Close-up detection
    runCloseUpDetection = args.xCloseupDetection
    if not runCloseUpDetection:
        close_up_model_name = ""
    if args.CSurma:
        close_up_model_name = surmaStr
    close_up_threshold = args.closeUpThreshold[0]

    #Face detection
    runFaceDetection = args.xFaceDetection
    if not runFaceDetection:
        faceDetModel = ""
    if args.dlib:
        faceDetModel = dlibStr
    elif args.haar:
        faceDetModel = haarStr
    elif args.mtcnn:
        faceDetModel = mtcnnStr
    elif args.dnn:
        faceDetModel = dnnStr

    #Image Quality Assessment
    runIQA = args.xIQA
    if not runIQA:
        iqa_model_name = ""
    if args.IQAOcampo:
        iqa_model_name = ocampoStr
    brisque_threshold = args.brisqueThreshold[0]

    runBlur = args.xBlurDetection
    if not runBlur:
        blur_model_name = ""
    if args.BSVD:
        blur_model_name = svdStr
    elif args.BLaplacian:
        blur_model_name = laplacianStr
    svd_threshold = args.svdThreshold[0]
    laplacian_threshold = args.laplacianThreshold[0]

    processFolder = False
    processFile = False
    if os.path.isdir(destination):
        processFolder = True
        if destination[-1] != "/":
            destination = destination + "/"
        print("is folder")
    elif os.path.isfile(destination):
        processFile = True
        print("is file")
        name, ext = os.path.splitext(destination)
        if ext != ".ts" and ext != ".mp4" and ext != ".mkv":
            raise Exception("The input file is not a video file")
    else:
        raise Exception("The input destination was neither file or directory")

    try:
        if not os.path.exists(thumbnail_output):
            os.mkdir(thumbnail_output)

    except OSError:
        print("Error: Couldn't create thumbnail output directory")
        return

    if staticThumbnailSec:
        get_static(destination, staticThumbnailSec, downscaleOutput, thumbnail_output)
        return
    loadingModelsStarts=time.time()
    if close_up_model_name == surmaStr:
        close_up_model = keras.models.load_model(surma_closeup_model)

    if logo_model_name == eliteserienStr:
        logo_detection_model = keras.models.load_model(eliteserien_logo_model)
    elif logo_model_name == soccernetStr:
        logo_detection_model = keras.models.load_model(soccernet_logo_model)
    loadingModelsEnds=time.time()
    models_loading=loadingModelsEnds-loadingModelsStarts

    if processFile:
        create_thumbnail(name + ext, downscaleOutput, downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBlur, blur_model_name, svd_threshold, laplacian_threshold, runIQA, iqa_model_name, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, logo_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut, filename_output)
        
    elif processFolder:
        for f in os.listdir(destination):
            name, ext = os.path.splitext(f)
            if ext == ".ts" or ext == ".mp4" or ext == ".mkv":
                create_thumbnail(destination + name + ext, downscaleOutput , downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBlur, blur_model_name, svd_threshold, laplacian_threshold, runIQA, iqa_model_name, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, logo_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut, filename_output)

    def logMetrics(directory,fileName):
     completeName = os.path.join(directory,fileName)
     header=['platform',
             'date_time',
             'nfe',
             'frame_extraction',
             'models_loading',
             'logo_detection',
             'closeup_detection',
             'face_detection',
             'blur_detection',
             'iq_predicition',
             'total']
     with open (completeName+".csv",'a+' ) as file:
        writer = csv.writer(file)
        if os.stat(completeName+".csv").st_size == 0:
            writer.writerow(header)
        #data to be written
        total=frame_extraction+models_loading+logo_detection+closeup_detection+face_detection+blur_detection+iq_predicition
        data=[platform.system(),
              datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
              totalFramesToExtract,
              "{0:.3f}".format(frame_extraction) if frame_extraction>0 else "disabled",
              "{0:.3f}".format(models_loading) if models_loading>0 else "disabled",
              f'{logo_model_name} '+"{0:.3f}".format(logo_detection)if logo_detection>0 else "disabled",
              f'{close_up_model_name} '+"{0:.3f}".format(closeup_detection)if closeup_detection>0 else "disabled",
              f'{faceDetModel} '+"{0:.3f}".format(face_detection)if face_detection>0 else "disabled",
              f'{blur_model_name} '+"{0:.3f}".format(blur_detection)if blur_detection>0 else "disabled",
              f'{iqa_model_name} '+"{0:.3f}".format(iq_predicition)if iq_predicition>0 else "disabled",
              "{0:.3f}".format(total)]
        writer.writerow(data)
    logMetrics("../results","performance_metrics")

def create_thumbnail(video_path, downscaleOutput, downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBlur, blur_model_name, svd_threshold, laplacian_threshold, runIQA, iqa_model_name, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, logo_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut, filename_output):
    frameExtractionStarts=time.time()
    video_filename = video_path.split("/")[-1]
    #frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
    frames_folder = frames_folder_outer + "/frames/"
    if not os.path.exists(frames_folder_outer):
        os.mkdir(frames_folder_outer)
    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)

    #frames_folder = frames_folder_outer + "/"

    # Read the video from specified path

    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fps = cam.get(cv2.CAP_PROP_FPS)

    duration = totalFrames/fps

    if annotationSecond:
        if beforeAnnotationSecondsCut:
            cutStartSeconds = annotationSecond - beforeAnnotationSecondsCut
        if afterAnnotationSecondsCut:
            cutEndSeconds = duration - (annotationSecond + afterAnnotationSecondsCut)


    cutStartFrames = fps * cutStartSeconds
    cutEndFrames = fps * cutEndSeconds


    if totalFrames < cutStartFrames + cutEndFrames:
        raise Exception("All the frames are cut out")
    

    remainingFrames = totalFrames - (cutStartFrames + cutEndFrames)
    remainingSeconds = remainingFrames / fps

    if fpsExtract:
        totalFramesToExtract = math.floor(remainingSeconds * fpsExtract)
    if framerateExtract:
        totalFramesToExtract = math.floor(remainingFrames * framerateExtract)


    currentframe = 0
    # frames to skip
    frame_skip = (totalFrames-(cutStartFrames + cutEndFrames))//totalFramesToExtract
    numFramesExtracted = 0
    stopFrame = totalFrames-cutEndFrames
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe > stopFrame:
            break
        if currentframe < cutStartFrames:
            currentframe += 1
            continue
        if currentframe % frame_skip == 0 and numFramesExtracted < totalFramesToExtract:
            # if video is still left continue creating images
            name = frames_folder + 'frame' + str(currentframe) + '.jpg'
            #name = 'frame' + str(currentframe) + '.jpg'
            width = int(frame.shape[1] * downscaleOnProcessing)
            height = int(frame.shape[0] * downscaleOnProcessing)
            dsize = (width, height)
            img = cv2.resize(frame, dsize)
            cv2.imwrite(name, img)
            numFramesExtracted += 1

        currentframe += 1
    frameExtractionEnds=time.time()

        
    priority_images = groupFrames(frames_folder, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runLogoDetection, runCloseUpDetection, close_up_threshold, logo_threshold)
    finalThumbnail = ""

    for priority in priority_images:
        if finalThumbnail != "":
            break
        priority = dict(sorted(priority.items(), key=lambda item: item[1], reverse=True))

        blur_filtered = []
        if runBlur:
            blurDetectionStarts=time.time()
            if blur_model_name == svdStr:
                for image in priority:
                    blur_score = estimate_blur_svd(image)
                    if blur_score < svd_threshold:
                        blur_filtered.append(image)
            if blur_model_name == laplacianStr:
                for image in priority:
                    blur_score = estimate_blur_laplacian(image)
                    if blur_score > laplacian_threshold:
                        blur_filtered.append(image)
            blurDetectionEnds=time.time()
            if runBlur:
                global blur_detection
                blur_detection=blurDetectionEnds-blurDetectionStarts


        else:
            for image in priority:
                blur_filtered.append(image)


        if runIQA:
            IQAStarts=time.time()
            if iqa_model_name == ocampoStr:
                bestScore = 0
                for image in blur_filtered:
                    score = predictBrisque(image)
                    if finalThumbnail == "":
                        bestScore = score
                        finalThumbnail = image
                    if score < brisque_threshold:
                        finalThumbnail = image
                        break
                    if score < bestScore:
                        bestScore = score
                        finalThumbnail = image
            IQAEnds=time.time()
            if runIQA:
                global iq_predicition
                iq_predicition=IQAEnds-IQAStarts

        else:
            for image in blur_filtered:
                finalThumbnail = image
                break
    if finalThumbnail == "":
        for priority in priority_images:
            if finalThumbnail != "":
                break
            for image in priority:
                finalThumbnail = image
                break

    if finalThumbnail != "":
        newName = ""
        if filename_output == "":
            newName = video_filename.split(".")[0] + "_" + filename_additional +  ".jpg"
        else:
            newName = filename_output
            extension_added = len(newName.split(".")) == 2
            if not extension_added:
                newName = newName + ".jpg"
            
        imageName = finalThumbnail.split("/")[-1].split(".")[0]
        frameNum = int(imageName.replace("frame", ""))

        cam.set(1, frameNum)
        ret, frame = cam.read()
        if downscaleOutput != 1.0:
            width = int(frame.shape[1] * downscaleOutput)
            height = int(frame.shape[0] * downscaleOutput)
            dsize = (width, height)
            frame = cv2.resize(frame, dsize)

        cv2.imwrite(thumbnail_output + newName, frame)
        print("Thumbnail created. Filename: " + newName)
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        #secInVid = (frameNum / totalFrames) * duration

        try: 
            shutil.rmtree(frames_folder)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    global frame_extraction
    frame_extraction=frameExtractionEnds-frameExtractionStarts
    print("Done")
    #Metrics logging function
    return

def groupFrames(frames_folder, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runLogoDetection, runCloseUpDetection, close_up_threshold, logo_threshold):
    test_generator = None
    TEST_SIZE = 0
    if runCloseUpDetection or runLogoDetection:
        test_data_generator = ImageDataGenerator(rescale=1./255)
        IMAGE_SIZE = 200
        TEST_SIZE = len(next(os.walk(frames_folder))[2])
        IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
        test_generator = test_data_generator.flow_from_directory(
                frames_folder + "../",
                target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                batch_size=1,
                class_mode="binary",
                shuffle=False)

    logos = []
    if runLogoDetection:
        logoDetectionStarts=time.time()

        logo_probabilities = logo_detection_model.predict(test_generator, TEST_SIZE)

        for index, probability in enumerate(logo_probabilities):
            image_path = frames_folder + test_generator.filenames[index].split("/")[-1]
            if probability > logo_threshold:
                logos.append(image_path)
        logoDetectionEnds=time.time()
        if runLogoDetection:
             global logo_detection
             logo_detection=logoDetectionEnds-logoDetectionStarts
    priority_images = [{} for x in range(4)]
    if runCloseUpDetection:
        faceDetection=[]
        closeUpDetectionStarts=time.time()
        probabilities = close_up_model.predict(test_generator, TEST_SIZE)

        for index, probability in enumerate(probabilities):
            #The probability score is inverted:
            if close_up_model_inverted:
                probability = 1 - probability

            image_path = frames_folder + test_generator.filenames[index].split("/")[-1]

            if image_path in logos:
                priority_images[3][image_path] = probability

            elif probability > close_up_threshold:
                if runFaceDetection:
                    faceDetectionStarts=time.time()
                    face_size = detect_faces(image_path, faceDetModel)
                    if face_size > 0:
                        priority_images[0][image_path] = face_size
                    else:
                        priority_images[1][image_path] = probability
                    faceDetectionEnds=time.time()


                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[2][image_path] = probability
        closeUpDetectionEnds=time.time()
        if runCloseUpDetection:
            global closeup_detection
            closeup_detection=closeUpDetectionEnds-closeUpDetectionStarts
        if runFaceDetection:
            global face_detection
            face_detection=faceDetectionEnds-faceDetectionStarts

    else:
        probability = 1
        for image in os.listdir(frames_folder):
            image_path = frames_folder + image
            if image_path in logos:
                priority_images[3][image_path] = probability
            if runFaceDetection:
                face_size = detect_faces(image_path, faceDetModel)
                if face_size > 0:
                    priority_images[0][image_path] = face_size
                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[1][image_path] = probability
    return priority_images

def get_static(video_path, secondExtract, downscaleOutput, outputFolder):
    video_filename = video_path.split("/")[-1]
    #frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
    frames_folder = frames_folder_outer + "/temp/"


    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))-1
    fps = cam.get(cv2.CAP_PROP_FPS)

    duration = totalFrames/fps


    cutStartFrames = fps * secondExtract


    if totalFrames < cutStartFrames:
        raise Exception("All the frames are cut out")

    currentframe = 0
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe <= cutStartFrames:
            currentframe += 1
            continue
        width = int(frame.shape[1] * downscaleOutput)
        height = int(frame.shape[0] * downscaleOutput)
        dsize = (width, height)
        img = cv2.resize(frame, dsize)
        newName = video_filename.split(".")[0] + "_static_thumbnail.jpg"
        cv2.imwrite(outputFolder + newName, img)
        break


def predictBrisque(image_path):
    img = cv2.imread(image_path)
    brisqueScore = brisque.score(img)

    return brisqueScore

def estimate_blur_svd(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv


def estimate_blur_laplacian(image_file):
    #img = cv2.imread(
    img = cv2.imread(image_file,cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(img, cv2.CV_64F)
    score = np.var(blur_map)
    return score

def detect_faces(image, faceDetModel):
    biggestFace = 0
    if faceDetModel == dlibStr:
        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for result in faces:
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            size = y1-y
            if biggestFace < size:
                biggestFace = size

    elif faceDetModel == haarStr:
        face_cascade = cv2.CascadeClassifier(haarXml)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == mtcnnStr:
        detector = MTCNN()
        img = cv2.imread(image)
        faces = detector.detect_faces(img)

        for result in faces:
            x, y, w, h = result['box']
            size = h
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == dnnStr:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        img = cv2.imread(image)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                height = y1 - y
                if biggestFace < height:
                    biggestFace = height

    else:
        print("No face detection model in use")
    return biggestFace

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def above_zero_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x <=0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive int"%(x,))
    return x

def above_zero_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x <= 0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x


if __name__ == "__main__":
    main()
