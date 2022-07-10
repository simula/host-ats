import os
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from os.path import exists

def generate():

	try:
		filePath = fileName.get()
		if not exists(filePath):
			print("File doesn't exist!")
			raise Exception
		filename = filePath.split('/')[-1]
		name, ext = os.path.splitext(filename)

		print("Number frames to extract: %s\nFace detection model: %s\nFilename: %s" % (numFramesExtractVar.get(), faceVar.get(), fileName.get()))

		annotationStr = ""
		if annotationMarkVar.get():
			annotationStr = "-as " + annotationMarkVar.get()
			if beforeAnnotationVar.get():
				annotationStr += " -bac " + beforeAnnotationVar.get()
			if afterAnnotationVar.get():
				annotationStr += " -aac " + afterAnnotationVar.get()

		csStr = "-css " + cutStartVar.get()
		ceStr = "-ces " + cutEndVar.get()


		downsamplingAlt = ["-nfe " + numFramesExtractVar.get(), "-fre " + downsamplingRatioVar.get(), "-fps " + fpsVar.get()]
		downsamplingStr = downsamplingAlt[down_sampling_var.get()-1]

		internalProcessingStr = ""
		if internalProcessingVar.get():
			internalProcessingStr = "-ds " + str(float(internalProcessingVar.get())/100)

		outputImageStr = ""
		if outputImageVar.get():
			outputImageStr = "-dso " + str(float(outputImageVar.get())/100)

		runLogoStr = "-xl" if not runLogoVar.get() else "-L" + logoVar.get() + " -logothr " + str(float(logoThresholdVar.get())/100)

		runCloseup = "" if not runcloseupVar.get() else "-C" + closeupVar.get() + " -cuthr " + str(float(closeupThresholdVar.get())/100)

		runFaceStr = "-xf" if not runFaceVar.get() else "-" + faceVar.get()

		runIQAStr = "-xi" if not runIQAVar.get() else "-IQA" + iqaVar.get() + " -brthr " + brisqueVar.get()

		blurStr = ""
		if blurVar.get() == "SVD":
			blurStr = "-BSVD -svdthr " + blurThresholdVar.get()
		elif blurVar.get() == "Laplacian":
			blurStr = "-BLaplacian -lapthr " + blurThresholdVar.get()

		runBlurStr = "-xb" if not runBlurVar.get() else blurStr

		runStatic = "python3 create_thumbnail.py %s -st 4\n" % (filePath)
		if os.system(runStatic) != 0:
			raise Exception('runStatic did not work')
		runMLbased = 'python3 create_thumbnail.py %s %s %s %s %s %s %s %s %s %s %s %s\n' % (filePath, annotationStr, csStr, ceStr, downsamplingStr, internalProcessingStr, outputImageStr, runLogoStr, runCloseup, runFaceStr, runIQAStr, runBlurStr)
		print(runMLbased)
		if os.system(runMLbased) != 0:
			raise Exception('runMLbased did not work')
		outputname = '../results/' + name + '_thumbnail.jpg'
		staticOutputname = '../results/' + name + '_static_thumbnail.jpg'

		imgS = Image.open(staticOutputname)
		imgS = imgS.resize((width,height), Image.ANTIALIAS)
		photoImgS = ImageTk.PhotoImage(imgS)
		master.imgS = photoImgS
		canvas1.create_image(20,20, anchor=NW, image=photoImgS)
		imgML = Image.open(outputname)
		imgML = imgML.resize((width,height), Image.ANTIALIAS)
		photoImgML = ImageTk.PhotoImage(imgML)
		master.imgML = photoImgML
		canvas2.create_image(20,20, anchor=NW, image=photoImgML)
		generate_response_text.set("")
	except:
		generate_response_text.set("Creating thumbnail did not work with the given arguments!")
		print("Creating thumbnail did not work with the given arguments")

def open_file():
	video_file = askopenfilename()
	video_file_text.set(video_file)
	master.imgS = ""
	master.imgML = ""
	if len(video_file) == 0:
		video_file_text.set("No file selected")
	fileName.delete(0, 'end')
	fileName.insert(0, video_file)
def display_face_det_models():
	if runFaceVar.get():
		faceDropDown.config(state='normal')
	else:
		faceDropDown.config(state='disabled')

def display_brisque_thr():
	if runIQAVar.get():
		brisqueVar.config(state='normal')
		iqaDropDown.config(state='normal')
	else:
		brisqueVar.config(state='disabled')
		iqaDropDown.config(state='disabled')

def display_blur_thr():
	if runBlurVar.get():
		blurThresholdVar.config(state='normal')
		blurDropDown.config(state='normal')
	else:
		blurThresholdVar.config(state='disabled')
		blurDropDown.config(state='disabled')
def display_logo_det_models():
	if runLogoVar.get():
		logoDropDown.config(state='normal')
		logoThresholdVar.config(state='normal')
	else:
		logoDropDown.config(state='disabled')
		logoThresholdVar.config(state='disabled')

def disable_other_entries_downsampling():
	if down_sampling_var.get() == 1:
		numFramesExtractVar.config(state='normal')
		downsamplingRatioVar.config(state='disabled')
		fpsVar.config(state='disabled')
	if down_sampling_var.get() == 2:
		numFramesExtractVar.config(state='disabled')
		downsamplingRatioVar.config(state='normal')
		fpsVar.config(state='disabled')
	if down_sampling_var.get() == 3:
		numFramesExtractVar.config(state='disabled')
		downsamplingRatioVar.config(state='disabled')
		fpsVar.config(state='normal')
def display_close_up():
	if runcloseupVar.get():
		closeupThresholdVar.config(state='normal')
		closeupDropDown.config(state='normal')
	else:
		closeupThresholdVar.config(state='disabled')
		closeupDropDown.config(state='disabled')
def change_blur_label(model):
	if model == "SVD":
		blur_threshold_label_text.set("SVD Blur threshold: value between [0,1]")
		blurThresholdVar.delete(0, 'end')
		blurThresholdVar.insert(10, "0.6")
	elif model == "Laplacian":
		blur_threshold_label_text.set("Laplacian blur threshold: any float")
		blurThresholdVar.delete(0, 'end')
		blurThresholdVar.insert(10, "1000")


master = tk.Tk()
master.winfo_toplevel().title("HOST-ATS Graphical User Interface")
faceVar = StringVar(master)
faceVar.set("dnn") # default value
logoVar = StringVar(master)
logoVar.set("Eliteserien2019")
blurVar = StringVar(master)
blurVar.set("SVD")
iqaVar = StringVar(master)
iqaVar.set("Ocampo")
closeupVar = StringVar(master)
closeupVar.set("Surma")
blur_threshold_label_text = StringVar(master)
blur_threshold_label_text.set("SVD Blur threshold: value between [0,1]")
runIQAVar = BooleanVar(value=True)
runLogoVar = BooleanVar(value=True)
runFaceVar = BooleanVar(value=True)
runBlurVar = BooleanVar(value=True)
runcloseupVar = BooleanVar(value=True)
video_file_text = StringVar(value="No file selected")
generate_response_text = StringVar(value="")
generate_button_state = StringVar(value='disabled')
pre_processing = LabelFrame(master, text="Step 1. Pre-processing", font=('Arial', 20), padx=10, pady=10)
pre_processing.grid( padx=10, pady=10)
trimming = LabelFrame(pre_processing, text="1a. Trimming", padx=10, pady=10)
trimming.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(trimming, text= "Annotation mark").grid(sticky=tk.W)
tk.Label(trimming, text= "Drop frames X seconds before the annotation").grid(sticky=tk.W)
tk.Label(trimming, text= "Drop frames Y seconds after the annotation").grid(sticky=tk.W)
tk.Label(trimming, text="Seconds to cut in start of video").grid(sticky=tk.W)
tk.Label(trimming, text="Seconds to cut in end of video").grid(sticky=tk.W)
down_sampling = LabelFrame(pre_processing, text="1b. Down-sampling", padx=10, pady=10)
down_sampling.grid(padx=10, pady=10, sticky=tk.W)
down_sampling_var = IntVar()
down_sampling_var.set(1)
tk.Radiobutton(down_sampling, text="Number of Frames to extract", variable=down_sampling_var, value=1, command=disable_other_entries_downsampling).grid(sticky=tk.W)
tk.Radiobutton(down_sampling, text="Downsampling ratio: value between [0,1]", variable=down_sampling_var, value=2, command=disable_other_entries_downsampling).grid(sticky=tk.W)
tk.Radiobutton(down_sampling, text="Frame per second", variable=down_sampling_var, value=3, command=disable_other_entries_downsampling).grid(sticky=tk.W)
down_scaling = LabelFrame(pre_processing, text="1c. Down-scaling", padx=10, pady=10)
down_scaling.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(down_scaling, text="For internal processing (%)").grid(sticky=tk.W)
tk.Label(down_scaling, text="For output image (%)").grid(sticky=tk.W)

content_analysis = LabelFrame(master, text="Step 2. Content Analysis", font=('Arial', 20), padx=10, pady=10)
content_analysis.grid(row=0, column=3, padx=10, pady=10, sticky=tk.N)
logo_detection = LabelFrame(content_analysis, text="2a. Logo Detection", padx=10, pady=10)
logo_detection.grid(padx=10, pady=10, sticky=tk.W)
runLogoCheckbutton = tk.Checkbutton(logo_detection, command=display_logo_det_models, text="Run Logo Detection", variable=runLogoVar)
tk.Label(logo_detection, text="Logo Detection Model").grid(row=1, sticky=tk.W)
tk.Label(logo_detection, text="Logo detection threshold (%)").grid(row=2, sticky=tk.W)
close_up_shot_detection = LabelFrame(content_analysis, text="2b. Close-up Shot Detection", padx=10, pady=10)
close_up_shot_detection.grid(padx=10, pady=10, sticky=tk.W)
runCloseCheckbutton = tk.Checkbutton(close_up_shot_detection, command=display_close_up,text="Run Close-up Shot Detection", variable=runcloseupVar)
tk.Label(close_up_shot_detection, text="Close-up shot detection model").grid(row=1, sticky=tk.W)
tk.Label(close_up_shot_detection, text="Close-up detection threshold (%)").grid(row=2, sticky=tk.W)
face_detection = LabelFrame(content_analysis, text="2c. Face Detection", padx=10, pady=10)
face_detection.grid(padx=10, pady=10, sticky=tk.W)
runFaceCheckbutton = tk.Checkbutton(face_detection, command=display_face_det_models, text="Run Face Detection", variable=runFaceVar)
tk.Label(face_detection, text="Face Detection Model").grid(row=1, sticky=tk.W)

image_quality_analysis = LabelFrame(master, text="Step 3. Image Quality Analysis", font=('Arial', 20), padx=10, pady=10)
image_quality_analysis.grid(row=0, column=5, padx=10, pady=10, sticky=tk.N)
image_quality_prediction = LabelFrame(image_quality_analysis, text="3a. Image Quality Prediction", padx=10, pady=10)
image_quality_prediction.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(image_quality_prediction, text="Image quality prediction model").grid(row=1, sticky=tk.W)

tk.Label(image_quality_prediction, text="BRISQUE threshold value").grid(row=2, sticky=tk.W)
blur_detection = LabelFrame(image_quality_analysis, text="3b. Blur Detection", padx=10, pady=10)
blur_detection.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(blur_detection, text="Blur Detection Model").grid(row=1, sticky=tk.W)
blur_threshold_label = tk.Label(blur_detection, textvariable=blur_threshold_label_text).grid(row=2, sticky=tk.W)

file_processing = LabelFrame(master, padx=10, pady=10)
file_processing.grid(row=1, column=3, columnspan=4, padx=10, pady=10, sticky=tk.N+tk.W+tk.E)
video_file_text_label = tk.Label(file_processing, textvariable=video_file_text, font=('Arial', 8)).grid(sticky=tk.W, column=1)
generate_button = tk.Button(file_processing, text='Generate', command=generate).grid(sticky=tk.W)
generate_response = tk.Label(file_processing, textvariable=generate_response_text, fg="#FF0000",font=('Arial', 8)).grid(sticky=tk.W, column=1, row=1)

annotationMarkVar = tk.Entry(trimming, width=5)
beforeAnnotationVar = tk.Entry(trimming, width=5)
afterAnnotationVar = tk.Entry(trimming, width=5)
cutStartVar = tk.Entry(trimming, width=5)
cutEndVar = tk.Entry(trimming, width=5)
numFramesExtractVar = tk.Entry(down_sampling, width=5)
downsamplingRatioVar = tk.Entry(down_sampling, width=5, state='disabled')
fpsVar = tk.Entry(down_sampling, width=5, state='disabled')
internalProcessingVar = tk.Entry(down_scaling, width=5)
outputImageVar = tk.Entry(down_scaling, width=5)

logoThresholdVar = tk.Entry(logo_detection, width=5)
closeupThresholdVar = tk.Entry(close_up_shot_detection, width=5)

faceDropDown = tk.OptionMenu(face_detection, faceVar, "haar", "dlib", "mtcnn", "dnn")
logoDropDown = tk.OptionMenu(logo_detection, logoVar, "Eliteserien2019", "Soccernet")
blurDropDown = tk.OptionMenu(blur_detection, blurVar, "SVD", "Laplacian", command=change_blur_label)
iqaDropDown = tk.OptionMenu(image_quality_prediction, iqaVar, "Ocampo")
closeupDropDown = tk.OptionMenu(close_up_shot_detection, closeupVar, "Surma")
fileName = tk.Entry(file_processing)
runIQACheckbutton = tk.Checkbutton(image_quality_prediction, command=display_brisque_thr, text="Run Image Quality Prediction", variable=runIQAVar)
brisqueVar = tk.Entry(image_quality_prediction, width=5)
runBlurCheckbutton = tk.Checkbutton(blur_detection, command=display_blur_thr, text="Run Blur Detection", variable=runBlurVar)
blurThresholdVar = tk.Entry(blur_detection, width=8)
numFramesExtractVar.insert(10, "50")
cutStartVar.insert(10, "0")
cutEndVar.insert(10, "0")
internalProcessingVar.insert(10, "50")
outputImageVar.insert(10, "100")
logoThresholdVar.insert(10, "10")
closeupThresholdVar.insert(10, "75")
brisqueVar.insert(10, "35")
blurThresholdVar.insert(10, "0.6")


annotationMarkVar.grid(row=0, column=1)
beforeAnnotationVar.grid(row=1, column=1)
afterAnnotationVar.grid(row=2, column=1)
cutStartVar.grid(row=3, column=1)
cutEndVar.grid(row=4, column=1)
numFramesExtractVar.grid(row=0, column=1)
downsamplingRatioVar.grid(row=1, column=1)
fpsVar.grid(row=2, column=1)
internalProcessingVar.grid(row=0, column=1)
outputImageVar.grid(row=1, column=1)

runLogoCheckbutton.grid(row=0, column=0, sticky=tk.W)
logoDropDown.grid(row=1, column=1)
logoThresholdVar.grid(row=2, column=1)

runCloseCheckbutton.grid(row=0, column=0)
closeupDropDown.grid(row=1, column=1)
closeupThresholdVar.grid(row=2, column=1)

runFaceCheckbutton.grid(row=0, column=0)
faceDropDown.grid(row=1, column=1)

runIQACheckbutton.grid(row=0, column=0)
iqaDropDown.grid(row=1, column=1)
brisqueVar.grid(row=2, column=1)

runBlurCheckbutton.grid(row=0, column=0, sticky=tk.W)
blurDropDown.grid(row=1,column=1)
blurThresholdVar.grid(row=2,column=1)




tk.Button(file_processing,
		text='Select File',
		command=open_file).grid(row=0, sticky=tk.W)

ratio = 1.7777778
height = 140
width = int(height * ratio)
staticLabel = tk.Label(file_processing, text="Static thumbnail:").grid(row=3, column=0)
canvas1 = Canvas(file_processing, width=width, height=height)
canvas1.grid(row=4, column=0)
mlBasedLabel = tk.Label(file_processing, text="ML-based thumbnail:").grid(row=3, column=1)
canvas2 = Canvas(file_processing, width=width, height=height)
canvas2.grid(row=4, column=1)

master.mainloop()

tk.mainloop()
