import os
import moviepy.video.io.ImageSequenceClip

image_folder = r"C:\Users\krk1g19\github\prj_3\Datasets\Graph4Data"
fps = 24
image_files = []
for i in range(400):
    fname = image_folder + "\graphviz_" + str(i) + ".png"
    if os.path.isfile(fname):
        image_files.append(fname)
# image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile("my_video.mp4")
