import os
import imageio

# image folder
#image_folder='images'
image_folder = 'TTC_medianDist_FAST-BRIEF'

# set frame rate [frames / second]
fps = 1

# get image files from image folder
image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]

# sort image file list by names in ascending order
image_files.sort()
print("List image files in image folder:\n")
for (i, img) in enumerate(image_files):
    print("Image file no. {}: {}\n".format(i, img))

# make a movie
#movie_filepath = 'my_movie.gif'
movie_filepath = 'fig_1_final_results_ttc_method4_FAST-BRIEF.gif'
with imageio.get_writer(movie_filepath, mode='I') as writer:
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)
    print("Movie writte to: {}".format(movie_filepath))
