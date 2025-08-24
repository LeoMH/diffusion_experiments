ffmpeg -framerate 2 -pattern_type glob -i 'vid_result/*.png' -vcodec libx264 -acodec aac out.mp4
ffmpeg -i out.mp4 -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac out_ppt.mp4
