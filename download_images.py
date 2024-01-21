from simple_image_download import simple_image_download as si 
response = si.simple_image_download
keyword = ["building workers"]
for kw in keyword:
	response().download(kw,200)