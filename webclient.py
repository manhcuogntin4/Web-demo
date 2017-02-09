import SOAPpy
import os
import urllib, cStringIO
from PIL import Image
import requests
from io import BytesIO

# response = requests.get("http://lucadeparis.free.fr/copywrong/lutonadio/gedeon_lutonadio_cni.jpg")
# img = Image.open(BytesIO(response.content))
# im=img.save("temp.png")

server = SOAPpy.SOAPProxy("http://127.0.0.1:5000/")

# filename=""
# cwd = os.getcwd()
# # print cwd
# path=os.path.join(cwd, 'temp.png')
# file = cStringIO.StringIO(urllib.urlopen("http://lucadeparis.free.fr/copywrong/lutonadio/gedeon_lutonadio_cni.jpg").read())
# img = Image.open(file)
cnis, result = server.ocr("http://2.bp.blogspot.com/_b4e7-vewGoI/TQSH3AKlwYI/AAAAAAAAAAQ/Ov4FGLKN9fQ/s1600/CNI+Recto.jpg")
print cnis, result

