from PIL import Image
from test_1 import proc
def generate():
    img1, img2 = Image.open('demo/res.jpg'), Image.open('demo/res.jpg')
    size1, size2 = img1.size, img2.size
    joint = Image.new('F', (size1[0] + size2[0], size1[1]))
    loc1, loc2 = (0, 0), (size1[0], 0)
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    joint = joint.convert('RGB')
    joint.save("demo/Img/test/example.jpg")
    proc()
if __name__=='__main__':
    generate()    