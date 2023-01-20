from PIL import Image
import os



width = 256
height = 256



def generate_name(index):
    new_name = ""
    if index < 10:
        new_name ="0000" + str(index) + ".jpg"
    elif index < 100:
        new_name ="000" + str(index) + ".jpg"
    elif index < 1000:
        new_name = "00" + str(index) + ".jpg"
    elif index < 10000:
        new_name = "0" + str(index) + ".jpg"
    else:
        new_name = str(index)+".jpg"
    return new_name

if __name__ == '__main__':   
    id = 0
    data_dir = "brain_organoid_out"
    out_dir = "brain"
    img_list = os.listdir(data_dir)
    print(img_list)
    for img_file in img_list:
        if img_file.endswith(".png") or img_file.endswith(".jpg") or img_file.endswith("tif"):
            img = Image.open(os.path.join(data_dir,img_file))
            try:
                new_img = img.resize((width,height),Image.BILINEAR).convert('RGB')
                new_name = generate_name(id)
                id = id + 1
                new_img.save(os.path.join(out_dir,new_name))
                print(id)
            except Exception as e:
                print(e)