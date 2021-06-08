import cv2
from aug.core.utils import read_image,show
import aug
import glob
import argparse

# imgs_path = "D:\\instals\\aug-master\\images\\*.*"
# out_path = "./res/"

parser = argparse.ArgumentParser()
parser.add_argument("--imgPath",type=str,default="D:\\instals\\aug-master\\images",help="should be: D:\\instals\\aug-master\\images")
parser.add_argument("--out",type=str,default="./res/",help="to save results")
parser.add_argument("--numSteps",nargs="+",type=int,default=[50,17],help="for gridDistortion")
parser.add_argument("--distorLimit",nargs="+",type=float,default=[0.5,0.1],help="for gridDistortion")

parser.add_argument("--darkness",type=int,default=50,help="for flashlight")
parser.add_argument("--alpha",type=float,default=0.12,help="for flashligh")

getArgs = parser.parse_args()
imgs_path = getArgs.imgPath + "\\*.*"
out_path = getArgs.out

dist_num_steps = getArgs.numSteps
dist_limit = getArgs.distorLimit

flash_darknesss = getArgs.darkness
flash_alpha = getArgs.alpha
# print(dist_limit)


class Augment(aug.Pipeline):
    def __init__(self):
        super(Augment, self).__init__()

        self.seq = aug.Sequential(

            aug.GridDistortion(tuple(dist_num_steps), distort_limit=tuple(dist_limit)),
            aug.Flashlight(bg_darkness=flash_darknesss, alpha=flash_alpha), 

        )
    def apply(self,sample):

        return self.seq.apply(sample).image

augment = Augment()

def read_all_imgs(data_path,out_path):
    c = 1
    for fl in glob.glob(data_path):
        get_orig_name = fl.split('\\')[4].split('.')[0]
        # print(get_orig_name)
        img_read = cv2.imread(fl)
        img = cv2.cvtColor(img_read,cv2.COLOR_BGR2RGB)

        augment_imgs = augment.apply(aug.Sample(image=img))
        cv2.imwrite(f"{out_path}aug_{c}_img_{get_orig_name}.jpg",augment_imgs)
        print(f"Done {c}!")
        c+=1





read_all_imgs(imgs_path,out_path)