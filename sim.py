import math
import lpips
import torch
import numpy as np
from PIL import Image

class ImageSimularity:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='alex')
        
    def loadImage(self,path):
        return torch.from_numpy(np.array(Image.open(path).convert('RGB'), dtype = "float32")).clone()
        
    def PSNR_mono(self,img1, img2):
        loss = torch.mean((img1 - img2)**2)
        psnr = 10. * math.log(255**2/loss) /math.log(10.)
        return psnr
    
    def PSNR_rgb(self,img1, img2):
        loss = torch.mean((img1 - img2)**2)
        psnr = 10. * math.log(255**2/loss) /math.log(10.)
        return psnr

    def SSIM_mono(self,monoImg1, monoImg2, C1 = None, C2 = None):
        # https://doi.org/10.1109/TIP.2003.819861
        L = 255
        K1, K2 = 0.01, 0.03
        if C1 is None:
            C1 = (K1*L)**2
        if C2 is None:
            C2 = (K2*L)**2
        N = monoImg1.shape[0] * monoImg1.shape[1]
        mean1 = torch.mean(monoImg1)
        sd1 = (torch.sum((monoImg1 - mean1)**2)/N)**(1/2)
        mean2 = torch.mean(monoImg2)
        sd2 = (torch.sum((monoImg2 - mean2)**2)/N)**(1/2)
        coSd = torch.sum(torch.multiply(
            monoImg1 - mean1, monoImg2 - mean2))/N
        return float((2*mean1*mean2 + C1)*(2*coSd + C2)/(mean1**2+mean2**2+C1)/(sd1**2+sd2**2+C2))
    
    def SSIM_rgb(self,img1, img2):
        r1 = img1[:,:,0]
        r2 = img2[:,:,0]
        g1 = img1[:,:,1]
        g2 = img2[:,:,1]
        b1 = img1[:,:,2]
        b2 = img2[:,:,2]
        return (self.SSIM_mono(r1,r2)+self.SSIM_mono(g1,g2)+self.SSIM_mono(b1,b2))/3
  
    def LPIPS_mono(self,monoImg0, monoImg1,loss_fn_alex=None):
     return float(self.loss_fn_alex(monoImg0, monoImg1)[0][0][0][0])

    def LPIPS_rgb(self,img0, img1):
        return sum([self.LPIPS_mono(img0[:,:,channel],img1[:,:,channel]) for channel in range(3)])/3

if __name__ == "__main__":
    imSim = ImageSimularity()
    path_img0 = "a.png"
    img0 =imSim.loadImage(path_img0)
    paths = ["b.png"]
    for path_img1 in paths:
        print(f"Comparing {path_img0} and {path_img1}")
        img1 =imSim.loadImage(path_img1)

        # Lower means further/more different. Higher means more similar.
        print("PSNR",imSim.PSNR_rgb(img0,img1))
        print("SSIM",imSim.SSIM_rgb(img0,img1))
        # Higher means further/more different. Lower means more similar.
        print("LPIPS", imSim.LPIPS_rgb(img0, img1),)