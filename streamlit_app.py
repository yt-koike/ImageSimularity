import streamlit as st
from sim import ImageSimularity

st.title("画像類似度比較 Web アプリ")

filepath1 = st.file_uploader('画像をアップロードしてください.', type=['jpg', 'jpeg', 'png'],key=1)
filepath2 = st.file_uploader('画像をアップロードしてください.', type=['jpg', 'jpeg', 'png'],key=2)

if filepath1 is not None and filepath2 is not None:
    st.image([filepath1,filepath2], width=300)
   
    imSim = ImageSimularity()
    img0 =imSim.loadImage(filepath1)
    img1 =imSim.loadImage(filepath2)

    # Lower means further/more different. Higher means more similar.
    st.write("PSNR",round(imSim.PSNR_rgb(img0,img1),4))
    st.write("SSIM",round(imSim.SSIM_rgb(img0,img1),4))
    # Higher means further/more different. Lower means more similar.
    st.write("LPIPS", round(imSim.LPIPS_rgb(img0, img1),4))
