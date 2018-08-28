# ActionRecognition

## two-stream-action-recognition

1. Clone this repo
```
https://github.com/jeffreyhuang1/two-stream-action-recognition.git
```

2. Put the jupyter notebooks into the above repo folder.

Put EasyStart and TestVideo into **two-stream-action-recognition**.

3. Download data

**Attention**: The combined zip file takes more than 20GB, you can try this algorithm just on RGB images. The cat process needs large space of memory.
- RGB images
```
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
unzip ucf101_jpegs_256.zip
```
- Optical Flow
```
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003

cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
unzip ucf101_tvl1_flow.zip
```
4. Model

RGB images are trained with **Spatial CNN** while Optical Flow images are trained with **Motion CNN**, please refer to the two-stream-action-recognition repo for detail.

5. Pre-trained Model

The author put pre-trained model on google drive. We can use the pre-trained model because the training process takes a long time.
- [Spatial resent101](https://drive.google.com/drive/folders/1gVB5StqgoDJ3IxHUn7zoTzTNxzz3du3d?usp=sharing)
- [Motion resent101](https://drive.google.com/drive/folders/1z3fYUOJx_l3BW-NSb7ti0DsyGLFk6Z7J?usp=sharing)

Download these models to your machine.

6. Train and Test

Modify one function in **dataloader/spatial_dataloader.py** at line 22.

``` python
    def load_ucf_image(self, video_name, index):
        path = self.root_dir + '/v_' + video_name + '/frame'
        img = Image.open(path +str(index).zfill(6)+'.jpg')
        transformed_img = self.transform(img)
        img.close()
        return transformed_img
```
Open **EasyStart** notebook, and run cells carefully.

7. Test on one video

In order to extract a video to frames, we should install ffmpeg.
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd **./ffmpeg folder/**
sudo cp ffmpeg ffprobe /usr/local/bin
```

Open **TestVideo** notebook, and run cells carefully.