# Preparation

1. Setup venv requirements.txt: `pip install -r requirements.txt`
2. Download trained weights `net_e382_l0.hdf5` from [Google Drive](https://drive.google.com/file/d/182JeC23uO6DJKDbAlD3f4hP6Lsib4CpX/view?usp=share_link)
3. Prepare project  
   * Prepare testing images and put it into ./images/test_data 
   * Put the trained weights at ./
```
./
├── images
│   └── test_data
│   └── test_results
│── net_e382_l0.hdf5
│── face_detection_and_anonymizing.py
```
4. Modify `run.py` 
   * Use `'device=cuda:0'` instead of `'device=cpu'` if CUDA is available 
5. Execute `run.py` then all the images in `./images/test_data` will be processed and the outcomes will be saved into `./images/test_results` with face blurs.
