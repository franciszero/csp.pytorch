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
   * Using `'device=cpu'` instead of `'device=cuda:0'` if CUDA is not available 
5. Execute `run.py` then all the images in `./images/test_data` will be processed and output to `./images/test_results` with face blurs.
