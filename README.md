# Preparation

1. Download trained weights `net_e382_l0.hdf5` from [Google Drive](https://drive.google.com/file/d/182JeC23uO6DJKDbAlD3f4hP6Lsib4CpX/view?usp=share_link)
2. Prepare project files  
   * Prepare testing images and put it into ./images/test_data 
   * Put the trained weights at ./
```
./
├── images
│   └── test_data
│   └── test_results
│── net_e382_l0.hdf5
│── run.py
```
3. Setup venv requirements.txt: `pip install -r requirements.txt`
4. Execute `run.py` then all the images in `./images/test_data` will be processed and output to `./images/test_results` with face blurs.

Video Instruction is at [here](https://drive.google.com/file/d/15iaa_21r_JZPWMu0h3tGVcMb2RvlQUL7/view?usp=share_link) on Google Drive.
