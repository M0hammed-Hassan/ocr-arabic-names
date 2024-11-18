# OCR Arabic Names
This repository demonstrates a custom Optical Character Recognition (OCR) pipeline to extract Arabic names from images. It focuses on building a complete end-to-end solution, including preprocessing, training a neural network, and deploying the model via a Django API endpoint.

## Main Repository Structur
```
ocr_arabic_name/
│
├── data/
│   ├── name1.png
│   ├── name1.txt
│   └── ...
│
├── src/
│   ├── configs.py           
│   ├── dataset.py    
│   ├── main.py           
│   ├── model.py     
│   ├── seeding.py            
│   ├── utils.py         
│
├── app/
│   ├── manager.py
│   └── ...
│
├── checkpoints/  
│   ├── best.pth.tar  
│   ├── last.pth.tar
│
├── logs/  
│   ├── run_date
│   └── ...
│     
├── run.sh               
│
├── report.pdf  
│         
├── README.md    

```

## To Use My Checkpoints
- [Downlad Checkpoints](https://drive.google.com/file/d/15R76NcTc-ZULc_QX-JM7dvwDvfDJVS-E/view?usp=sharing)

## To Start Training The Pipeline
- Using shell command:

     ```./run.sh```

- Using python command:

     ```python src/main.py```

## To Run The App
- Change the directory to app

     ```cd app```

- Run this command with a port like 8000

     ``` python manage.py runserver 8000 ```

Your Django API should now be running at ```http://127.0.0.1:8000/api/predict/```