# Project Kitty: Pose Estimation and Corrective Analytics for Swimming
## Naina Bhalla (Roll Number: 240674)

## Overview

Project Kitty leverages pose estimation, machine learning, and video processing to provide real-time feedback on swimming technique—including kick, breathing, hand entry, and posture. The pipeline is inspired by the futuristic sports tech seen in Supa Strikas.

---

## Features

- Automated scraping and downloading of swimming videos from YouTube
- Video standardization (resolution, frame rate, aspect ratio)
- Pose keypoint extraction and motion segmentation using MediaPipe
- Feature engineering and labeled dataset generation
- Model training (Random Forest, SVM, MLP) for action classification
- Real-time analytics and feedback overlay on video
- Modular, privacy-preserving design

---

## Directory Structure

ProjectKitty/ <br>
├── README.md <br>
├── requirements.txt <br>
├── videos/ # Raw downloaded videos....Not kept in the directory/drive(Too large) <br> 
├── standardized_videos/ # Preprocessed videos...... [Drive link](https://drive.google.com/drive/folders/1dkJblUBHRqyztgMOAv28MtFGMUaJZ0D-?usp=sharing) <br>
├── segmented_videos/ # JSONs with keypoints/features/labels <br>
├── models/ # Trained model files (.pkl) <br>
├── chosen_models/ # Models with better performance chosen <br>
├── src/ <br>
│ ├── video_scraper.py <br>
│ ├── video_standardizer.py <br>
│ ├── keypoint_extractor.py <br>
│ ├── model_training.ipynb <br>
│ └── analyzer.py <br>
├── test_videos/ <br> #Contains videos used for displayed results
└── results/ # Output videos, reports <br>


---

## Environment Setup

### 1. Clone the Repository

git clone https://github.com/naina-bhalla/ProjectKitty.git <br>
cd ProjectKitty


### 2. Create a Virtual Environment

python3 -m venv venv <br>
source venv/bin/activate <br>


### 3. Install Dependencies

pip install -r requirements.txt


### 4. Install FFmpeg

- **Ubuntu/Debian:**  
  `sudo apt-get update && sudo apt-get install ffmpeg`
- **Mac (Homebrew):**  
  `brew install ffmpeg`

---

## How to Run: 

### 1. Download Swimming Videos

python src/video_scraper.py

Videos will be saved in `videos/` organized by stroke.

### 2. Standardize Videos

python src/video_standardizer.py

Standardized videos will be saved in `standardized_videos/` by stroke.

### 3. Extract Keypoints and Segment

Open and run `src/keypoint_extractor.ipynb` to extract keypoints for each motion by stroke.  
JSONs with keypoints/features/labels for each will be stored in `segmented_videos/`.

### 4. Model Training

Open and run `src/model_training.ipynb` to train classifiers for each motion.  
Trained models will be saved in `models/`.

### 5. Run Analytics and Feedback Overlay

I chose the MLP models and inserted them into `chosen_models/` to use for predictions.

python src/analyzer.py --input test.mp4 --model_dir models/ --output output.avi

python src/analyzer.py --input test1.mp4 --model_dir models/ --output output1.avi

- The output video will have all feedback overlays for each motion.

---

## Model Performance

Below are the label distributions and evaluation metrics for each motion, using SVM, Random Forest, and MLP classifiers.  
**Random Forest** consistently achieved the highest accuracy and was chosen as the default model for deployment.

---

### **KICK**
**Label distribution:**  
`inconsistent_kick`: 1414  
`consistent_kick`: 950


Classification reports: 

#### SVM
```
                   precision    recall  f1-score   support

  consistent_kick       0.72      0.60      0.65       199
inconsistent_kick       0.74      0.83      0.78       274

         accuracy                           0.73       473
        macro avg       0.73      0.72      0.72       473
     weighted avg       0.73      0.73      0.73       473
```

#### RandomForest
```
                   precision    recall  f1-score   support

  consistent_kick       1.00      1.00      1.00       199
inconsistent_kick       1.00      1.00      1.00       274

         accuracy                           1.00       473
        macro avg       1.00      1.00      1.00       473
     weighted avg       1.00      1.00      1.00       473
```

#### MLP
```
                   precision    recall  f1-score   support

  consistent_kick       0.98      0.99      0.98       199
inconsistent_kick       0.99      0.98      0.99       274

         accuracy                           0.99       473
        macro avg       0.98      0.99      0.98       473
     weighted avg       0.99      0.99      0.99       473
```


---

### **BREATHING**
**Label distribution:**  
`balanced_breathing`: 445  
`breathing_bias_detected`: 245


Classification reports: 

#### SVM
```
                         precision    recall  f1-score   support

     balanced_breathing       0.98      1.00      0.99        91
breathing_bias_detected       1.00      0.96      0.98        47

               accuracy                           0.99       138
              macro avg       0.99      0.98      0.98       138
           weighted avg       0.99      0.99      0.99       138
```

#### RandomForest
```
                         precision    recall  f1-score   support

     balanced_breathing       1.00      1.00      1.00        91
breathing_bias_detected       1.00      1.00      1.00        47

               accuracy                           1.00       138
              macro avg       1.00      1.00      1.00       138
           weighted avg       1.00      1.00      1.00       138
```

#### MLP
```
                         precision    recall  f1-score   support

     balanced_breathing       0.99      1.00      0.99        91
breathing_bias_detected       1.00      0.98      0.99        47

               accuracy                           0.99       138
              macro avg       0.99      0.99      0.99       138
           weighted avg       0.99      0.99      0.99       138
```


---

### **HAND ENTRY**
**Label distribution:**  
`smooth_hand_entry`: 1281  
`unstable_hand_entry`: 461  
`asymmetric_hand_entry`: 415


Classification reports: 

#### SVM
```
                       precision    recall  f1-score   support

asymmetric_hand_entry       1.00      0.96      0.98        90
    smooth_hand_entry       0.97      1.00      0.98       252
  unstable_hand_entry       1.00      0.94      0.97        90

             accuracy                           0.98       432
            macro avg       0.99      0.97      0.98       432
         weighted avg       0.98      0.98      0.98       432
```

#### RandomForest
```
                       precision    recall  f1-score   support

asymmetric_hand_entry       1.00      1.00      1.00        90
    smooth_hand_entry       1.00      1.00      1.00       252
  unstable_hand_entry       1.00      1.00      1.00        90

             accuracy                           1.00       432
            macro avg       1.00      1.00      1.00       432
         weighted avg       1.00      1.00      1.00       432
```

#### MLP
```
                       precision    recall  f1-score   support

asymmetric_hand_entry       0.99      0.99      0.99        90
    smooth_hand_entry       1.00      1.00      1.00       252
  unstable_hand_entry       1.00      0.99      0.99        90

             accuracy                           1.00       432
            macro avg       0.99      0.99      0.99       432
         weighted avg       1.00      1.00      1.00       432
```


---

### **OVERALL POSTURE**
**Label distribution:**  
`stable_posture`: 460  
`unstable_posture`: 281


Classification reports: 

#### SVM
```
                  precision    recall  f1-score   support

  stable_posture       0.98      1.00      0.99        87
unstable_posture       1.00      0.97      0.98        62

        accuracy                           0.99       149
       macro avg       0.99      0.98      0.99       149
    weighted avg       0.99      0.99      0.99       149
```

#### RandomForest
```
                  precision    recall  f1-score   support

  stable_posture       1.00      1.00      1.00        87
unstable_posture       1.00      1.00      1.00        62

        accuracy                           1.00       149
       macro avg       1.00      1.00      1.00       149
    weighted avg       1.00      1.00      1.00       149
```

#### MLP
```
                  precision    recall  f1-score   support

  stable_posture       0.98      1.00      0.99        87
unstable_posture       1.00      0.97      0.98        62

        accuracy                           0.99       149
       macro avg       0.99      0.98      0.99       149
    weighted avg       0.99      0.99      0.99       149
```


---

**Summary:**  
- Random Forest achieved perfect accuracy on all motions but was likely overfitting.
- SVM and MLP also performed strongly, with MLP nearly matching Random Forest.
- The dataset is well-balanced for each motion, and the models generalize well.


- See `results/` for annotated sample videos and feedback overlays.

---

## Code Structure

| Script/Notebook           | Purpose                                              |
|---------------------------|-----------------------------------------------------|
| video_scraper.py          | Scrape and download swimming videos from YouTube    |
| video_standardizer.py     | Standardize video resolution, FPS, aspect ratio     |
| keypoint_extractor.py     | Extract pose keypoints, segment, and label motions  |
| model_training.ipynb      | Train and evaluate ML models for each motion        |
| analyzer.py               | Overlay feedback on video using trained models      |

---

## Evaluation & Comparative Analysis

- All models (Random Forest, SVM, MLP) were evaluated using accuracy, precision, recall, and F1-score.
- Random Forest consistently achieved the highest scores but it was likely overfitting. Hence, the chosen model was MLP.

---


