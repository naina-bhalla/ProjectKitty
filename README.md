# Project Kitty: Pose Estimation and Corrective Analytics for Swimming
**Author:** Naina Bhalla (Roll Number: 240674)

## Overview
**Project Kitty** leverages pose estimation, machine learning, and video processing to provide real-time feedback on swimming technique—including kick, breathing, hand entry, and overall posture. The pipeline is inspired by the futuristic sports tech seen in *Supa Strikas*.

---

## Features
*   **Automated Scraping:** Downloads swimming videos directly from YouTube.
*   **Video Standardization:** Normalizes resolution, frame rate, and aspect ratio.
*   **Pose Estimation:** Extracts keypoints and segments motion using Google MediaPipe.
*   **Machine Learning:** Trains classifiers (Random Forest, SVM, MLP) on engineered features.
*   **Real-Time Analytics:** Overlays technique feedback directly onto the video.
*   **Privacy-Preserving:** Modular design keeps video processing secure and efficient.

---

## Directory Structure

```text
ProjectKitty/
├── README.md
├── requirements.txt
├── videos/                 # Raw downloaded videos (Not kept in repo due to size)
├── standardized_videos/    # Preprocessed videos (See Drive link below)
├── segmented_videos/       # JSONs containing keypoints, features, and labels
├── models/                 # Trained model files (.pkl)
├── chosen_models/          # The final deployed models (MLP)
├── src/                    # Source code
│   ├── video_scraper.py
│   ├── video_standardizer.py
│   ├── keypoint_extractor.py
│   ├── model_training.ipynb
│   └── analyzer.py
├── test_videos/            # Sample videos used for displayed results
└── results/                # Output videos with feedback overlays and reports

```

> **Dataset Access:** [Standardized Videos Google Drive Link](https://drive.google.com/drive/folders/1dkJblUBHRqyztgMOAv28MtFGMUaJZ0D-?usp=sharing)

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/naina-bhalla/ProjectKitty.git](https://github.com/naina-bhalla/ProjectKitty.git)
cd ProjectKitty

```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Install FFmpeg

* **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install ffmpeg`
* **Mac (Homebrew):** `brew install ffmpeg`

---

## How to Run

**1. Download Swimming Videos**

```bash
python src/video_scraper.py

```

*Videos will be saved in `videos/`, organized by stroke.*

**2. Standardize Videos**

```bash
python src/video_standardizer.py

```

*Standardized videos will be saved in `standardized_videos/` by stroke.*

**3. Extract Keypoints and Segment**
Open and run `src/keypoint_extractor.ipynb` to extract keypoints for each motion by stroke. JSONs with keypoints/features/labels will be stored in `segmented_videos/`.

**4. Model Training**
Open and run `src/model_training.ipynb` to train classifiers for each motion. Trained models will be saved in `models/`.

**5. Run Analytics and Feedback Overlay**
The MLP models have been selected for deployment and placed in `chosen_models/`.

```bash
python src/analyzer.py --input test.mp4 --model_dir chosen_models/ --output results/output.avi
python src/analyzer.py --input test1.mp4 --model_dir chosen_models/ --output results/output1.avi

```

*The output video will display real-time feedback overlays for each analyzed motion.*

---

## Model Performance & Evaluation

All models (Random Forest, SVM, MLP) were evaluated using accuracy, precision, recall, and F1-score. While Random Forest consistently achieved perfect accuracy (indicating likely overfitting), **MLP** performed exceptionally well without overfitting and was chosen as the default deployment model.

* The dataset is well-balanced for each motion.
* See the `results/` folder for annotated sample videos and feedback overlays.

**Label Distribution:**

* `inconsistent_kick`: 1414
* `consistent_kick`: 950

```text
#### SVM
                   precision    recall  f1-score   support
  consistent_kick       0.72      0.60      0.65       199
inconsistent_kick       0.74      0.83      0.78       274
         accuracy                           0.73       473

#### RandomForest
                   precision    recall  f1-score   support
  consistent_kick       1.00      1.00      1.00       199
inconsistent_kick       1.00      1.00      1.00       274
         accuracy                           1.00       473

#### MLP
                   precision    recall  f1-score   support
  consistent_kick       0.98      0.99      0.98       199
inconsistent_kick       0.99      0.98      0.99       274
         accuracy                           0.99       473

```

**Label Distribution:**

* `balanced_breathing`: 445
* `breathing_bias_detected`: 245

```text
#### SVM
                         precision    recall  f1-score   support
     balanced_breathing       0.98      1.00      0.99        91
breathing_bias_detected       1.00      0.96      0.98        47
               accuracy                           0.99       138

#### RandomForest
                         precision    recall  f1-score   support
     balanced_breathing       1.00      1.00      1.00        91
breathing_bias_detected       1.00      1.00      1.00        47
               accuracy                           1.00       138

#### MLP
                         precision    recall  f1-score   support
     balanced_breathing       0.99      1.00      0.99        91
breathing_bias_detected       1.00      0.98      0.99        47
               accuracy                           0.99       138

```

**Label Distribution:**

* `smooth_hand_entry`: 1281
* `unstable_hand_entry`: 461
* `asymmetric_hand_entry`: 415

```text
#### SVM
                       precision    recall  f1-score   support
asymmetric_hand_entry       1.00      0.96      0.98        90
    smooth_hand_entry       0.97      1.00      0.98       252
  unstable_hand_entry       1.00      0.94      0.97        90
             accuracy                           0.98       432

#### RandomForest
                       precision    recall  f1-score   support
asymmetric_hand_entry       1.00      1.00      1.00        90
    smooth_hand_entry       1.00      1.00      1.00       252
  unstable_hand_entry       1.00      1.00      1.00        90
             accuracy                           1.00       432

#### MLP
                       precision    recall  f1-score   support
asymmetric_hand_entry       0.99      0.99      0.99        90
    smooth_hand_entry       1.00      1.00      1.00       252
  unstable_hand_entry       1.00      0.99      0.99        90
             accuracy                           1.00       432

```

**Label Distribution:**

* `stable_posture`: 460
* `unstable_posture`: 281

```text
#### SVM
                  precision    recall  f1-score   support
  stable_posture       0.98      1.00      0.99        87
unstable_posture       1.00      0.97      0.98        62
        accuracy                           0.99       149

#### RandomForest
                  precision    recall  f1-score   support
  stable_posture       1.00      1.00      1.00        87
unstable_posture       1.00      1.00      1.00        62
        accuracy                           1.00       149

#### MLP
                  precision    recall  f1-score   support
  stable_posture       0.98      1.00      0.99        87
unstable_posture       1.00      0.97      0.98        62
        accuracy                           0.99       149

```

---

## Code Structure

| Script / Notebook | Purpose |
| --- | --- |
| `video_scraper.py` | Scrape and download swimming videos from YouTube |
| `video_standardizer.py` | Standardize video resolution, FPS, and aspect ratio |
| `keypoint_extractor.py` | Extract pose keypoints, segment, and label motions |
| `model_training.ipynb` | Train and evaluate ML models for each distinct motion |
| `analyzer.py` | Overlay predictive feedback on video using trained models |

