# Real-Time Crowd Anomaly Detection (CLIP + YOLOv8 Fusion)

This repository contains a seminar project for **crowd anomaly detection in surveillance videos** using a lightweight **feature-fusion pipeline**:

- **CLIP visual embeddings** for semantic frame understanding  
- **YOLOv8 detections** summarized into compact object-level statistics  
- **Fusion** (concatenation) into a single video-level representation for classification  

The main implementation is provided as a Jupyter notebook:
- `ProjectRealTimeCrowdAnomaly.ipynb`

---

## What This Project Does

Given a video represented as a folder of frames, the pipeline:

1. **Samples frames** from each video (commonly a fixed number, e.g., 32) using a uniform strategy.  
2. **Extracts CLIP features** per frame (image embeddings).  
3. **Aggregates frame embeddings** into a single **video vector** using pooling (e.g., mean and/or max across frames).  
4. **Runs YOLOv8** on the same sampled frames and converts detections into a **small stats vector** (e.g., counts / confidence-based summaries per class).  
5. **Fuses** CLIP video vector + YOLO stats vector into one feature vector.  
6. Trains and evaluates ML classifiers for:
   - **Binary classification**: Normal vs Anomaly  
   - (Optional) **Multi-class classification**: anomaly type (if labels exist)  

---

## Repository Structure

- `ProjectRealTimeCrowdAnomaly.ipynb` — end-to-end notebook (data → features → training → evaluation)  
- `README.md` — project documentation  
- `.gitignore` — ignores local artifacts (datasets, caches, etc.)  

---

## Setup

### 1) Create an environment (recommended)

**Conda**
```bash
conda create -n crowd-anomaly python=3.10 -y
conda activate crowd-anomaly
```

## Dataset Format
```txt
Dataset/
  Train/
    NormalVideos/
      video_001/
        frame_0001.jpg
        frame_0002.jpg
        ...
    Robbery/
      video_105/
        frame_0001.jpg
        ...
  Test/
    NormalVideos/
    Robbery/
    ...
```
