# ROS Bag to AVI Video Conversion Guide

**Purpose:** Convert ROS2 bag recordings to .avi video files for offline analysis and test set creation

**Source:** Based on `ubm-fsae` repository instructions (https://github.com/ubm-driverless/ubm-fsae)

**Last Updated:** 2026-01-26

---

## 🎯 Overview

This procedure converts image topics from a ROS2 bag recording into .avi video files on the ASU (Autonomous System Unit).

**Use Cases:**
1. **Offline video analysis** - Review camera footage from car runs
2. **Test set creation** - Extract frames for YOLO annotation
3. **Debugging** - Inspect specific scenarios or edge cases
4. **Documentation** - Create videos for presentations/reports

---

## 📋 Prerequisites

**Hardware:**
- ASU (Ubuntu 24.04, ROS2 installed)
- Sufficient storage for .avi output (~1-2 GB per minute)

**Software:**
- ROS2 workspace with `ubm-fsae` repository
- ZED 2i stereo camera drivers
- Image-to-video conversion node

**Input:**
- ROS2 bag file (`.mcap` or `.db3` format)
- Contains image topics: `/zed2i/left/image_rect_color` and `/zed2i/right/image_rect_color`

---

## 🚀 Step-by-Step Procedure

### Step 1: Configure Launch File

**File:** `ubm-fsae/launch/bringup.py`

**Edit the following parameters:**

```python
# Enable ZED camera node
zed_node = True  # Set to True

# Enable AVI conversion node
images_to_avi_node = True  # Set to True

# Configure output settings
output_video_name = 'test_run_2026_01_26'  # Your video name
output_video_path = '/home/ubm/Videos/'     # Save location
```

**Important Settings:**
- `video_fps`: 30 (default, matches ZED 2i recording rate)
- `video_codec`: 'XVID' (good compression/quality balance)
- `image_topic`: Configure for left/right camera separately

### Step 2: Launch the Bringup File

**Open terminal on ASU:**

```bash
cd ~/ubm-fsae

# Source ROS2 workspace
source install/setup.bash

# Launch bringup (starts ZED node + AVI converter)
ros2 launch launch/bringup.py
```

**Expected Output:**
```
[INFO] [launch]: Starting ZED camera node...
[INFO] [launch]: Starting images_to_avi_node...
[INFO] [images_to_avi]: Waiting for image topics...
```

### Step 3: Play Back ROS Bag

**Open a new terminal:**

```bash
cd ~/rosbags  # Or wherever your bags are stored

# List available bags
ls -lh *.mcap

# Play back the desired bag
ros2 bag play <name_of_rosbag>.mcap

# Example:
ros2 bag play test_run_2026_01_22_lidar1.mcap
```

**Playback Options:**
```bash
# Play at half speed (for slow motion)
ros2 bag play <bag> --rate 0.5

# Play at double speed (faster processing)
ros2 bag play <bag> --rate 2.0

# Loop playback (for testing)
ros2 bag play <bag> --loop
```

### Step 4: Monitor Conversion Progress

**In the launch terminal, you should see:**

```
[INFO] [images_to_avi]: Recording frame 100/1000
[INFO] [images_to_avi]: Recording frame 200/1000
...
[INFO] [images_to_avi]: Video conversion complete
```

**Check output:**

```bash
ls -lh ~/Videos/
# Should show: test_run_2026_01_26.avi
```

### Step 5: Stop and Finalize

**Once bag playback completes:**

1. Wait 5-10 seconds for video writer to finalize
2. Press `Ctrl+C` in launch terminal
3. Verify video file exists and is not corrupted

**Verify video:**

```bash
# Check file size (should be > 0 bytes)
ls -lh ~/Videos/test_run_2026_01_26.avi

# Play video to verify
vlc ~/Videos/test_run_2026_01_26.avi
# Or: ffplay ~/Videos/test_run_2026_01_26.avi
```

---

## 🎥 Output Format

**Video Specifications:**
- **Format:** AVI (Audio Video Interleave)
- **Codec:** XVID (H.264 fallback)
- **Resolution:** 1280×720 per camera (2560×720 if stereo stitched)
- **FPS:** 60 ⚠️ **NOTE:** Exported at 60 FPS even though recorded at 30 FPS (frame duplication)
- **Color:** RGB (from `image_rect_color` topic)

**File Naming Convention:**
```
<run_date>_<location>_<lidar_num>.avi

Examples:
- test_run_2026_01_26_lidar1.avi
- test_run_2026_01_26_lidar2.avi
- competition_run_FSG_2026.avi
```

---

## 📊 Actual Results (2026-01-26 Test)

**Successfully created:**
- `media/20_11_2025_Rioveggio_Test_LidarTest1.avi`
  - Size: 203 MB
  - Resolution: 2560×720 (stereo stitched)
  - **FPS: 60** (exported at 60 FPS, recorded at 30 FPS)
  - Frames: 1,454
  - Duration: 24.2 seconds

- `media/20_11_2025_Rioveggio_Test_LidarTest2.avi`
  - Size: 194 MB
  - Resolution: 2560×720 (stereo stitched)
  - **FPS: 60** (exported at 60 FPS, recorded at 30 FPS)
  - Frames: 1,374
  - Duration: 22.9 seconds

**Image Format:**
- Stereo stitched: Left and right images **side-by-side** (2560×720 total)
- Split required: Each frame must be split down the middle for YOLO processing
  - Left image: 0:1280 pixels
  - Right image: 1280:2560 pixels

**⚠️ Important Note: 60 FPS Export**
- Videos exported at **60 FPS** even though recorded at 30 FPS
- This means frames are duplicated (each original frame appears twice)
- For test set creation, extract every **60 frames** to get 1 unique frame per second
- This represents **2 seconds of real-world time** (since original was 30 FPS)

---

## ⚠️ Important Notes

### Real-Time Recording (Not Recommended)

You **can** use this method to record videos in real-time while the car is running:

```bash
# Launch bringup with live camera
ros2 launch launch/bringup.py

# AVI conversion happens in real-time
```

**⚠️ WARNING:** This will **slow down CPU and SSD performance** and is **NOT recommended** for real-time applications (like autonomous driving).

**Why:**
- Video encoding is CPU-intensive
- Disk writes compete with other critical processes
- May cause frame drops or latency spikes

**Recommendation:** Record ROS bags during runs, convert to AVI offline.

### Storage Requirements

**Rough estimates:**
- ROS bag: ~1 GB per minute (compressed)
- AVI video: ~1-2 GB per minute (depends on codec)
- Requires 2-3× bag size for conversion workspace

**Example:** 10-minute test run
- Bag: ~10 GB
- AVI: ~15 GB
- Total: ~25 GB needed

---

## 🐛 Troubleshooting

### Problem: "No image topics found"

**Solution:**
```bash
# Check available topics in bag
ros2 bag info <bag_name>.mcap

# Verify image topics exist:
# - /zed2i/left/image_rect_color
# - /zed2i/right/image_rect_color
```

### Problem: "Video file is 0 bytes"

**Causes:**
1. Bag playback finished before video writer initialized
2. Incorrect image topic name
3. Insufficient disk space

**Solution:**
- Wait for "Recording frame X" messages before starting playback
- Check topic names in launch file
- Free up disk space: `df -h`

### Problem: "Video is corrupted or won't play"

**Solution:**
```bash
# Re-encode with ffmpeg
ffmpeg -i corrupted.avi -c:v libx264 -crf 20 fixed.avi

# Or convert to MP4 (more compatible)
ffmpeg -i video.avi -c:v libx264 -c:a copy video.mp4
```

### Problem: "Conversion is very slow"

**Causes:**
- CPU overload (other processes running)
- Slow disk I/O (writing to network drive)
- High resolution/framerate

**Solution:**
- Close unnecessary applications
- Write to local SSD (not network drive)
- Reduce playback rate: `ros2 bag play <bag> --rate 0.5`

---

## 🔄 Next Steps: Test Set Creation

**After AVI conversion, create YOLO test set:**

1. **Extract frames** from .avi videos (Python/FFmpeg)
2. **Split stereo images** (left/right separation)
3. **Upload to Roboflow** for annotation
4. **Label cones** (5 classes: blue, yellow, orange, large_orange, unknown)
5. **Download YOLO format** dataset
6. **Evaluate models** on UBM test set

**See:** `docs/TODO.md` for detailed test set creation workflow

---

## 📁 File Locations (ASU)

```
~/ubm-fsae/
├── launch/
│   └── bringup.py          # Main launch file (edit here)
├── src/
│   └── images_to_avi_node/ # AVI conversion node source
└── rosbags/                # Input: ROS bag recordings

~/Videos/                    # Output: AVI video files
├── lidar1.avi
└── lidar2.avi
```

---

## 📚 References

- **UBM FSAE Repository:** https://github.com/ubm-driverless/ubm-fsae
- **ROS2 Bag Documentation:** https://docs.ros.org/en/rolling/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html
- **ZED 2i Camera:** https://www.stereolabs.com/docs/ros2/zed-node/

---

**Last Updated:** 2026-01-26
**Tested On:** ASU - Ubuntu 24.04 LTS, ROS2 Jazzy, ZED SDK 5.0.4
