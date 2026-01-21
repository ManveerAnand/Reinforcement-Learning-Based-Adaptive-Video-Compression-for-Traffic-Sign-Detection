"""
Phase 1 - Test Pre-trained YOLOv8 on CURE-TSD Frame
Goal: Understand YOLO detection format and verify it works on traffic signs
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO
from phase1.video_loader import VideoLoader
import sys
sys.path.insert(0, 'src')


print("=" * 70)
print("PHASE 1 - YOLOv8 PRE-TRAINED TEST")
print("=" * 70)

# Load sample frame
print("\n📹 Step 1: Loading sample frame...")
video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
loader = VideoLoader(video_path)

# Get frame where sign is visible (frame 100)
frame = loader.get_frame(100)
print(f"✓ Loaded frame 100")
print(f"  Shape: {frame.shape}")
print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")

# Load pre-trained YOLOv8 (COCO dataset - has traffic signs)
print("\n🤖 Step 2: Loading pre-trained YOLOv8n model...")
model = YOLO('yolov8n.pt')  # Nano model (fastest)
print("✓ Model loaded")

# Run detection
print("\n🔍 Step 3: Running detection on frame...")
results = model.predict(frame, verbose=False)
print(f"✓ Detection complete")

# Analyze results
print("\n📊 Step 4: Analyzing detections...")
print("-" * 70)

result = results[0]  # First result (we only have 1 image)

# Get detected objects
boxes = result.boxes
num_detections = len(boxes)

print(f"Total detections: {num_detections}")

if num_detections > 0:
    print(f"\n{'ID':<4} {'Class Name':<20} {'Confidence':<12} {'BBox (x1,y1,x2,y2)'}")
    print("-" * 70)

    for i, box in enumerate(boxes):
        # Extract info
        cls_id = int(box.cls[0])  # Class ID
        conf = float(box.conf[0])  # Confidence
        xyxy = box.xyxy[0].cpu().numpy()  # Bounding box [x1, y1, x2, y2]

        # Get class name from COCO dataset
        class_name = model.names[cls_id]

        print(f"{i:<4} {class_name:<20} {conf:<12.4f} "
              f"({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")

else:
    print("⚠️ No detections found!")
    print("   This is expected - pre-trained YOLO is trained on COCO dataset")
    print("   which has limited traffic sign classes.")

# Check what COCO classes are available
print("\n📋 Step 5: Checking COCO classes relevant to traffic...")
print("-" * 70)
print("COCO dataset classes (80 total):")
traffic_related = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                   'traffic light', 'stop sign']

print("\nTraffic-related classes in COCO:")
for cls_name in traffic_related:
    if cls_name in model.names.values():
        cls_id = [k for k, v in model.names.items() if v == cls_name][0]
        print(f"  - {cls_name:<20} (ID: {cls_id})")

print("\n⚠️ NOTE: COCO only has 'stop sign' and 'traffic light'")
print("   CURE-TSD has 14 sign types (Speed Limit 30/50/60/70/80, Yield, etc.)")
print("   → We MUST fine-tune YOLOv8 on CURE-TSD (Phase 5)")

# Save annotated image
print("\n💾 Step 6: Saving annotated image...")
annotated_frame = result.plot()  # Draw boxes on image

output_path = "outputs/yolo_test/frame100_pretrained_yolo.jpg"
os.makedirs("outputs/yolo_test", exist_ok=True)
cv2.imwrite(output_path, annotated_frame)
print(f"✓ Saved to: {output_path}")

# Detection format summary
print("\n" + "=" * 70)
print("✅ PHASE 1 - YOLOv8 TEST COMPLETE")
print("=" * 70)
print("\n📌 Key Findings:")
print("1. YOLOv8 detection format:")
print("   - boxes.cls[i]: Class ID (int)")
print("   - boxes.conf[i]: Confidence score (float 0-1)")
print("   - boxes.xyxy[i]: Bounding box [x1, y1, x2, y2] (pixels)")
print("\n2. Pre-trained model limitations:")
print("   - COCO has only 'stop sign' class")
print("   - CURE-TSD needs 14 sign classes")
print("   - Must fine-tune in Phase 5")
print("\n3. Next steps:")
print("   - Convert CURE-TSD labels to YOLO format")
print("   - Fine-tune YOLOv8n on all 14 sign classes")
print("   - Integrate with feature_extractor.py")
print("\n🎯 Ready for Phase 3: Compression Simulator!")
