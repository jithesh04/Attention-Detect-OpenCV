from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2
def Face_detection(image):
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        result1 = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        result2 = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

        return list(result1["labels"]).count(1),list(result2["labels"]).count(1)



