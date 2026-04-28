from torch.utils.data import Dataset
import os
from PIL import Image
import cv2


class Fire_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_video=10, include_test_videos=False):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.include_test_videos = include_test_videos
        self.samples = []
        self.error_logs = []
        self._load_data()
    
    def _load_data(self):
        image_exts = ["jpg", "jpeg", "png"]
        video_exts = ["mp4", "avi", "mov"]
        
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                ext = f.split(".")[-1].lower()
                full_path = os.path.join(root, f)
                
                try:
                    # Determine label based on folder structure
                    label = self._get_label(root, f)
                    
                    # Skip unlabeled test videos unless explicitly included
                    if label is None:
                        if self.include_test_videos:
                            self.samples.append((full_path, -1, "video"))
                        continue
                    
                    # for images
                    if ext in image_exts:
                        self.samples.append((full_path, label, "image"))
                    
                    # for videos: store frame references (lazy loading)
                    elif ext in video_exts:
                        frame_count = self._get_frame_count(full_path)
                        if frame_count > 0:
                            frame_indices = [int(i * frame_count / self.frames_per_video) 
                                           for i in range(self.frames_per_video)]
                            for frame_idx in frame_indices:
                                self.samples.append((full_path, label, "frame", frame_idx))
                
                except Exception as e:
                    self.error_logs.append((full_path, str(e)))
    
    def _get_label(self, root, filename):
        """
        Determine label based on directory structure and filename
        Returns: 0 (No Fire), 1 (Fire), or None (unlabeled)
        """
        root_lower = root.lower()
        filename_lower = filename.lower()
        
        # For image data
        if "img_data" in root_lower:
            if "default" in root_lower:
                return 0  # No Fire
            elif "fire" in root_lower or "smoke" in root_lower:
                return 1  # Fire
        
        # For video data
        elif "video_data" in root_lower:
            # test_videos are unlabeled
            if "test_videos" in root_lower:
                return None  # Unlabeled
            
            # train_videos - check filename
            # NOTE: "nofire" must be checked BEFORE "fire" because
            # "fire" is a substring of "nofire".
            elif "train_videos" in root_lower:
                if "nofire" in filename_lower:
                    return 0  # No Fire
                elif "fire" in filename_lower or "smoke" in filename_lower:
                    return 1  # Fire
        
        return None  # Unknown/unlabeled
    
    def _get_frame_count(self, video_path):
        """Get total frame count without loading frames into memory"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    
    def _load_frame(self, video_path, frame_idx):
        """Load a single frame from a video on demand"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if sample[2] == "image":
            data, label, _ = sample
            img = Image.open(data).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        
        elif sample[2] == "frame":
            video_path, label, _, frame_idx = sample
            img = self._load_frame(video_path, frame_idx)
            if img is None:
                # Fallback: return a black image if frame load fails
                img = Image.new("RGB", (224, 224))
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        
        elif sample[2] == "video":
            data, label, _ = sample
            return data, label


from torch.utils.data import Subset

class Subset_Dataset(Dataset):
    def __init__(self, subset: Subset, main_transform, fire_train_transform, no_fire_train_transform, is_train=False):
        self.subset = subset
        self.main_transform = main_transform
        self.fire_train_transform = fire_train_transform
        self.no_fire_train_transform = no_fire_train_transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.is_train:
            if label == 0:
                image = self.no_fire_train_transform(image)
            else:
                image = self.fire_train_transform(image)
        else:
            image = self.main_transform(image)
        return image, label