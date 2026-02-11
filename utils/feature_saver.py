import os
import numpy as np

class FeatureSaver:
    def __init__(self, save_dir, task, model_name, seed):
        """
        Initialize the feature saver.
        """
        self.feature_save_dir = f"./features/{task}/{model_name}/seed{seed}_features"
        os.makedirs(self.feature_save_dir, exist_ok=True)
        
        # Dictionary to store features for each data split
        self.features = {
            'train': {'ehr': [], 'cxr': [], 'labels': []},
            'val': {'ehr': [], 'cxr': [], 'labels': []},
            'test': {'ehr': [], 'cxr': [], 'labels': []}
        }
    
    def add_features(self, split, ehr_feat, cxr_feat, labels):
        """
        Add features to the internal storage for a given data split.
        """
        self.features[split]['ehr'].append(ehr_feat)
        self.features[split]['cxr'].append(cxr_feat)
        self.features[split]['labels'].append(labels)
    
    def save_features(self, split, epoch, hidden_size=None):
        """
        Save accumulated features for the given split to disk as a .npz file.
        """
        features_dict = self.features[split]
        if len(features_dict['ehr']) > 0:
            ehr_features = np.vstack(features_dict['ehr'])
            cxr_features = np.vstack(features_dict['cxr'])
            labels = np.vstack(features_dict['labels'])
            
            save_path = os.path.join(
                self.feature_save_dir, 
                f"{split}_features_epoch_{epoch}.npz"
            )
            
            np.savez(
                save_path,
                ehr_features=ehr_features,
                cxr_features=cxr_features,
                labels=labels,
                hidden_size=hidden_size,
                epoch=epoch
            )
            
            # Clear the feature lists after saving
            for key in features_dict:
                features_dict[key] = []
                
            if split == 'train':
                print(f"Save the features in epoch {epoch}")
    
    def clear_features(self, split):
        """
        Clear all stored features for a specific data split.
        """
        for key in self.features[split]:
            self.features[split][key] = [] 