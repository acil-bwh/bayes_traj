# base_model.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    
    @abstractmethod
    def augment_df_with_traj_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to augment the dataframe with trajectory information.
        
        Parameters
        ----------
        df : pandas DataFrame
            DataFrame to be augmented.
            
        Returns
        -------
        df_aug : pandas DataFrame
            Augmented DataFrame. This DataFrame will be the same as the input, 
            but it will have additional columns: 'traj', 'traj_*', where 'traj'
            is an integer identifier indicating the most probable trajectory
            assignment, and there will be multiple 'traj_*' columns for each
            trajectory indicating the probability of assignment to that 
            trajectory.
        """
        pass
