import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta
from base_utils import T1DEXI_Utils
import warnings

warnings.filterwarnings("ignore")

class ExerciseTypeEmbedding(nn.Module):
    def __init__(self, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)  # 0: aerobic, 1: resistance, 2: interval

    def forward(self, indices):
        return self.embedding(indices)

class DataAggregator:
    """
    Aggregates multimodal time-series data (glucose, meal, exercise, demographics, and net insulin on board)
    for individuals using insulin delivery devices (AID or non-AID).

    This class extracts and synchronises asynchronous health-related events using timestamp alignment
    (within a ±5 minute window) for downstream use in forecasting models.

    Attributes:
        device_category (str): Device type, either 'aid' or 'non_aid'.
        utils (T1DEXI_Utils): Helper interface to retrieve and format T1DEXI dataset partitions.
    """
    def __init__(self, device_category):
        """
        Initialise the aggregator with a specified device category and load utility interface.

        Args:
            device_category (str): Category of insulin device ('aid' or 'non_aid').
        """
        self.utils = T1DEXI_Utils('../T1DEXI_dataset/')
        self.device_category = device_category
        self.exercise_embedding_model = ExerciseTypeEmbedding()

    def encode_device_category(self, category: str) -> int:
        """Binary encoder for device type: 1 = 'aid', 2 = 'non_aid'."""
        return 1 if category == 'aid' else 2

    def encode_day_night(self, timestamp: pd.Timestamp) -> int:
        """Classifies time as day (1) if 06:00 ≤ hour < 22:00, else night (0)."""
        return 1 if 6 <= timestamp.hour < 22 else 0

    def encode_gender(self, sex: str) -> int:
        """Encodes biological sex: 1 = Female ('F'), 0 = Male."""
        return 1 if sex == 'F' else 0

    def encode_meal_category(self, category: str) -> int:
        """
        Encodes meal category into integer codes for model use.

        Returns:
            int: Category ID from 1 to 8.
        """
        category_map = {
            'USUAL DAILY CONSUMPTION': 1,
            'RESCUE CARBS': 2,
            'Breakfast': 3,
            'Dinner': 4,
            'Lunch': 5,
            'Evening Snack': 6,
            'Afternoon Snack': 7,
        }
        return category_map.get(category, 8)  # Default: Morning Snack

    def encode_exercise_type(self, actarmcd: str) -> int:
        """
        Encodes exercise type based on study arm into:
            1 = AEROBIC, 2 = RESISTANCE, 3 = INTERVAL (default fallback).
        """
        arm_map = {'AEROBIC': 1, 'RESISTANCE': 2}
        return arm_map.get(actarmcd, 3)  # Default: INTERVAL

    def time_diff_agg(self, df: pd.DataFrame, time_col: str, target_time: pd.Timestamp) -> pd.Series:
        """
        Finds the closest row in df to target_time within a ±5-minute window.

        Args:
            df (pd.DataFrame): DataFrame to search.
            time_col (str): Name of the timestamp column.
            target_time (datetime): Timestamp to align to.

        Returns:
            pd.Series: Row with the closest match, or empty Series if none within tolerance.
        """
        df_temp = df.copy()
        df_temp['time_diff'] = (df_temp[time_col] - target_time).abs()
        nearest = df_temp[df_temp['time_diff'] <= timedelta(minutes=5)]
        if not nearest.empty:
            return nearest.loc[nearest['time_diff'].idxmin()].drop(['time_diff'])
        else:
            return pd.Series(dtype='object')  # empty series

    def safe_get(self, series: pd.Series, key: str):
        """
        Retrieves key from a Series or returns pd.NA if empty.

        Args:
            series (pd.Series): Target row.
            key (str): Column name.

        Returns:
            object: Value at key or pd.NA
        """
        # Safely retrieve a key from a Series or return pd.NA if the Series is empty.
        return series.get(key, pd.NA) if not series.empty else pd.NA       
        

    def data_pool(self) -> pd.DataFrame:
        """
        Aligns all data modalities (glucose, meals, exercise, insulin, demographics) per subject
        into a unified record per glucose reading.

        Returns:
            pd.DataFrame: Fully aligned and encoded dataset for the specified device category.
        """
        print(f'Pulling exercise data ...')
        exercise_data = self.utils.get_exercise_data(self.device_category)[['USUBJID', 'PRSTDTC', 'EXCINTSY', 'SNKBEFEX', 'PLNEXDUR', 'RESQCARB']]

        print(f'Pulling demographic data ...')
        dem_data = self.utils.get_demographics_data(self.device_category)[['USUBJID', 'SEX', 'ACTARMCD']]

        print(f'Pulling meal data ...')
        meal_data = self.utils.get_meal_data(self.device_category)[['USUBJID', 'MLDTC', 'MLDOSE', 'MLCAT']]

        print(f'Pulling glucose data ...')
        glucose_data = self.utils.get_glucose_data(self.device_category)[['USUBJID', 'LBDTC', 'LBORRES']]

        print(f'Pulling netIOB data ...\n')
        netiob_data = self.utils._get_data(f'NETIOB_{self.device_category}')[0]

        users = glucose_data['USUBJID'].unique()
        final_df = pd.DataFrame()

        for subjid in users:
            # Pre-filter and convert timestamps
            print(f'Processing user {subjid}\'s data')
            user_glucose = glucose_data[glucose_data['USUBJID'] == subjid].copy()
            user_glucose['LBDTC'] = pd.to_datetime(user_glucose['LBDTC'])
            user_glucose.sort_values('LBDTC', inplace=True)

            user_meal = meal_data[meal_data['USUBJID'] == subjid].copy()
            user_meal['MLDTC'] = pd.to_datetime(user_meal['MLDTC'])
            user_meal.sort_values('MLDTC', inplace=True)

            user_exercise = exercise_data[exercise_data['USUBJID'] == subjid].copy()
            user_exercise['PRSTDTC'] = pd.to_datetime(user_exercise['PRSTDTC'])
            user_exercise.sort_values('PRSTDTC', inplace=True)

            user_dem = dem_data[dem_data['USUBJID'] == subjid].iloc[0]
            
            user_netiob = netiob_data[netiob_data['USUBJID'] == subjid].copy()
            user_netiob['DATETIME'] = pd.to_datetime(user_netiob['DATETIME'])
            user_netiob.sort_values('DATETIME', inplace=True)

            gender_code = self.encode_gender(user_dem['SEX'])
            arm_code = self.encode_exercise_type(user_dem['ACTARMCD'])
            pump_code = self.encode_device_category(self.device_category)

            records = []

            for _, row in user_glucose.iterrows():
                
                glucose_time = row['LBDTC']
                # Get hour and day of the week features
                hour_feature = glucose_time.hour + glucose_time.minute / 60 + glucose_time.second / 3600
                day_feature = glucose_time.dayofweek  # Monday = 0, Sunday = 6
                
                record = {
                    'USUBJID': subjid,
                    'SEX': int(gender_code),
                    'ACTARMCD': int(arm_code),
                    'PUMP': int(pump_code),
                    'LBDTC': row['LBDTC'],
                    'DAY_NIGHT': int(self.encode_day_night(row['LBDTC'])),
                    'LBORRES': row['LBORRES'],
                    'SIN_H': np.sin(2 * np.pi * hour_feature / 24),
                    'COS_H': np.cos(2 * np.pi * hour_feature / 24),
                    'SIN_DOW': np.sin(2 * np.pi * day_feature / 7),
                    'COS_DOW': np.cos(2 * np.pi * day_feature / 7)
                }

                # NETIOB match
                netiob_match = self.time_diff_agg(user_netiob, 'DATETIME', glucose_time)
                record['NETIOB'] = self.safe_get(netiob_match, 'NETIOB')

                # MEAL match
                meal_match = self.time_diff_agg(user_meal, 'MLDTC', glucose_time)
                record['MLDOSE'] = self.safe_get(meal_match, 'MLDOSE')
                raw_mlcat = self.safe_get(meal_match, 'MLCAT')
                record['MLCAT'] = int(self.encode_meal_category(raw_mlcat)) if pd.notna(raw_mlcat) else pd.NA

                # EXERCISE match
                exercise_match = self.time_diff_agg(user_exercise, 'PRSTDTC', glucose_time)
                for k in ['EXCINTSY', 'SNKBEFEX', 'PLNEXDUR', 'RESQCARB']:
                    value = self.safe_get(exercise_match, k)
                    record[k] = int(value) if pd.notna(value) else 0

                records.append(record)

            # Build DataFrame per subject and cast to nullable Int
            user_df = pd.DataFrame(records)            
            final_df = pd.concat([final_df, user_df], ignore_index=True)

        return final_df    
    
def compute_z_scores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Ensure datetime index and sort
    df['LBDTC'] = pd.to_datetime(df['LBDTC'])
    df.sort_values(['USUBJID', 'LBDTC'], inplace=True)
    df.set_index('LBDTC', inplace=True)
    df.sort_index(inplace=True)  # Ensure index is monotonic before rolling
    
    # Apply rolling window over 24 hours
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # ensures float64
        roll = df[col].rolling('24H', min_periods=12)
        
        # Compute z-score
        df[f'{col}_ZSCORE_24'] = (df[col] - roll.mean()) / (roll.std() + 1e-6)
    
    df.reset_index(inplace=True)
    
    return df

if __name__ == '__main__':
    
    combined_results = []
    for pump in ['aid', 'non_aid']:
        aggregator = DataAggregator(pump)
        df = aggregator.data_pool()
        combined_results.append(df)

    # Concatenate all results into a single dataframe
    unified_df = pd.concat(combined_results, ignore_index=True)
    
    # Compute Glucose z-score
    unified_df = compute_z_scores(unified_df, ['LBORRES', 'NETIOB'])

    print('finishing up...')
    unified_df.to_csv('glucose_forecast_data.csv', index=False)
    T1DEXI_Utils().dataframe_to_xport(unified_df, '../T1DEXI_dataset/glucose_forecast_data.xpt')
    
