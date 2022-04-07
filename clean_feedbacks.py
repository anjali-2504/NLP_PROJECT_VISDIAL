import numpy as np
import pandas as pd
import os
import sys

if __name__=='__main__':
    folder_name =""
    feedbacks = pd.read_csv(os.path.join(folder_name,'15Ep_test_feedbacks.csv'))
    comments = pd.read_csv(os.path.join(folder_name,'15Ep_test_comments.csv'))
    results = pd.read_csv(os.path.join(folder_name,'15Ep_test_results.csv'))

    bools = np.logical_not(feedbacks['feedback'].isna().to_numpy())

    feedbacks_ = feedbacks[bools]
    comments_ = comments[bools]
    results_ = results[bools]

    feedbacks_.to_csv(os.path.join(folder_name,'15Ep_test_feedbacks.csv'),index=False)
    comments_.to_csv(os.path.join(folder_name,'15Ep_test_comments.csv'),index=False)
    results_.to_csv(os.path.join(folder_name,'15Ep_test_results.csv'),index=False)
    
