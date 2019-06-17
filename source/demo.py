from data.data_stuff import testing_denoise
from data.plot_data import feature_extraction, plot_raw

def demo():
    print('Raw data')
    plot_raw()
    print('Feature Extraction')
    feature_extraction()
    print('Denoise')
    testing_denoise()

demo()
