import gdown

data_url = 'https://drive.google.com/drive/folders/1qSCrp_2zSjxFOk1PNxrLW29v_Z_R4mic?usp=share_link'
model_url = 'https://drive.google.com/drive/folders/1dIfNX_TY3WMmjnILrPCOkRF5ZtLyCalB?usp=share_link'

gdown.download_folder(data_url, quiet=True, use_cookies=False)
gdown.download_folder(model_url, quiet=True, use_cookies=False)
