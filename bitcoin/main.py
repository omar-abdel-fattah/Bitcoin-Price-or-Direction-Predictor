import pandas as pd
filepath = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"

import ssl  # we need to import this library and tweak one setting due to fact we use HTTPS certificate(s)
ssl._create_default_https_context = ssl._create_unverified_context

                       # Now we want to create a dataframe and use Pandas' to_csv function to read in our file
pd.set_option("display.max_columns",None)
df = pd.read_csv(filepath, skiprows=1)  # we use skiprows parameter because first row contains our web address

                     # Now that we have loaded our data into the dataframe, we can preview it using the print & .head() function
print(df.head(800))  # print first 15 lines of dataframe
df.to_csv(r'E:\bitcoin\data_1.csv',index=False,header=True)
