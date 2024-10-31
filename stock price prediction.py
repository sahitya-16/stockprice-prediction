#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[2]:


pip install keras


# In[3]:


pip install tensorflow


# In[4]:


pip install optree


# In[5]:


import yfinance as yf


# In[6]:


from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)


# In[7]:


stock = "GOOG"
google_data = yf.download(stock, start, end)


# In[8]:


google_data.head()


# In[9]:


google_data.describe()


# In[10]:


google_data.info()


# In[11]:


google_data.isna().sum()


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


plt.figure(figsize = (15,5))
google_data['Adj Close'].plot()
plt.xlabel("years")
plt.ylabel("Adj Close")
plt.title("Closing price of Google data")


# In[14]:


def plot_graph(figsize, values, column_name):
    plt.figure()
    values.plot(figsize = figsize)
    plt.xlabel("years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of Google data")


# In[15]:


google_data.columns


# In[16]:


for column in google_data.columns:
    plot_graph((15,5),google_data[column], column)


# In[17]:


temp_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(sum(temp_data[1:6])/5)


# In[18]:


import pandas as pd
data = pd.DataFrame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
data.head()


# In[19]:


data['MA'] = data.rolling(5).mean()
data


# In[20]:


for i in range(2004,2025):
    print(i,list(google_data.index.year).count(i))


# In[21]:


google_data['MA_for_250_days'] = google_data['Adj Close'].rolling(250).mean()


# In[22]:


google_data['MA_for_250_days'][0:250].tail()


# In[23]:


plot_graph((15,5), google_data[['Adj Close','MA_for_250_days']], 'MA_for_250_days')


# In[24]:


google_data['MA_for_100_days'] = google_data['Adj Close'].rolling(100).mean()
plot_graph((15,5), google_data[['Adj Close','MA_for_100_days']], 'MA_for_100_days')


# In[25]:


Adj_close_price = google_data[['Adj Close']]


# In[26]:


max(Adj_close_price.values),min(Adj_close_price.values) 


# In[27]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Adj_close_price)
scaled_data


# In[28]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Adj_close_price)
scaled_data


# In[29]:


len(scaled_data)


# In[30]:


x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
    
import numpy as np
x_data, y_data = np.array(x_data), np.array(y_data)


# In[31]:


x_data[0],y_data[0]


# In[32]:


int(len(x_data)*0.7)


# In[33]:


4908-100-int(len(x_data)*0.7)


# In[34]:


splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]


# In[35]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[36]:


from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[37]:


pip install optree


# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[42]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[43]:


model.fit(x_train, y_train, batch_size=1, epochs = 2)


# In[44]:


model.summary()


# In[45]:


predictions = model.predict(x_test)


# In[46]:


predictions


# In[47]:


inv_predictions = scaler.inverse_transform(predictions)
inv_predictions


# In[48]:


inv_y_test = scaler.inverse_transform(y_test)
inv_y_test


# In[49]:


rmse = np.sqrt(np.mean( (inv_predictions - inv_y_test)**2))


# In[50]:


rmse


# In[51]:


ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
ploting_data.head()


# In[52]:


plot_graph((15,6), ploting_data, 'test data')


# In[53]:


plot_graph((15,6), pd.concat([Adj_close_price[:splitting_len+100],ploting_data], axis=0), 'whole data')


# In[54]:


model.save("Latest_stock_price_model.keras")


# In[ ]:




