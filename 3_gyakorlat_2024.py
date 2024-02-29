#!/usr/bin/env python
# coding: utf-8

# # Big Data algoritmusok és programozás - Python
# 
# # 3. gyakorlat
# 
# # Pandas
# 
# 
# _____
# A pandas prorgamkönyvtár az adatfeldolgozást és adatelemzést támogatja. 
# 
# Adatszerkezetet szolgáltat adattáblák és idősorok kezeléséhez. 
# 
# **Dokumentáció: [Pandas Documentation](https://pandas.pydata.org/docs/index.html#)**
# 
# _____

# ## 0. Installálás és importálás
# 
# **Anaconda környezetben, Pandas installálása terminál ablakban:**
#     
#     conda install pandas
#     
# vagy nem Anaconda környezetben:
# 
#     pip install pandas    
#     
# Ha nincs Anaconda környezet vagy nem sikerül a telepítés: [Pandas hivatalos dokumentációja a telepítésről.](https://pandas.pydata.org/getting_started.html)

# In[2]:


import numpy as np
import pandas as pd


# _____
# ## 1. Adattípusok
# 
# - `Series` - 1-dimenziós adat (oszlop)
# - `DataFrame` - 2-dimenziós adattábla sorokkal és oszlopokkal 

# A Series egy egydimenziós címkézett tömb, amely bármilyen adattípus tárolására képes.
# 
# A Series adattípus nagyon hasonló a NumPy tömbökhöz (array). 
# 
# Különbség: a Series címkékkel indexelt a tömböknél helyet jelölő számok helyett. 

# ### Series adatok készítése
# 
# Listák, numpy tömbök és dictionaryk konvertálása Series adatokká:

# In[3]:


cars = ['Audi','Toyota','Honda']
colours = ['blue','red','white']
numbers = np.array([10,20,30])
dic = {'x':10,'y':20,'z':30}


# In[4]:


pd.Series


# **Lista -> Series**

# In[5]:


pd.Series(data=cars)


# In[6]:


pd.Series(data=colours,index=cars)


# In[7]:


pd.Series(colours,cars)


# **NumPy Arrays -> Series**

# In[9]:


pd.Series(cars)


# In[10]:


pd.Series(numbers,cars)


# **Dictionary -> Series**

# In[11]:


pd.Series(dic)


# ### A Series adatok
# 
# A pandas Series különböző típusú objektumokat tartalmazhat:

# In[12]:


pd.Series(data=numbers)


# In[13]:


# Függvényeket is tartalmazhat (de nem valószínű, hogy ilyeneket használunk)
pd.Series([sum,max,print])


# In[14]:


pd.Series(data=[sum,'b',1])


# _____
# ## Indexek használata
# 
# Pandas az index neveket vagy számokat az információ gyors eléréséhez használja.
# 
# Példák információszerzésre index használatával:

# In[15]:


cars1 = pd.Series([10,20,15,10],index = ['BMW', 'Toyota','Honda', 'Audi'])


# In[16]:


cars1


# In[17]:


cars2 = pd.Series([5,30,15,5],index = ['Mercedes', 'Opel','Toyota','BMW'])


# In[18]:


cars2


# In[19]:


cars1['Toyota']


# Műveletek végzése indexek alapján:

# In[20]:


cars1 + cars2


# _____
# ## DataFrames
# 
# Többdimenziós adatszerkezet. 
# 
# Tekinthetjük úgy, mint Series-ek együttese, amelyek azonos indexet használnak. 
# 

# In[21]:


from numpy.random import randn


# In[22]:


np.random.seed(101) # seed: ugyanolyan véletlen számokat kapunk vele 


# In[23]:


df = pd.DataFrame(randn(4,5),index='A B C D'.split(),columns='E F G H I'.split())


# In[24]:


df


# In[25]:


#Ha nincs megadva index, akkor megszámozza a sorokat és az oszlopokat.
df2 = pd.DataFrame(randn(4,5))


# In[26]:


df2


# In[27]:


car_data = pd.DataFrame({"Type": cars, "Colour": colours})


# In[28]:


car_data


# **Feladat: DataFrame készítése**
# 
# A félévben felvett kurzusokat, azok heti óraszámát és a krediteket tartalmazza.

# In[42]:


dataframe = pd.DataFrame({"Tárgynév": ["ERP", "Big DAta"], "Óraszám":[2,3], "Kredit":[4,6]})
dataframe


# In[47]:


cars = ['ERP I','Big Data']
számok = [0,1]
#dataframe = pd.DataFrame( data=cars ,index= számok,columns=['Tárgy neve', 'Kredit'] )
#dataframe


# ---
# ## 2. Adatok importálása
# 
# `pd.read_csv()`, `pd.read_excel()`
# 
# `df = pd.read_csv('file_név')` - `df` név használata gyakori

# In[39]:


car_sales = pd.read_csv('car_sales.csv')
car_sales


# A DataFrame szerkezete:
# ![image.png](attachment:image.png)

# _____
# ## 3. Adatok leírása
# 
# - Típus: `.dtypes`
# - Információ a DataFrame-ről: `.info()`
# - Statisztika: `.describe()`
# - Oszlopok: `.columns`
# - Sorok megjelenítése: `.head()`, `.tail()`
# - DataFrame hossza (sorok száma): `len(df)`

# In[48]:


car_sales.dtypes


# In[49]:


car_sales.info()


# ---
# **Egy kis kitérő: memóriahasználat optimalizása**

# In[50]:


col_types = {'Odometer (KM)': 'int32','Doors':'int8','Price ($)':'int32'}
#'Make':'category','Colour':'category'


# In[51]:


car_sales = pd.read_csv('car_sales.csv', dtype=col_types)


# In[52]:


car_sales.info()


# ---

# In[53]:


car_sales.describe()


# In[54]:


car_sales.describe().T


# In[55]:


car_sales.columns


# In[56]:


car_sales.head(5)


# In[57]:


car_sales.tail(3)


# In[58]:


len(car_sales)


# _____
# ## 4. Adatok elérése
# 
# Különböző módszerek az adatok eléréséhez.

# In[59]:


#Oszlop elérése - oszlop nevének megadása (szögletes zárójelben a DataFrame neve mögé írva)
car_sales['Make']


# In[60]:


#Több oszlop elérése - oszlopok nevének megadása listaként (dupla szögletes zárójelben!)
car_sales[['Make','Price ($)']]


# Egy DataFrame oszlop olyan, mint a Series

# In[61]:


type(car_sales['Make'])


# In[62]:


type(car_sales[['Make','Price ($)']])


# **Új oszlop készítése:**

# In[63]:


car_sales['Reduced Price'] = car_sales['Price ($)'] * 0.9


# In[64]:


car_sales


# **Oszlop törlése**

# In[65]:


car_sales.drop('Reduced Price',axis=1)


# In[66]:


# Nem változtatja meg a DataFrame-et (ha nem adjuk meg külön)!
car_sales


# In[67]:


car_sales.drop('Reduced Price',axis=1,inplace=True)


# In[68]:


car_sales


# **Sor törlése**

# In[69]:


#Alapértelmezés: axis=0, sor törlése
car_sales.drop(9)


# In[70]:


car_sales.drop(9,axis=0)


# **Sorok kiválasztása**

# In[71]:


car_sales.loc[[1,2]]


# Sorok kiválasztása index alapján (címke helyett)

# In[72]:


car_sales.iloc[[1,2]]


# In[73]:


df


# In[74]:


df.loc['B']


# In[75]:


df.iloc[1] # ha index alapján választok akkor iloc


# In[96]:


#3. és 4. sor kiválasztása a df DataFrame-ből index alapján
df.iloc[2::]
df.iloc [[2,3]]


# **Sorok vagy oszlopok részhalmazának kiválasztása**

# In[77]:


car_sales.loc[1,'Make']


# In[78]:


car_sales.loc[[3,5],['Make','Price ($)']]


# In[79]:


car_sales.loc[3:5,'Make':'Price ($)':2]


# **Értékek módosítása**

# In[80]:


car_sales


# In[121]:


car_sales.loc[1,'Price ($)'] = 5500


# In[82]:


car_sales.loc[1] = ['Honda', 'Blue', 90000, 4, 6000]


# In[83]:


car_sales['Doors'] = 4


# In[111]:


car_sales.loc[[1,2],['Doors']] = 5


# In[122]:


car_sales


# In[124]:


#utolsó oszlop első három elemének megváltoztatása [4500,6500,7500]-ra
car_sales.loc[[0,1,2],['Price ($)']] = [4500,6500,7500]
car_sales


# ### Feltételes kiválasztás (conditional selection)

# In[119]:


car_sales


# In[88]:


car_sales['Price ($)']> 7000


# In[109]:


car_sales[car_sales['Price ($)']>7000]


# In[90]:


car_sales[car_sales['Price ($)']>7000]['Make']


# In[91]:


car_sales[car_sales['Price ($)']>7000][['Make','Price ($)']]


# **Feladat: Autók adatainak (`Make`, `Odometer KM`) kiválasztása, amelyek 60000 km-nél kevesebbet futottak**

# In[110]:


car_sales[car_sales['Odometer (KM)']<60000][['Make','Odometer (KM)']]


# Két feltétel összekapcsolása | vagy & jellel, zárójelezéssel:

# In[125]:


car_sales[(car_sales['Make'] == 'Toyota') & (car_sales['Price ($)'] < 5000)]


# _____
# ## 5. Hiányzó adatok

# In[126]:


dfm = pd.read_csv('car_sales_missing.csv')


# In[136]:


dfm


# In[128]:


dfm.info()


# In[129]:


dfm.isnull().sum()


# In[130]:


dfm.dropna()


# In[131]:


dfm.dropna(axis=1)


# In[132]:


dfm.dropna(thresh=4)


# In[133]:


dfm.fillna(value='FILL VALUE')


# **Feladat: Hiányzó értékek helyettesítése**
# 
# - `Colour`: leggyakrabban előforduló szín
# - `Doors`: átlag egészre kerekítve
# - `Price`: hiányzó értékhez tartozó típus átlagára  

# In[134]:


#értékek számossága
dfm['Colour'].value_counts()


# In[135]:


#leggyakrabban előforduló érték kiválasztása
dfm['Colour'].value_counts().idxmax()


# In[ ]:





# In[ ]:





# In[137]:


#.isnull  hiányzó értékek keresése
dfm[dfm['Price ($)'].isnull()]


# In[138]:


make = dfm[dfm['Price ($)'].isnull()]['Make']
make


# In[139]:


make.iloc[0]


# In[140]:


dfm[dfm['Make'] == make.iloc[0]]


# In[141]:


dfm[dfm['Make'] == make.iloc[0]]['Price ($)'].mean()


# In[142]:


dfm['Price ($)'].fillna(value=dfm[dfm['Make'] == make.iloc[0]]['Price ($)'].mean(),inplace=True)


# In[143]:


dfm


# _____
# ## 6. Csoportosítás (groupby)
# 
# Adatsorok csoportosítása és aggregációs függvények alkalmazása.

# In[144]:


car_sales


# **Csoportosítsuk az adatokat egy oszlopnév ('Make') szerint a `.groupby()` alkalmazásával. Ez egy `DataFrameGroupBy` objektumot állít elő:**

# In[145]:


car_sales.groupby('Make')


# Ez az objektum új változóként elmenthető:

# In[146]:


by_make = car_sales.groupby('Make')


# In[147]:


by_make


# Az objektumra alkalmazhatjuk az aggregálási műveleteket.

# **Átlag (mean)**

# In[148]:


by_make.mean()


# In[149]:


by_make.mean().astype(int)


# In[150]:


car_sales.groupby('Make').mean()


# **További aggregálási műveletek:**

# In[151]:


by_make.std()


# In[152]:


by_make.min()


# In[153]:


by_make.max()


# In[154]:


by_make.count()


# In[155]:


by_make.describe()


# In[156]:


by_make.describe().loc['Toyota']


# In[157]:


by_make.describe().transpose()


# In[158]:


by_make.describe().transpose()['Toyota']


# _____
# ## 6. Adatok manipulálása

# ### Egyedi értékek

# In[159]:


car_sales


# In[160]:


car_sales['Make'].unique()


# In[161]:


car_sales['Make'].nunique()


# In[162]:


car_sales.value_counts()


# ### Adatok kiválasztása

# In[163]:


cars = car_sales[(car_sales['Price ($)'] > 5000) & (car_sales['Colour'] != 'Blue')]


# In[164]:


cars


# ### Függvények alkalmazása

# In[165]:


def sale(x):
    return x*0.9


# In[166]:


car_sales['Price ($)'].apply(sale)


# In[167]:


car_sales.apply(len)


# In[168]:


car_sales.sum()


# In[169]:


car_sales['Price ($)'].apply(lambda p: p/0.9)


# In[170]:


car_sales


# In[172]:


#Módosítás: változzon meg az ár a DataFrame-ben a lambda függvénynek megfelelően


# ### Oszlop végleges törlése

# In[173]:


del car_sales['Doors']


# In[174]:


car_sales


# ### Új oszlop készítése

# In[175]:


# Pandas Series használatával
doors = pd.Series([4,4,5,4,3,5,4,4,5,4])
car_sales['Doors'] = doors
car_sales


# In[176]:


# Python lista használatával
engine = [2.0,1.3,1.3,4.0,2.0,1.6,2.0,2.0,3.0,1.3]
car_sales['Engine'] = engine
car_sales


# In[177]:


# Másik oszlop(ok)ból
car_sales['Price per KM'] = car_sales['Price ($)'] / car_sales['Odometer (KM)']
car_sales


# In[178]:


# Egy érték használatával
car_sales['Wheels'] = 4
car_sales


# ### DataFrame rendezése (sorting)

# In[179]:


car_sales.sort_values(by='Price ($)')


# In[180]:


# Rendezés Make szerint abc sorrendben és azon belül Price ($) szerint csökkenő sorrendben


# ## Forgatás (pivot)

# In[181]:


car_sales


# In[182]:


car_sales.pivot_table(values='Price ($)',index=['Make','Colour'],columns=['Doors'])


# **Feladat: Forgatással (pivot) készítse el az alábbi táblázatot**
# 
# ![image.png](attachment:image.png)

# In[ ]:





# -----
# ## 7. Adat input és output
# 
# ### Excel

# In[183]:


dfe = pd.read_excel('Golf.xlsx',sheet_name='Sheet1')


# In[184]:


dfe


# In[185]:


dfe.to_excel('Golf_out.xlsx',sheet_name='Sheet1')


# ### CSV

# In[186]:


df = pd.read_csv('goods.csv')
df


# In[187]:


#Index beállítása (Id oszlop)


# In[188]:


df.to_csv('goods_out.csv',index=False)


# **Forgatással (pivot) készítse el az alábbi táblázatot!**
# 
# ![image-4.png](attachment:image-4.png)

# In[ ]:





# ### HTML Input
# 
# A Pandas read_html egy weboldalon levő táblázatot olvas be és DataFrame objektumok listájával tér vissza.
# Például a [Failed Bank List](https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/) weboldalon alkalmazva:

# In[189]:


dfh = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')


# In[190]:


dfh[0]


# In[191]:


df_amk =pd.read_html('https://amk.uni-obuda.hu/szakdolgozat-zarodolgozat-es-diplomamunka/')


# In[192]:


df_amk[0]


# In[193]:


#Header és index beállítása


# ## 8. További hasznos információk a Pandas-ról
# 
# 10 minutes to pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
# 
# Pandas getting started tutorial: https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/index.html
# 
# Essential basic functionality: https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html

# ## 9. Gyakorló feladat
# 1. Készítsen DataFrame-et a `penguins.csv` fájlból
# 2. Jelenítse meg az első 5 sort
# 3. Jelenítse meg az oszlopokat és azok típusait
# 4. Jelenítse meg, hogy hány sora van az adathalmaznak
# 5. Van-e hiányzó érték? Hány hiányzó érték van oszloponként?
# 6. Törölje a hiányzó értékeket tartalmazó sorokat
# 7. Mennyi a legnehezebb és a legkönnyebb pingvin tömege?
# 8. Mennyi a legnehezebb és legkönnyebb 'Adelie' pingvin tömege?
# 9. Jelenítse meg a pingvinek átlagos tömegét fajtánként és nemenként
# 10. Jelenítse meg a pingvinek számát fajtánként és szigetenként

# In[ ]:





# In[ ]:




