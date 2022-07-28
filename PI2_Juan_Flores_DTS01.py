import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import urllib.request 
from pprint import pprint 
from html_table_parser import HTMLTableParser
import yfinance as yf
import matplotlib.dates as mpdates
from mpl_finance import candlestick_ohlc
sns.set()
'''
Programa creado por Juan Gerardo Flores Hernández
contacto: gerardo3070@ciencias.unam.mx

'''
print('DISCLAIMER: NO SON RECOMENDACIONES DE INVERSIÓN')
def url_get_contents(url):
    req = urllib.request.Request(url = url)
    f = urllib.request.urlopen(req)
    return f.read()
xhtml = url_get_contents('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'). decode('utf-8')
p = HTMLTableParser()
p.feed(xhtml)
claves=p.tables[0]
keys=pd.DataFrame(claves)
columns=keys.iloc[0,:]
keys.columns=columns
keys.drop(0, inplace=True)
keys.reset_index(inplace=True)
keys.drop('index', inplace=True, axis=1)
keys['Symbol'] = keys.apply(lambda row: str(row['Symbol']).replace('.','-'), axis=1)
for i in range(len(keys.Symbol)):
    globals()[keys.Symbol[i]] = yf.download(keys.Symbol[i], start='2000-01-01', end='2021-12-31')
    globals()[keys.Symbol[i]].reset_index(inplace=True)
    if globals()[keys.Symbol[i]].size==0:
        globals()[keys.Symbol[i]] = yf.download(keys.Symbol[i], start='2000-01-01')
        globals()[keys.Symbol[i]].reset_index(inplace=True)
    print('Ingesta correcta del activo: ', keys.Symbol[i])
    globals()[keys.Symbol[i]].Close.fillna(globals()[keys.Symbol[i]].Open.shift(1), inplace=True)
    globals()[keys.Symbol[i]]['Year'] = globals()[keys.Symbol[i]]['Date'].dt.year
    globals()[keys.Symbol[i]]['Month'] = globals()[keys.Symbol[i]]['Date'].dt.month
    globals()[keys.Symbol[i]]['Weekday'] = globals()[keys.Symbol[i]]['Date'].dt.day_name()
    globals()[keys.Symbol[i]]['Gap_returns'] = np.log(globals()[keys.Symbol[i]].Open/globals()[keys.Symbol[i]].Close.shift(1)).fillna(0)
    globals()[keys.Symbol[i]]['Intra_returns'] = np.log(globals()[keys.Symbol[i]].Close/globals()[keys.Symbol[i]].Open).fillna(0)
    globals()[keys.Symbol[i]]['Variations'] = globals()[keys.Symbol[i]]['Adj Close'].pct_change()
    globals()[keys.Symbol[i]]['Volatility'] = globals()[keys.Symbol[i]].Variations.rolling(50).std()*100*(50)**0.5
    globals()[keys.Symbol[i]]['MA55'] = globals()[keys.Symbol[i]].Close.rolling(55).mean()
    globals()[keys.Symbol[i]]['MA10'] = globals()[keys.Symbol[i]].Close.rolling(10).mean()
SP500=globals()[keys.Symbol[0]]
for i in range(1, len(keys.Symbol)):
    SP500.append(globals()[keys.Symbol[i]], ignore_index=True)
days=globals()[keys.Symbol[0]].Weekday.unique()
gap=[]
diff=[]
volat=[]
intra=[]
for elem in days:
    gap.append(SP500[SP500.Weekday==elem]['Gap_returns'].sum())
    diff.append((SP500[SP500.Weekday==elem]['Close']-SP500[SP500.Weekday==elem]['Open']).sum())
    volat.append(SP500[SP500.Weekday==elem]['Volatility'].sum())
    intra.append(SP500[SP500.Weekday==elem]['Intra_returns'].sum())
intra2=np.multiply(intra,intra)
intra2=list(intra2)
colors = pd.cut(diff, bins=[np.NINF, 0, np.inf], labels=['red', 'green'], right=False)
plt.scatter(gap, diff, color=colors)
plt.legend()
plt.plot([np.min(gap)-1,np.max(gap)+1], [0,0], color='red')
for i, label in enumerate(days):
    plt.text(gap[i], diff[i], label)
plt.xlim(np.min(gap)-1,np.max(gap)+1)
plt.xlabel('Gap Return')
plt.ylabel('Day Price Variation (U$D)')
plt.title('Day Price Variation per Total Gap Return');
plt.scatter(volat, diff, color=colors)
plt.legend()
plt.plot([np.min(volat)-3000,np.max(volat)+3000], [0,0], color='red')
for i, label in enumerate(days):
    plt.text(gap[i], diff[i], label)
plt.xlim(np.min(volat)-3000,np.max(volat)+3000)
plt.xlabel('Volatility')
plt.ylabel('Day Price Variation (U$D)')
plt.title('Day Price Variation per Total Volatility');
if days[volat.index(np.max(volat))] == days[gap.index(np.max(gap))]:
    print('The best day to invest, considering Total Gap Return, is: ', days[gap.index(np.max(gap))])
else:
    pos=[]
    for i in range(len(gap)):
        if gap[i]<abs((np.max(gap)-np.min(gap))/len(gap)):
            pos.append(gap[i])
    print('The best day to invest, considering Total Gap Return, is: ', days[gap.index(np.min(pos))])
plt.scatter(intra, diff, color=colors)
plt.legend()
plt.plot([np.min(intra)-1,np.max(intra)+1], [0,0], color='red')
for i, label in enumerate(days):
    plt.text(intra[i], diff[i], label)
plt.xlim(np.min(intra)-1,np.max(intra)+1)
plt.xlabel('Intraday Return')
plt.ylabel('Day Price Variation (U$D)')
plt.title('Day Price Variation per Total Intraday Return');
plt.scatter(volat, diff, color=colors)
plt.legend()
plt.plot([np.min(volat)-3000,np.max(volat)+3000], [0,0], color='red')
for i, label in enumerate(days):
    plt.text(gap[i], diff[i], label)
plt.xlim(np.min(volat)-3000,np.max(volat)+3000)
plt.xlabel('Volatility')
plt.ylabel('Day Price Variation (U$D)')
plt.title('Day Price Variation per Total Volatility');
if days[volat.index(np.max(volat))] == days[intra.index(np.max(intra))]:
    print('The best day to invest, considering Total Intradays value, is: ', days[intra.index(np.max(intra))])
else:
    print('The best day to invest considering Total Intradays value, is: ', days[intra2.index(np.max(intra2))])
plt.scatter(volat, diff, color=colors)
plt.legend()
plt.plot([np.min(volat)-3000,np.max(volat)+3000], [0,0], color='red')
for i, label in enumerate(days):
    plt.text(gap[i], diff[i], label)
plt.xlim(np.min(volat)-3000,np.max(volat)+3000)
plt.xlabel('Volatility')
plt.ylabel('Day Price Variation (U$D)')
plt.title('Day Price Variation per Total Volatility');
print('The best day to invest, considering Total Volatility, is: ', days[volat.index(np.max(volat))])
sector=keys['GICS Sector'].value_counts()
indust=sector.index.values
prim=[indust[0]]
for i in range(1, len(sector)-1):
    if sector[0] - sector[i] <= 5:
        prim.append(indust[i])
print('The best industries to invest in are : ')
for elem in prim:
    print(elem)
S_P = yf.download('^GSPC', start='2000-01-01', end='2021-12-31')
S_P.reset_index(inplace=True)
print('Ingesta correcta del activo: ', 'S_P')
S_P.Close.fillna(S_P.Open.shift(1), inplace=True)
S_P['Variations'] = S_P['Adj Close'].pct_change()
S_P['Volatility'] = S_P.Variations.rolling(50).std()*100*(50)**0.5
S_P['MA55'] = S_P.Close.rolling(55).mean()
S_P['MA10'] = S_P.Close.rolling(10).mean()
S_P['Date2']=S_P['Date']
S_P['Date'] = S_P['Date'].map(mpdates.date2num) 
fig = plt.figure() 
fig.set_figheight(10) 
fig.set_figwidth(10) 
ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), rowspan=3, colspan=4) 
ax2 = plt.subplot2grid(shape=(4, 4), loc=(3, 0), rowspan=1, colspan=4) 
candlestick_ohlc(ax1, S_P.values, width = 0.6, 
                 colorup = 'green', colordown = 'red',  
                 alpha = 0.8)
ax1.plot(S_P.Date, S_P.MA55, color='orange', label='MA55')
ax1.plot(S_P.Date, S_P.MA10, color='purple', alpha=0.70, label='MA10') 
ax1.grid(True) 
ax1.set_xlabel('Date') 
ax1.set_ylabel('Price (U$D)') 
plt.suptitle('Price Candlestick Chart',fontsize=20)
date_format = mpdates.DateFormatter('%d-%m-%Y') 
ax1.xaxis.set_major_formatter(date_format) 
fig.autofmt_xdate() 
fig.tight_layout()
ax2.plot(S_P.Date, S_P.Volatility, label='Volatility')
ax2.xaxis.set_major_formatter(date_format) 
ax2.plot(S_P.Date, np.full(len(S_P.Date), S_P.Volatility.mean()+7), color='r')
ax2.set_ylabel('Volatility') 
ax2.set_xlabel('Date (dd-mm-yyyy)') 
lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
fig.legend(lines, labels, loc = 'center right')
plt.tight_layout() 
plt.show()
S_P['Month'] = S_P['Date2'].dt.month
S_P_imp=S_P[S_P['Volatility']>=S_P.Volatility.mean()+7]
S_P_imp['Year'] = S_P_imp['Date2'].dt.year
years=S_P_imp.Year.unique()
print('The moments with more Volatility were:')
for elem in years:
    print('Month:', round(S_P_imp[S_P_imp['Year']==elem].Month.mean()), 'Year: ', elem)
growth=[]
for i in range(len(keys.Symbol)):
    growth.append(globals()[keys.Symbol[i]]['Adj Close'][len(globals()[keys.Symbol[i]]['Adj Close'])-1]-globals()[keys.Symbol[i]].Close[0])
company = pd.DataFrame(list(zip(keys.Security,growth)), columns = ['Security','Total_Growth'])
company.sort_values(by=['Total_Growth'], ascending=False, inplace=True)
company.reset_index(inplace=True)
company2 = pd.DataFrame(list(zip(keys.Symbol,growth)), columns = ['Symbol','Total_Growth'])
company2.sort_values(by=['Total_Growth'], ascending=False, inplace=True)
company2.reset_index(inplace=True)
print('The best 9 companies to invest in are:')
for i in range(9):
    print(i+1, '.-', company.Security[i], '-', company2.Symbol[i], sep='')
for i in range(9):
    globals()[company2.Symbol[i]]['Date'] = globals()[company2.Symbol[i]]['Date'].map(mpdates.date2num)
for i in range(9):
    print('-------------------------------', i+1, '.-', company.Security[i], '-', company2.Symbol[i], '-------------------------------', sep='')
    fig = plt.figure() 
    fig.set_figheight(10) 
    fig.set_figwidth(10) 
    ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), rowspan=3, colspan=4) 
    ax2 = plt.subplot2grid(shape=(4, 4), loc=(3, 0), rowspan=1, colspan=4) 
    candlestick_ohlc(ax1, globals()[company2.Symbol[i]].values, width = 0.6, 
                    colorup = 'green', colordown = 'red',  
                    alpha = 0.8)
    ax1.grid(True) 
    ax1.set_xlabel('Date') 
    ax1.set_ylabel('Price (U$D)') 
    plt.suptitle('Price Candlestick Chart',fontsize=20)
    date_format = mpdates.DateFormatter('%d-%m-%Y') 
    ax1.xaxis.set_major_formatter(date_format) 
    fig.autofmt_xdate() 
    fig.tight_layout()
    ax2.plot(globals()[company2.Symbol[i]].Date, globals()[company2.Symbol[i]].Volatility, label='Volatility')
    ax2.xaxis.set_major_formatter(date_format) 
    ax2.plot(globals()[company2.Symbol[i]].Date, np.full(len(globals()[company2.Symbol[i]].Date), globals()[company2.Symbol[i]].Volatility.mean()+7), color='r')
    ax2.set_ylabel('Volatility') 
    ax2.set_xlabel('Date (dd-mm-yyyy)') 
    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    fig.legend(lines, labels, loc = 'center right')
    plt.tight_layout() 
    plt.show()
    