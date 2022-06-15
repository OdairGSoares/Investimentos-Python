import mplfinance as mpf
import pandas
from matplotlib import animation

indice=0

lista=[]

def animar(i):

    global indice
    global lista

    info=pandas.read_csv("./data/WIN.csv")

    if indice==0:
        lista.append(info.loc[indice])
    elif info.loc[indice]['Date']!=info.loc[indice-1]['Date']:
        lista.append(info.loc[indice-1])

    if indice==1:
        lista.pop(0)

    ultimo_close_minuto=pandas.DataFrame(lista,columns=['Date','Open','High','Low','Close'])

    ultimo_close_minuto=ultimo_close_minuto.set_index(pandas.DatetimeIndex(ultimo_close_minuto['Date']))

    ax1.clear()

    indice=indice+1

    mpf.plot(ultimo_close_minuto,ax=ax1, type = "candle",style='yahoo')
    
fig=mpf.figure(style="yahoo",figsize=(7,8))

ax1 = fig.add_subplot(1,1,1)

ani=animation.FuncAnimation(fig,animar,interval=0.500)

mpf.show()
