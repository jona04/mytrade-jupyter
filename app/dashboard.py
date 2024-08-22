import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from streamlit_plotly_events import plotly_events

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_ta as ta
import os

# Function to read the CSV and return the dataframe
def read_csv_to_dataframe(file_path):
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    
    # Ensure that the 'Open', 'High', 'Low', 'Close' columns are numeric
    df = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)
    
    return df



# Função para analisar o comportamento com valores e novos parâmetros
def analyze_trends_with_values(data, long_sma=200, short_sma=50, bollinger_window=40, adx_window=30, rsi_window=30):
    analysis = []

    # Pega a última data e hora do índice
    last_datetime = data.index[-1].strftime("%Y-%m-%d %H:%M")
    last_day_of_week = data.index[-1].strftime("%A")

    # Analisando Médias Móveis
    sma_short_current = data[f'SMA_{short_sma}'].iloc[-1]
    sma_long_current = data[f'SMA_{long_sma}'].iloc[-1]

    if sma_short_current > data[f'SMA_{short_sma}'].iloc[-2]:
        analysis.append(f"A média móvel de {short_sma} períodos está subindo, com valor atual de {sma_short_current:.2f}.")
    else:
        analysis.append(f"A média móvel de {short_sma} períodos está descendo, com valor atual de {sma_short_current:.2f}.")
    
    if sma_long_current > data[f'SMA_{long_sma}'].iloc[-2]:
        analysis.append(f"A média móvel de {long_sma} períodos está subindo, com valor atual de {sma_long_current:.2f}.")
    else:
        analysis.append(f"A média móvel de {long_sma} períodos está descendo, com valor atual de {sma_long_current:.2f}.")
    
    # Cruzamento de Médias Móveis
    if sma_short_current > sma_long_current:
        analysis.append(f"A média móvel de {short_sma} períodos cruzou acima da de {long_sma}, indicando uma possível tendência de alta.")
    else:
        analysis.append(f"A média móvel de {short_sma} períodos está abaixo da de {long_sma}, indicando uma possível tendência de baixa.")
    
    # Analisando Bandas de Bollinger
    close_current = data['Close'].iloc[-1]
    upper_band_current = data[f'BBU_{bollinger_window}_2.0'].iloc[-1]
    lower_band_current = data[f'BBL_{bollinger_window}_2.0'].iloc[-1]
    
    if pd.notna(upper_band_current) and pd.notna(lower_band_current):
        if close_current > upper_band_current:
            analysis.append(f"O preço está acima da banda superior de Bollinger (valor atual: {close_current:.2f}), com a banda superior em {upper_band_current:.2f}.")
        elif close_current < lower_band_current:
            analysis.append(f"O preço está abaixo da banda inferior de Bollinger (valor atual: {close_current:.2f}), com a banda inferior em {lower_band_current:.2f}.")
        else:
            analysis.append(f"O preço está dentro das Bandas de Bollinger (valor atual: {close_current:.2f}).")
    else:
        analysis.append("As Bandas de Bollinger não estão disponíveis para a análise.")

    # Volatilidade nas Bandas de Bollinger
    if pd.notna(upper_band_current) and pd.notna(lower_band_current):
        band_width = upper_band_current - lower_band_current
        previous_band_width = data[f'BBU_{bollinger_window}_2.0'].iloc[-2] - data[f'BBL_{bollinger_window}_2.0'].iloc[-2]
        if band_width > previous_band_width:
            analysis.append(f"As Bandas de Bollinger ({bollinger_window} períodos) estão se alargando, indicando aumento na volatilidade.")
        else:
            analysis.append(f"As Bandas de Bollinger ({bollinger_window} períodos) estão se estreitando, indicando diminuição na volatilidade.")
    
    # Analisando ADX
    adx_current = data[f'ADX_{adx_window}'].iloc[-1]
    if adx_current > 25:
        analysis.append(f"A tendência é forte, com ADX ({adx_window} períodos) em {adx_current:.2f}.")
    else:
        analysis.append(f"A tendência é fraca ou inexistente, com ADX ({adx_window} períodos) em {adx_current:.2f}.")

    # Analisando RSI
    rsi_current = data[f'RSI_{rsi_window}'].iloc[-1]
    if rsi_current > 70:
        analysis.append(f"O RSI ({rsi_window} períodos) indica que o mercado está sobrecomprado, com valor de {rsi_current:.2f}.")
    elif rsi_current < 30:
        analysis.append(f"O RSI ({rsi_window} períodos) indica que o mercado está sobrevendido, com valor de {rsi_current:.2f}.")
    else:
        analysis.append(f"O RSI ({rsi_window} períodos) está em uma zona neutra, com valor de {rsi_current:.2f}.")

    # Divergência RSI e Preço
    previous_close = data['Close'].iloc[-2]
    previous_rsi = data[f'RSI_{rsi_window}'].iloc[-2]
    if (close_current > previous_close and rsi_current < previous_rsi) or (close_current < previous_close and rsi_current > previous_rsi):
        analysis.append("Há uma divergência entre o RSI e o preço, sugerindo uma possível reversão.")

    # Retornando a análise como uma string com data, hora, dia da semana e informações de janela
    return f"Data e Hora da análise: {last_datetime} ({last_day_of_week}). " + " ".join(analysis)



# Function to create and save a candlestick chart with Bollinger Bands, ADX, and RSI
def plot_candlestick_with_indicators(data, 
                                    start_date,
                                    end_date,
                                    long_sma=200, 
                                    short_sma=50,
                                    bollinger_window=40, 
                                    adx_window=30, 
                                    rsi_window=30):
    # Cálculo das Médias Móveis, Bandas de Bollinger, ADX e RSI usando pandas_ta
    data[f'SMA_{short_sma}'] = ta.sma(data['Close'], length=short_sma)
    data[f'SMA_{long_sma}'] = ta.sma(data['Close'], length=long_sma)
    bollinger_bands = data.ta.bbands(close=data['Close'], length=bollinger_window, std=2)
    data = pd.concat([data, bollinger_bands], axis=1)
    data[f'ADX_{adx_window}'] = ta.adx(data['High'], data['Low'], data['Close'], length=adx_window)[f'ADX_{adx_window}']
    data[f'RSI_{rsi_window}'] = ta.rsi(data['Close'], length=rsi_window)

    data = data.loc[start_date:end_date]

    # Criando o subplot com 3 linhas, ajustando a altura dos subplots
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.6, 0.2, 0.2], 
                        subplot_titles=("Candlestick com Médias Móveis e Bandas de Bollinger", "ADX", "RSI"))

    # Preço de fechamento em formato de Candlestick (primeiro subplot)
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'), row=1, col=1)

    # Médias móveis (primeiro subplot)
    fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{short_sma}'], mode='lines', name=f'SMA_{short_sma}', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{long_sma}'], mode='lines', name=f'SMA_{long_sma}', line=dict(color='purple')), row=1, col=1)

    # Bandas de Bollinger (primeiro subplot)
    fig.add_trace(go.Scatter(x=data.index, y=data[f'BBU_{bollinger_window}_2.0'], mode='lines', name='Upper Band', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data[f'BBM_{bollinger_window}_2.0'], mode='lines', name='Middle Band', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data[f'BBL_{bollinger_window}_2.0'], mode='lines', name='Lower Band', line=dict(color='green', dash='dash')), row=1, col=1)

    # ADX no segundo subplot
    fig.add_trace(go.Scatter(x=data.index, y=data[f'ADX'], mode='lines', name='ADX', line=dict(color='blue')), row=2, col=1)

    # RSI no terceiro subplot
    fig.add_trace(go.Scatter(x=data.index, y=data[f'RSI'], mode='lines', name='RSI', line=dict(color='green')), row=3, col=1)

    # Linhas de sobrecompra e sobrevenda para o RSI
    fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines', name='Overbought', line=dict(color='red', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines', name='Oversold', line=dict(color='red', dash='dash')), row=3, col=1)

    # Configurações do layout com zoom de caixa ajustado para ambos os eixos e tamanhos ajustados dos subplots
    fig.update_layout(
        title='Grafico com Médias Móveis, Bandas de Bollinger, ADX e RSI',
        xaxis_title='Time',
        yaxis_title='Price',
        dragmode='zoom',  # Habilitar o modo de zoom de caixa
        xaxis_rangeslider_visible=False,  # Remover o slider do eixo x
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Habilitar zoom vertical para o gráfico principal
        yaxis2=dict(scaleanchor="x", scaleratio=1),  # Habilitar zoom vertical para o gráfico do ADX
        yaxis3=dict(scaleanchor="x", scaleratio=1),  # Habilitar zoom vertical para o gráfico do RSI
        height=1000,  # Ajustar altura para acomodar os subplots
    )

    # retorna o gráfico
    return fig


# Function to plot Candlestick with Moving Averages and Bollinger Bands
def plot_candlestick_with_bollinger(data, long_sma=200, short_sma=50, bollinger_window=40):


    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'))

    # Short and long moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{short_sma}'], mode='lines', name=f'SMA_{short_sma}', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{long_sma}'], mode='lines', name=f'SMA_{long_sma}', line=dict(color='purple')))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data[f'BBU_{bollinger_window}_2.0'], mode='lines', name='Upper Band', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data[f'BBM_{bollinger_window}_2.0'], mode='lines', name='Middle Band', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data[f'BBL_{bollinger_window}_2.0'], mode='lines', name='Lower Band', line=dict(color='green', dash='dash')))

    fig.update_layout(title='Candlestick with Moving Averages and Bollinger Bands',
                      xaxis_title='Time', yaxis_title='Price', dragmode='zoom')

    return fig

# Function to plot ADX
def plot_adx(data, adx_window=30):

    fig = go.Figure()

    # ADX line
    fig.add_trace(go.Scatter(x=data.index, y=data[f'ADX_{adx_window}'], mode='lines', name='ADX', line=dict(color='blue')))

    fig.update_layout(title='ADX Indicator', xaxis_title='Time', yaxis_title='ADX', dragmode='zoom')

    return fig

# Function to plot RSI
def plot_rsi(data, rsi_window=30):

    fig = go.Figure()

    # RSI line
    fig.add_trace(go.Scatter(x=data.index, y=data[f'RSI_{rsi_window}'], mode='lines', name='RSI', line=dict(color='green')))

    # Overbought and Oversold lines
    fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines', name='Overbought', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines', name='Oversold', line=dict(color='red', dash='dash')))

    fig.update_layout(title='RSI Indicator', xaxis_title='Time', yaxis_title='RSI', dragmode='zoom')

    return fig

def generate_bollinger_bands_data(data, long_sma=200, short_sma=50, bollinger_window=40):
    data[f'SMA_{short_sma}'] = ta.sma(data['Close'], length=short_sma)
    data[f'SMA_{long_sma}'] = ta.sma(data['Close'], length=long_sma)
    bollinger_bands = data.ta.bbands(close=data['Close'], length=bollinger_window, std=2)
    data = pd.concat([data, bollinger_bands], axis=1)
    return data

def generate_adx_data(data, adx_window=30):
    data[f'ADX_{adx_window}'] = ta.adx(data['High'], data['Low'], data['Close'], length=adx_window)[f'ADX_{adx_window}']
    return data

def generate_rsi_data(data, rsi_window=30):
    data[f'RSI_{rsi_window}'] = ta.rsi(data['Close'], length=rsi_window)
    return data


# Sidebar
st.title('Stock Dashboard')
ticker = st.sidebar.text_input('Ticker', value='XAUUSD')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-12-31'))

#Gera dados
df1d = read_csv_to_dataframe('../XAUUSD_D1.csv')
data = generate_bollinger_bands_data(df1d)
data = generate_adx_data(data)
data = generate_rsi_data(data)
data = data.loc[start_date: end_date]

# Plotando os gráficos
candlestick_fig = plot_candlestick_with_bollinger(data)
adx_fig = plot_adx(data)
rsi_fig = plot_rsi(data)

# Define container
with st.container(height=500):
    selected_points = plotly_events(candlestick_fig,click_event=True)
    st.plotly_chart(adx_fig, use_container_width=True)
    st.plotly_chart(rsi_fig, use_container_width=True)


# Capitura eventos
if selected_points:
    data_point = data.iloc[selected_points[0]['pointIndex']]
    st.write(data_point)

# Gerando a análise do gráfico com valores dos indicadores
text_analysis = analyze_trends_with_values(data)

# Mostrando a análise gerada
st.write(text_analysis)
